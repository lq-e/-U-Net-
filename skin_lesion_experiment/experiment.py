"""Train and evaluate two compact U-Nets on ISIC 2016, on CPU."""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.nn import functional as F

from vendor.unet_parts import DoubleConv, Down, OutConv, Up

METHODS = {"baseline": 1.0, "boundary_weighted": 3.0}
METRICS = ("dice", "iou", "boundary_f1")


class SmallUNet(nn.Module):
    """Three-level assembly using the GitHub Pytorch-UNet building blocks."""

    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(3, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 32)
        self.up1 = Up(64, 16, bilinear=True)
        self.up2 = Up(32, 8, bilinear=True)
        self.up3 = Up(16, 8, bilinear=True)
        self.outc = OutConv(8, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


def loss_function(logits, target, boundary_weight):
    pixel_bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if boundary_weight > 1:
        # A one-pixel dilation minus erosion forms a band across the contour.
        band = F.max_pool2d(target, 3, stride=1, padding=1)
        band += F.max_pool2d(-target, 3, stride=1, padding=1)
        weights = 1 + (boundary_weight - 1) * band
        bce = (pixel_bce * weights).sum((1, 2, 3)) / weights.sum((1, 2, 3))
    else:
        bce = pixel_bce.mean((1, 2, 3))
    probabilities = logits.sigmoid().flatten(1)
    labels = target.flatten(1)
    intersection = (probabilities * labels).sum(1)
    soft_dice = (2 * intersection + 1e-6) / (probabilities.sum(1) + labels.sum(1) + 1e-6)
    return (bce + (1 - soft_dice)).mean()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_split(cache, split):
    ids = cache[split + "_id"]
    rgb = torch.from_numpy(cache[split + "_x"]).permute(0, 3, 1, 2)
    masks = torch.from_numpy(cache[split + "_y"])[:, None]
    return ids, rgb, masks


def inputs(rgb):
    return rgb.to(dtype=torch.float32, memory_format=torch.channels_last) / 255.0


@torch.inference_mode()
def predict(model, images, batch_size):
    model.eval()
    return torch.cat([model(inputs(batch)) >= 0 for batch in images.split(batch_size)])


def dice_scores(prediction, target):
    pred, true = prediction.flatten(1), target.flatten(1)
    total = pred.sum(1) + true.sum(1)
    return torch.where(total > 0, 2 * (pred & true).sum(1) / total, 1.0)


def train(method, weight, images, masks, train_indices, val_indices, args):
    # Both methods restart from identical weights and use the same minibatches and flips.
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    model = SmallUNet().to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.0001)
    best, history = -1.0, []
    checkpoint_path = args.out / f"{method}.pt"
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        order = train_indices[torch.randperm(len(train_indices), generator=rng)]
        for batch in order.split(args.batch):
            rgb, target = inputs(images[batch]), masks[batch].float()
            for axis in (2, 3):
                if torch.rand((), generator=rng) < 0.5:
                    rgb, target = rgb.flip(axis), target.flip(axis)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(rgb), target, weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        validation_dice = dice_scores(
            predict(model, images[val_indices], args.batch), masks[val_indices]
        ).mean().item()
        row = {"method": method, "epoch": epoch, "train_loss": total_loss / len(train_indices),
               "validation_dice": validation_dice, "lr": optimizer.param_groups[0]["lr"],
               "elapsed_s": round(time.perf_counter() - start, 2)}
        history.append(row)
        if validation_dice > best:
            best = validation_dice
            torch.save({"model": model.state_dict(), "method": method, "epoch": epoch,
                        "validation_dice": best, "seed": args.seed,
                        "boundary_weight": weight, "input_size": 128}, checkpoint_path)
        write_csv(args.out / f"history_{method}.csv", history)
        scheduler.step()
        print(f"{method} {epoch:02}/{args.epochs} loss={row['train_loss']:.4f} "
              f"val_dice={validation_dice:.4f} elapsed={row['elapsed_s']:.1f}s", flush=True)
    return history[-1]["elapsed_s"]


def boundary_f1(pred, true, tolerance=2):
    """Inner contours matched by Euclidean distance at 128x128 resolution."""
    pred_edge = pred & ~ndimage.binary_erosion(pred, structure=np.ones((3, 3)))
    true_edge = true & ~ndimage.binary_erosion(true, structure=np.ones((3, 3)))
    if not pred_edge.any() or not true_edge.any():
        return float(not pred_edge.any() and not true_edge.any())
    precision = (ndimage.distance_transform_edt(~true_edge)[pred_edge] <= tolerance).mean()
    recall = (ndimage.distance_transform_edt(~pred_edge)[true_edge] <= tolerance).mean()
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def evaluate(ids, rgb, masks, args):
    target = masks[:, 0].numpy()
    predictions = {}
    rows = [{"id": case, "target_fraction": float(mask.mean())} for case, mask in zip(ids, target)]
    summary = {}
    for method in METHODS:
        saved = torch.load(args.out / f"{method}.pt", weights_only=True, map_location="cpu")
        model = SmallUNet().to(memory_format=torch.channels_last)
        model.load_state_dict(saved["model"])
        output = predict(model, rgb, args.batch)[:, 0].numpy()
        predictions[method] = output
        scores = []
        for row, pred, true in zip(rows, output, target):
            inter, union = (pred & true).sum(), (pred | true).sum()
            case = (2 * inter / (pred.sum() + true.sum()), inter / union, boundary_f1(pred, true))
            scores.append(case)
            row.update({f"{method}_{key}": float(value) for key, value in zip(METRICS, case)})
        summary[method] = dict(zip(METRICS, np.mean(scores, axis=0).tolist()))
        summary[method].update(best_epoch=saved["epoch"], validation_dice=saved["validation_dice"])
    write_csv(args.out / "test_per_image.csv", rows)
    # Paired resampling of test images, conditional on these two trained checkpoints.
    rng = np.random.default_rng(2026)
    bootstrap = rng.integers(0, len(rows), size=(5000, len(rows)))
    changes = {}
    for key in METRICS:
        delta = np.array([row[f"boundary_weighted_{key}"] - row[f"baseline_{key}"] for row in rows])
        changes[key] = {"mean": float(delta.mean()),
                        "ci95": np.quantile(delta[bootstrap].mean(1), [.025, .975]).tolist(),
                        "better": int((delta > 0).sum()), "worse": int((delta < 0).sum())}
    return summary, changes, predictions, rows


def draw(rgb, masks, predictions, rows, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pick cases at fixed quantiles of baseline Dice plus largest gain and loss.
    dice = np.array([row["baseline_dice"] for row in rows])
    diff = np.array([row["boundary_weighted_dice"] - row["baseline_dice"] for row in rows])
    rank = dice.argsort()
    picks = [rank[round(q * (len(rows) - 1))] for q in (0.1, 0.5, 0.9)]
    picks += [diff.argmax(), diff.argmin()]
    labels = ("10th percentile", "Median", "90th percentile", "Largest gain", "Largest loss")
    titles = ("Input", "Reference mask", "Baseline", "Boundary-weighted")
    fig, axes = plt.subplots(5, 4, figsize=(9, 11.3))
    for row_index, (index, label) in enumerate(zip(picks, labels)):
        row = rows[index]
        tiles = [rgb[index].permute(1, 2, 0).numpy(), masks[index, 0].numpy(),
                 predictions["baseline"][index], predictions["boundary_weighted"][index]]
        for col, tile in enumerate(tiles):
            ax = axes[row_index, col]
            ax.imshow(tile, cmap="gray", vmin=0 if col else None, vmax=1 if col else None)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index == 0:
                ax.set_title(titles[col], fontsize=11)
            if col >= 2:
                method = list(METHODS)[col - 2]
                ax.set_xlabel(f'Dice {row[f"{method}_dice"]:.3f}', fontsize=9)
        axes[row_index, 0].set_xlabel(f'{row["id"]} | {label}', fontsize=8)
    fig.suptitle("ISIC 2016 test examples", fontsize=15)
    fig.tight_layout()
    fig.savefig(args.out / "predictions.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    for method in METHODS:
        with (args.out / f"history_{method}.csv").open() as handle:
            history = list(csv.DictReader(handle))
        ax.plot([int(row["epoch"]) for row in history],
                [float(row["validation_dice"]) for row in history],
                marker=".", label=method.replace("_", " ").title())
    ax.set(xlabel="Epoch", ylabel="Validation Dice")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "validation_curves.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/isic2016_128.npz"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with np.load(args.cache) as cache:
        train_ids, images, masks = load_split(cache, "training")
        test_ids, test_rgb, test_masks = load_split(cache, "test")
        audits = json.loads(cache["duplicate_audit"].item())
    order = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(args.seed))
    val_indices, train_indices = order[:90], order[90:]
    split = dict(train=train_ids[train_indices.numpy()].tolist(),
                 validation=train_ids[val_indices.numpy()].tolist(), test=test_ids.tolist())
    if args.evaluate_only:
        existing = json.loads((args.out / "split.json").read_text())
        if existing != split:
            raise ValueError("Dataset or seed differs from the recorded split")
        timings = json.loads((args.out / "summary.json").read_text())["training_seconds"]
    else:
        (args.out / "split.json").write_text(json.dumps(split, indent=2))
        timings = {method: train(method, weight, images, masks, train_indices, val_indices, args)
                   for method, weight in METHODS.items()}
    summary, changes, predictions, rows = evaluate(test_ids, test_rgb, test_masks, args)
    report = {"models": summary, "paired_changes": changes, "training_seconds": timings,
              "parameters_per_model": sum(p.numel() for p in SmallUNet().parameters()),
              "seed": args.seed, "batch": args.batch, "threads": args.threads,
              "exact_training_duplicates_removed": audits,
              "python": platform.python_version(), "torch": str(torch.__version__),
              "split_sizes": {name: len(ids) for name, ids in split.items()}}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2))
    draw(test_rgb, test_masks, predictions, rows, args)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
