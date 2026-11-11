"""Build a small local cache from the official ISIC 2016 Part 3B ZIPs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from io import BytesIO
from pathlib import Path, PurePosixPath
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
from PIL import Image

BASE = "https://isic-archive.s3.amazonaws.com/challenges/2016/"
SIZE = 128


def read_pair(archive, image_path, mask_path):
    image_bytes = archive.read(image_path)
    with Image.open(BytesIO(image_bytes)) as image, Image.open(BytesIO(archive.read(mask_path))) as mask:
        if image.size != mask.size:
            raise ValueError(f"Image and mask sizes differ for {image_path}")
        rgb = np.asarray(image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR))
        label = np.asarray(mask.convert("L").resize((SIZE, SIZE), Image.Resampling.NEAREST)) >= 128
    if not label.any() or label.all():
        raise ValueError(f"Unexpected empty or full mask for {image_path}")
    return rgb, label, hashlib.sha256(image_bytes).hexdigest()


def load_archive(path, expected):
    with ZipFile(path) as archive:
        members = {PurePosixPath(name).name: name for name in archive.namelist()}
        ids = sorted(name[:-4] for name in members if name.startswith("ISIC_") and name.endswith(".jpg"))
        if len(ids) != expected:
            raise ValueError(f"Expected {expected} pairs in {path}, got {len(ids)}")
        with ThreadPoolExecutor(max_workers=4) as workers:
            jobs = [workers.submit(read_pair, archive, members[f"{case}.jpg"],
                                   members[f"{case}_Segmentation.png"]) for case in ids]
            x, y, hashes = zip(*(job.result() for job in jobs))
    return np.array(ids), np.stack(x), np.stack(y), hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    arrays, audits, test_ids, test_lookup = {}, [], set(), {}
    for split, count in [("Test", 379), ("Training", 900)]:
        path = args.data / f"{split}.zip"
        if not path.exists():
            url = BASE + f"ISBI2016_ISIC_Part3B_{split}_Data.zip"
            print(f"Downloading {url}", flush=True)
            temporary = path.with_suffix(".download")
            urlretrieve(url, temporary)
            temporary.replace(path)
        ids, images, masks, hashes = load_archive(path, count)
        if split == "Test":
            test_ids = set(ids)
            test_lookup = dict(zip(hashes, ids))
        else:
            keep = []
            for index, (case, fingerprint) in enumerate(zip(ids, hashes)):
                if case in test_ids or fingerprint in test_lookup:
                    audits.append({"removed_training_id": str(case),
                                   "matching_test_id": str(test_lookup.get(fingerprint, case))})
                else:
                    keep.append(index)
            ids, images, masks = ids[keep], images[keep], masks[keep]
        key = split.lower()
        arrays.update({key + "_id": ids, key + "_x": images, key + "_y": masks})
        print(f"Prepared {len(ids)} {key} pairs", flush=True)
    print(f"Training duplicates removed: {audits}", flush=True)
    arrays["duplicate_audit"] = np.array(json.dumps(audits))
    temporary = args.data / "isic2016_128.tmp.npz"
    np.savez(temporary, **arrays)
    temporary.replace(args.data / "isic2016_128.npz")


if __name__ == "__main__":
    main()
