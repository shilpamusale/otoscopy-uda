#!/usr/bin/env python3
"""Placeholder data-acquisition helper for the otoscopy-UDA datasets.

This script intentionally bundles NO image data. Each dataset must be obtained
from its original source under its own license (see DATA_CARD.md). Fill in the
per-dataset download / verification logic below once access terms are confirmed.

Usage:
    python scripts/download_data.py --dataset OSU --dest /path/to/OSUdataset
    python scripts/download_data.py --dataset OSU --dest /path/to/OSUdataset --verify-only
"""
import argparse
import sys
from pathlib import Path

DATASETS = {
    "eardrumDs": "Turkey (source domain)",
    "OSU":       "Nationwide Children's Hospital (Ohio) — target domain",
    "Chile":     "Chilean clinical hospital — target domain",
}

REQUIRED_SUBDIRS = ("Normal", "Abnormal")


def verify_layout(dest: Path) -> bool:
    """Check that dest/ has Normal/ and Abnormal/ with at least one image each."""
    ok = True
    if not dest.exists():
        print(f"[FAIL] destination does not exist: {dest}")
        return False
    for sub in REQUIRED_SUBDIRS:
        d = dest / sub
        if not d.is_dir():
            print(f"[FAIL] missing required subdir: {d}")
            ok = False
            continue
        n = sum(1 for p in d.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
        if n == 0:
            print(f"[WARN] no images found in: {d}")
            ok = False
        else:
            print(f"[ OK ] {d}  ({n} images)")
    return ok


def download(dataset: str, dest: Path) -> None:
    raise NotImplementedError(
        f"Download logic for '{dataset}' is not implemented.\n"
        f"Obtain this dataset from its original source (see DATA_CARD.md), place it at\n"
        f"  {dest}\n"
        f"with 'Normal/' and 'Abnormal/' subfolders, then re-run with --verify-only."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS),
                    help="which dataset to fetch / verify")
    ap.add_argument("--dest", required=True, type=Path,
                    help="destination directory for this dataset")
    ap.add_argument("--verify-only", action="store_true",
                    help="only check the on-disk layout; do not attempt download")
    args = ap.parse_args()

    print(f"Dataset: {args.dataset}  —  {DATASETS[args.dataset]}")
    print(f"Dest:    {args.dest}\n")

    if args.verify_only:
        return 0 if verify_layout(args.dest) else 1

    download(args.dataset, args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
