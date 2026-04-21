import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Batch convert numeric .txt files to .npy")
    p.add_argument(
        "--input-root",
        default=".",
        help="Root directory to recursively scan for .txt files",
    )
    p.add_argument(
        "--out-root",
        default=None,
        help="Optional output root; preserves relative folder structure from input-root",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without writing files",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit on first conversion error",
    )
    return p.parse_args()


def discover_txt_files(input_root: Path):
    return sorted([p for p in input_root.rglob("*.txt") if p.is_file()])


def resolve_output_path(src: Path, input_root: Path, out_root: Path | None):
    if out_root is None:
        return src.with_suffix(".npy")
    rel = src.relative_to(input_root)
    return (out_root / rel).with_suffix(".npy")


def load_numeric_txt(path: Path):
    # Raw movement text files are usually comma-separated; fall back to whitespace.
    try:
        return np.loadtxt(path, dtype=np.float32, delimiter=",")
    except Exception:
        return np.loadtxt(path, dtype=np.float32)


def main():
    args = parse_args()

    input_root = Path(args.input_root).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else None

    if not input_root.is_dir():
        raise ValueError(f"input-root is not a directory: {input_root}")

    txt_files = discover_txt_files(input_root)
    if not txt_files:
        print(f"No .txt files found under: {input_root}")
        return

    converted = 0
    skipped = 0
    failed = 0

    print(f"Found {len(txt_files)} .txt files under: {input_root}")
    if out_root is not None:
        print(f"Output root: {out_root}")

    for src in txt_files:
        dst = resolve_output_path(src, input_root, out_root)

        if dst.exists() and not args.overwrite:
            skipped += 1
            print(f"[SKIP] exists: {dst}")
            continue

        try:
            arr = load_numeric_txt(src)
            if not args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                np.save(dst, arr.astype(np.float32, copy=False))
            converted += 1
            action = "DRYRUN" if args.dry_run else "OK"
            print(f"[{action}] {src} -> {dst} shape={arr.shape}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {src} ({type(e).__name__}: {e})")
            if args.strict:
                raise

    print("\nSummary")
    print(f"- Converted: {converted}")
    print(f"- Skipped:   {skipped}")
    print(f"- Failed:    {failed}")


if __name__ == "__main__":
    main()
