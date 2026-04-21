import argparse
import shutil
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Simulate an ESP writing .npy files live")
    p.add_argument("--src-dir", required=True, help="Directory containing offline .npy files")
    p.add_argument("--dest-dir", required=True, help="Directory to copy files into (the live folder)")
    p.add_argument("--interval", type=float, default=2.0, help="Seconds to wait between writing files")
    return p.parse_args()


def main():
    args = parse_args()
    src_dir = Path(args.src_dir).resolve()
    dest_dir = Path(args.dest_dir).resolve()

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.rglob("*.npy"))
    if not files:
        print(f"No .npy files found in {src_dir}")
        return

    print(f"Found {len(files)} files. Starting simulation...")
    print(f"Destination: {dest_dir}")
    print(f"Interval: {args.interval} seconds\n")

    for i, f in enumerate(files):
        dest_path = dest_dir / f.name
        
        # Handle naming collisions if flattening a recursive search
        counter = 1
        while dest_path.exists():
            dest_path = dest_dir / f"{f.stem}_{counter}{f.suffix}"
            counter += 1

        print(f"[{i+1}/{len(files)}] Copying {f.name} -> {dest_path.name} ...")
        shutil.copy2(f, dest_path)
        
        if i < len(files) - 1:
            time.sleep(args.interval)

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
