import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Single-file preprocessing for converted .npy")
    p.add_argument("--input-npy", required=True, help="Path to input .npy file")
    p.add_argument("--output-npy", required=True, help="Path to output preprocessed .npy")
    p.add_argument(
        "--layout",
        choices=["auto", "samples_by_channels", "channels_by_samples"],
        default="auto",
        help="How to interpret 2D input array",
    )
    p.add_argument(
        "--keep-first-channel",
        action="store_true",
        help="Do not drop first channel (often timestamp in converted txt files)",
    )
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Output dtype",
    )
    return p.parse_args()


def preprocess_array(arr: np.ndarray, layout: str, keep_first_channel: bool):
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array from converted txt, got ndim={arr.ndim}")

    if layout == "auto":
        # Converted txt arrays are usually (samples, channels).
        arr_cs = arr.T if arr.shape[0] >= arr.shape[1] else arr
        inferred_layout = "samples_by_channels" if arr.shape[0] >= arr.shape[1] else "channels_by_samples"
    elif layout == "samples_by_channels":
        arr_cs = arr.T
        inferred_layout = layout
    else:
        arr_cs = arr
        inferred_layout = layout

    if not keep_first_channel and arr_cs.shape[0] > 1:
        arr_cs = arr_cs[1:, :]

    return arr_cs, inferred_layout


def main():
    args = parse_args()
    input_path = Path(args.input_npy).resolve()
    output_path = Path(args.output_npy).resolve()

    arr = np.load(input_path)
    processed, inferred_layout = preprocess_array(arr, args.layout, args.keep_first_channel)

    out_dtype = np.float32 if args.dtype == "float32" else np.float64
    processed = processed.astype(out_dtype, copy=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, processed)

    report = {
        "input_npy": str(input_path),
        "output_npy": str(output_path),
        "input_shape": list(arr.shape),
        "output_shape": list(processed.shape),
        "input_dtype": str(arr.dtype),
        "output_dtype": str(processed.dtype),
        "layout": inferred_layout,
        "dropped_first_channel": bool(not args.keep_first_channel and arr.ndim == 2 and arr.shape[0] >= 1),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
