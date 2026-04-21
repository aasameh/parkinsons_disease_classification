import json
from pathlib import Path

import numpy as np


def flatten_dict(data_dict):
    out = []

    def walk(d, acc):
        is_leaf = True
        for k, v in d.items():
            if isinstance(v, dict):
                is_leaf = False
                walk(v, acc.copy())
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                is_leaf = False
                for item in v:
                    walk(item, acc.copy())
            else:
                acc[k] = v
        if is_leaf:
            out.append(acc)

    walk(data_dict, {})
    return out


def load_raw_observation_matrix(observation_json: Path):
    with open(observation_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    rows = flatten_dict(meta)
    if not rows:
        raise ValueError("No records found in observation file")

    movement_root = observation_json.parent
    min_rows = int(min(r["rows"] for r in rows))

    all_records = []
    all_channels = []

    for r in rows:
        n_splits = int(r["rows"]) // min_rows
        arr = np.loadtxt(movement_root / str(r["file_name"]), dtype=np.float32, delimiter=",")
        arr = np.swapaxes(arr, 0, 1)

        channels = [f"{r['device_location']}_{ch}" for ch in r["channels"]]
        step = arr.shape[1] // n_splits

        if n_splits > 1:
            chunks = [arr[:, i : i + step] for i in range(0, arr.shape[1], step)]
            arr = np.concatenate(chunks, axis=0)
            split_channels = []
            for i in range(n_splits):
                for ch in channels:
                    split_channels.append(f"{r['record_name']}{i + 1}_{ch}")
            channels = split_channels
        else:
            channels = [f"{r['record_name']}_{ch}" for ch in channels]

        all_records.append(arr)
        all_channels.extend(channels)

    return np.concatenate(all_records, axis=0), all_channels


def load_sample(path: Path, bin_channels: int):
    if path.suffix.lower() == ".npy":
        sample = np.load(path).astype(np.float32)

        # Normalize .npy layouts to model format:
        # 2D  -> (channels, samples)
        # 3D  -> (windows, channels, samples)
        if sample.ndim == 2:
            channel_axes = [i for i, size in enumerate(sample.shape) if size == bin_channels]
            if len(channel_axes) == 1 and channel_axes[0] != 0:
                sample = np.moveaxis(sample, channel_axes[0], 0)
            elif len(channel_axes) == 0:
                raise ValueError(
                    f"2D .npy shape {sample.shape} has no axis matching expected channels ({bin_channels})."
                )
        elif sample.ndim == 3:
            channel_axes = [i for i, size in enumerate(sample.shape) if size == bin_channels]
            if len(channel_axes) == 1:
                sample = np.moveaxis(sample, channel_axes[0], 1)

                # Keep windows as axis 0 and samples as axis 2.
                # If dimensions look swapped (samples first, windows last), fix that.
                if sample.shape[0] > sample.shape[2]:
                    sample = np.transpose(sample, (2, 1, 0))
            elif len(channel_axes) == 0:
                raise ValueError(
                    f"3D .npy shape {sample.shape} has no axis matching expected channels ({bin_channels})."
                )
    else:
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % bin_channels != 0:
            raise ValueError(f".bin size ({raw.size}) not divisible by bin_channels ({bin_channels})")
        sample = raw.reshape((bin_channels, -1)).astype(np.float32)

    if sample.ndim not in [2, 3]:
        raise ValueError("Sample must be 2D (channels, samples) or 3D (windows, channels, samples)")

    return sample


def to_windows(sample: np.ndarray, window_minutes: float | None, sampling_rate: int):
    if sample.ndim == 3:
        return sample

    # Auto mode: treat incoming 2D sample as one window (real-time friendly default).
    if window_minutes is None:
        return np.expand_dims(sample, axis=0)

    window_samples = int(round(window_minutes * 60.0 * sampling_rate))
    if window_samples <= 0:
        raise ValueError("Invalid window length")
    if sample.shape[1] < window_samples:
        raise ValueError(f"Sample too short for one window ({sample.shape[1]} < {window_samples})")

    n_windows = sample.shape[1] // window_samples
    trimmed = sample[:, : n_windows * window_samples]
    windows = trimmed.reshape(sample.shape[0], n_windows, window_samples).transpose(1, 0, 2)
    return windows


def split_segments(window: np.ndarray, segment_samples: int, segment_stride: int):
    if segment_samples <= 0 or segment_stride <= 0:
        raise ValueError("segment-samples and segment-stride must be positive")
    if window.shape[1] < segment_samples:
        raise ValueError("Window too short for segment")

    segs = []
    for s in range(0, window.shape[1] - segment_samples + 1, segment_stride):
        segs.append(window[:, s : s + segment_samples])
    return segs
