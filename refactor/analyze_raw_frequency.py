import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from scipy.integrate import trapezoid

from preprocessing import load_raw_observation_matrix
from spectral import compute_psd, summarize_spectrum


def parse_args():
    p = argparse.ArgumentParser(description="Comprehensive frequency analysis before preprocessing")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--observation-json", default=None)
    src.add_argument("--npy", default=None, help="Path to converted .npy file")
    p.add_argument("--channel", default=None)
    p.add_argument("--channel-index", type=int, default=None)
    p.add_argument("--sampling-rate", type=int, default=100)
    p.add_argument("--out-dir", default="../data/out/frequency_analysis")
    return p.parse_args()


def select_signal_from_npy(npy_path: Path, channel_index: int | None):
    arr = np.load(npy_path)

    if arr.ndim == 1:
        idx = 0 if channel_index is None else int(channel_index)
        return arr.astype(np.float64), f"npy_ch_{idx}", "1d"

    if arr.ndim == 2:
        if arr.shape[0] >= arr.shape[1]:
            data = arr.T
            layout = "samples_by_channels"
        else:
            data = arr
            layout = "channels_by_samples"

        idx = int(channel_index) if channel_index is not None else (1 if data.shape[0] > 1 else 0)
        return data[idx].astype(np.float64), f"npy_ch_{idx}", layout

    if arr.ndim == 3:
        idx = int(channel_index) if channel_index is not None else (1 if arr.shape[1] > 1 else 0)
        return arr[0, idx].astype(np.float64), f"npy_win0_ch_{idx}", "windows_channels_samples"

    raise ValueError(f"Unsupported npy ndim={arr.ndim}")


def band_power(freqs, psd, f_low, f_high):
    mask = (freqs >= f_low) & (freqs <= f_high)
    return float(trapezoid(psd[mask], freqs[mask]))


def compute_statistics(signal):
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "rms": float(np.sqrt(np.mean(signal**2))),
        "peak_to_peak": float(np.ptp(signal)),
        "skewness": float(((signal - signal.mean())**3).mean() / (signal.std()**3 + 1e-12)),
    }


def interpret_motion(freq):
    if freq < 2:
        return "Slow voluntary movement"
    elif 3 <= freq <= 12:
        return "Possible tremor"
    else:
        return "High-frequency noise or vibration"


def main():
    args = parse_args()
    source_type = "observation_json" if args.observation_json else "npy"

    if args.observation_json:
        obs = Path(args.observation_json).resolve()
        data, channels = load_raw_observation_matrix(obs)

        if args.channel is None:
            idx = next((i for i, ch in enumerate(channels) if "Accelerometer" in ch), 0)
        else:
            idx = channels.index(args.channel)

        signal = data[idx].astype(np.float64)
        channel_name = channels[idx]
        stem = obs.stem
        source_path = str(obs)
        npy_layout = None
    else:
        npy_path = Path(args.npy).resolve()
        signal, channel_name, npy_layout = select_signal_from_npy(npy_path, args.channel_index)
        stem = npy_path.stem
        source_path = str(npy_path)

    # ===== Core Analysis =====
    summary = summarize_spectrum(signal, sampling_rate=args.sampling_rate)
    freqs, psd = compute_psd(signal, sampling_rate=args.sampling_rate)

    stats = compute_statistics(signal)

    band_summary = {
        "movement_0_2hz": band_power(freqs, psd, 0, 2),
        "tremor_3_12hz": band_power(freqs, psd, 3, 12),
        "high_freq_12_30hz": band_power(freqs, psd, 12, 30),
    }

    motion_type = interpret_motion(summary.dominant_freq_hz)

    # ===== Output Directory =====
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== Raw Signal Plot =====
    t = np.arange(signal.size) / float(args.sampling_rate)
    fig_t, ax_t = plt.subplots(figsize=(11, 4))
    ax_t.plot(t, signal, linewidth=0.8)
    ax_t.set_title(f"Raw Signal: {channel_name}")
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Amplitude")
    ax_t.grid(alpha=0.3)
    raw_plot = out_dir / f"{stem}_raw.png"
    fig_t.savefig(raw_plot, dpi=150)
    plt.close(fig_t)

    # ===== PSD Plot =====
    fig_f, ax_f = plt.subplots(figsize=(11, 4))
    ax_f.semilogy(freqs, psd)
    ax_f.axvline(50, linestyle="--", alpha=0.5, label="50 Hz")
    ax_f.axvline(60, linestyle="--", alpha=0.5, label="60 Hz")
    ax_f.axvline(summary.dominant_freq_hz, linestyle="-", label="Dominant")
    ax_f.set_title("Power Spectral Density")
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("PSD")
    ax_f.legend()
    ax_f.grid(alpha=0.3)
    psd_plot = out_dir / f"{stem}_psd.png"
    fig_f.savefig(psd_plot, dpi=150)
    plt.close(fig_f)

    # ===== Spectrogram =====
    f, t_spec, Sxx = spectrogram(signal, fs=args.sampling_rate, nperseg=256)
    fig_s, ax_s = plt.subplots(figsize=(11, 4))
    pcm = ax_s.pcolormesh(t_spec, f, Sxx, shading='gouraud')
    ax_s.set_ylim(0, 20)
    ax_s.set_title("Spectrogram")
    ax_s.set_xlabel("Time (s)")
    ax_s.set_ylabel("Frequency (Hz)")
    fig_s.colorbar(pcm, ax=ax_s)
    spec_plot = out_dir / f"{stem}_spectrogram.png"
    fig_s.savefig(spec_plot, dpi=150)
    plt.close(fig_s)

    # ===== Histogram =====
    fig_h, ax_h = plt.subplots(figsize=(6, 4))
    ax_h.hist(signal, bins=50)
    ax_h.set_title("Amplitude Distribution")
    hist_plot = out_dir / f"{stem}_hist.png"
    fig_h.savefig(hist_plot, dpi=150)
    plt.close(fig_h)

    # ===== Final Output =====
    output = {
        "source_type": source_type,
        "source_path": source_path,
        "channel": channel_name,
        "npy_layout": npy_layout,
        "sampling_rate": args.sampling_rate,

        "dominant_freq_hz": summary.dominant_freq_hz,
        "dominant_power": summary.dominant_power,
        "motion_type": motion_type,

        "band_powers": band_summary,
        "statistics": stats,

        "power_50hz": summary.power_50hz,
        "power_60hz": summary.power_60hz,
        "noise_floor": summary.noise_floor,
        "snr_50hz_db": summary.snr_50hz_db,
        "snr_60hz_db": summary.snr_60hz_db,

        "plots": {
            "raw": str(raw_plot),
            "psd": str(psd_plot),
            "spectrogram": str(spec_plot),
            "histogram": str(hist_plot),
        }
    }

    summary_path = out_dir / f"{stem}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()