from dataclasses import dataclass

import numpy as np
from scipy.signal import welch


@dataclass
class SpectrumSummary:
    dominant_freq_hz: float
    dominant_power: float
    power_50hz: float
    power_60hz: float
    noise_floor: float
    snr_50hz_db: float
    snr_60hz_db: float


def compute_psd(signal: np.ndarray, sampling_rate: int, nperseg: int = 512):
    if signal.ndim != 1:
        raise ValueError("signal must be 1D")
    nperseg = min(max(8, nperseg), signal.size)
    freqs, psd = welch(signal, fs=sampling_rate, nperseg=nperseg, scaling="density")
    return freqs, psd


def _bandpower(freqs: np.ndarray, psd: np.ndarray, center_hz: float, half_width_hz: float = 1.0) -> float:
    mask = (freqs >= center_hz - half_width_hz) & (freqs <= center_hz + half_width_hz)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def summarize_spectrum(signal: np.ndarray, sampling_rate: int, nperseg: int = 512) -> SpectrumSummary:
    freqs, psd = compute_psd(signal, sampling_rate=sampling_rate, nperseg=nperseg)

    if psd.size > 1:
        idx = int(np.argmax(psd[1:])) + 1
    else:
        idx = 0

    p50 = _bandpower(freqs, psd, 50.0)
    p60 = _bandpower(freqs, psd, 60.0)

    noise_mask = (freqs >= 2.0) & (freqs <= min(45.0, sampling_rate / 2.0))
    noise_floor = float(np.median(psd[noise_mask])) if np.any(noise_mask) else float(np.median(psd))

    eps = 1e-12
    snr_50 = 10.0 * np.log10((p50 + eps) / (noise_floor + eps))
    snr_60 = 10.0 * np.log10((p60 + eps) / (noise_floor + eps))

    return SpectrumSummary(
        dominant_freq_hz=float(freqs[idx]),
        dominant_power=float(psd[idx]),
        power_50hz=p50,
        power_60hz=p60,
        noise_floor=noise_floor,
        snr_50hz_db=float(snr_50),
        snr_60hz_db=float(snr_60),
    )
