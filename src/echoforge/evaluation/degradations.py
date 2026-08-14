from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, resample_poly, sosfilt

FloatAudio = NDArray[np.float32]


def _audio(value: NDArray[np.float32] | list[float], name: str) -> FloatAudio:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite non-empty mono audio")
    return np.ascontiguousarray(array)


def add_noise_at_snr(
    clean: NDArray[np.float32] | list[float],
    noise: NDArray[np.float32] | list[float],
    snr_db: float,
    *,
    seed: int = 17,
) -> FloatAudio:
    if not math.isfinite(snr_db) or snr_db < -10 or snr_db > 60:
        raise ValueError("snr_db must be finite and in [-10, 60]")
    clean_audio = _audio(clean, "clean")
    noise_audio = _audio(noise, "noise")
    clean_power = float(np.mean(np.square(clean_audio.astype(np.float64))))
    noise_power = float(np.mean(np.square(noise_audio.astype(np.float64))))
    if clean_power <= 1e-12 or noise_power <= 1e-12:
        raise ValueError("clean and noise must both contain measurable energy")

    rng = np.random.default_rng(seed)
    if noise_audio.size < clean_audio.size:
        repeats = int(np.ceil(clean_audio.size / noise_audio.size))
        noise_audio = np.tile(noise_audio, repeats)
    start = int(rng.integers(0, noise_audio.size - clean_audio.size + 1))
    aligned = noise_audio[start : start + clean_audio.size].astype(np.float64)
    aligned_power = float(np.mean(np.square(aligned)))
    if aligned_power <= 1e-12:
        raise ValueError("selected noise segment must contain measurable energy")
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scaled_noise = aligned * math.sqrt(target_noise_power / aligned_power)
    mixed = clean_audio.astype(np.float64) + scaled_noise
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed /= peak
    return mixed.astype(np.float32)


def telephone_channel(
    samples: NDArray[np.float32] | list[float],
    sample_rate: int = 16_000,
) -> FloatAudio:
    """Deterministic 300-3400 Hz band-limit plus 8 kHz round trip."""

    audio = _audio(samples, "samples")
    if sample_rate != 16_000:
        raise ValueError("telephone_channel v1 requires 16 kHz input")
    sos = butter(6, [300, 3400], btype="bandpass", fs=sample_rate, output="sos")
    filtered = sosfilt(sos, audio.astype(np.float64))
    down = resample_poly(filtered, 1, 2)
    up = resample_poly(down, 2, 1)[: audio.size]
    return np.ascontiguousarray(np.clip(up, -1.0, 1.0).astype(np.float32))
