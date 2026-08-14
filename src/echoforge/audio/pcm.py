from __future__ import annotations

import math
from dataclasses import dataclass
from math import gcd
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly

FloatAudio = NDArray[np.float32]


def _finite_audio(samples: NDArray[np.float32] | list[float]) -> FloatAudio:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("audio must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("audio samples must be finite")
    return np.ascontiguousarray(array)


@dataclass(frozen=True, slots=True)
class AudioChunk:
    sequence: int
    sample_rate: int
    samples: FloatAudio

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a non-negative integer")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise TypeError("sample rate must be an integer")
        if self.sample_rate < 8_000 or self.sample_rate > 48_000:
            raise ValueError("sample rate must be in [8000, 48000]")
        checked = _finite_audio(self.samples)
        if checked.size > self.sample_rate:
            raise ValueError("one audio chunk may contain at most one second")
        object.__setattr__(self, "samples", checked)

    @property
    def duration_seconds(self) -> float:
        return self.samples.size / self.sample_rate


def pcm16le_to_float32(payload: bytes) -> FloatAudio:
    if not isinstance(payload, bytes) or not payload or len(payload) % 2:
        raise ValueError("PCM16LE payload must be non-empty and even-sized")
    integers = np.frombuffer(payload, dtype="<i2")
    return np.ascontiguousarray(integers.astype(np.float32) / 32768.0)


def float32_to_pcm16le(samples: NDArray[np.float32] | list[float]) -> bytes:
    array = _finite_audio(samples)
    scaled = np.clip(np.rint(array.astype(np.float64) * 32768.0), -32768, 32767)
    return scaled.astype("<i2").tobytes()


def rms(samples: NDArray[np.float32] | list[float]) -> float:
    array = _finite_audio(samples).astype(np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def peak_dbfs(samples: NDArray[np.float32] | list[float], *, floor_db: float = -120.0) -> float:
    array = _finite_audio(samples)
    peak = float(np.max(np.abs(array)))
    if peak <= 0:
        return floor_db
    return max(floor_db, 20.0 * math.log10(peak))


def apply_gain_db(samples: NDArray[np.float32] | list[float], gain_db: float) -> FloatAudio:
    if not math.isfinite(gain_db) or gain_db < -60 or gain_db > 30:
        raise ValueError("gain_db must be finite and in [-60, 30]")
    array = _finite_audio(samples).astype(np.float64)
    scaled = array * (10.0 ** (gain_db / 20.0))
    return cast(FloatAudio, np.clip(scaled, -1.0, 1.0).astype(np.float32))


def resample_mono(
    samples: NDArray[np.float32] | list[float],
    source_rate: int,
    target_rate: int = 16_000,
) -> FloatAudio:
    array = _finite_audio(samples)
    if source_rate < 8_000 or source_rate > 192_000:
        raise ValueError("source_rate is outside the supported range")
    if target_rate < 8_000 or target_rate > 48_000:
        raise ValueError("target_rate is outside the supported range")
    if source_rate == target_rate:
        return cast(FloatAudio, array.copy())
    factor = gcd(source_rate, target_rate)
    result = resample_poly(array.astype(np.float64), target_rate // factor, source_rate // factor)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError("resampling produced invalid audio")
    return cast(FloatAudio, np.ascontiguousarray(result.astype(np.float32)))
