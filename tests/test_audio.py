"""Unit tests for PCM utilities and the bounded audio ring buffer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from echoforge.audio.pcm import (
    AudioChunk,
    apply_gain_db,
    float32_to_pcm16le,
    pcm16le_to_float32,
    peak_dbfs,
    resample_mono,
    rms,
)
from echoforge.audio.ring_buffer import AudioRingBuffer


def test_pcm16le_round_trip_and_saturation() -> None:
    samples = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    payload = float32_to_pcm16le(samples)
    decoded = pcm16le_to_float32(payload)
    np.testing.assert_allclose(decoded, samples, atol=1 / 32768)

    saturated = pcm16le_to_float32(float32_to_pcm16le([-2.0, 2.0]))
    np.testing.assert_array_equal(saturated, np.array([-1.0, 32767 / 32768], dtype=np.float32))


@pytest.mark.parametrize(
    "samples",
    [[], [[0.0]], [math.nan], [math.inf]],
)
def test_float32_to_pcm16le_rejects_empty_non_mono_or_non_finite(samples: object) -> None:
    with pytest.raises(ValueError):
        float32_to_pcm16le(samples)  # type: ignore[arg-type]


@pytest.mark.parametrize("payload", [b"", b"\x00", bytearray(b"\x00\x00")])
def test_pcm16le_to_float32_requires_non_empty_even_bytes(payload: object) -> None:
    with pytest.raises(ValueError):
        pcm16le_to_float32(payload)  # type: ignore[arg-type]


def test_audio_chunk_validates_shape_duration_and_makes_contiguous_float32() -> None:
    source = np.array([0.1, 0.2], dtype=np.float64)[::1]
    chunk = AudioChunk(sequence=4, sample_rate=16_000, samples=source)
    assert chunk.duration_seconds == pytest.approx(2 / 16_000)
    assert chunk.samples.dtype == np.float32
    assert chunk.samples.flags.c_contiguous
    with pytest.raises(ValueError):
        AudioChunk(sequence=0, sample_rate=16_000, samples=np.zeros(16_001, dtype=np.float32))


@pytest.mark.parametrize("sequence", [True, -1, 1.5])
def test_audio_chunk_rejects_non_integer_or_negative_sequence(sequence: object) -> None:
    with pytest.raises(ValueError):
        AudioChunk(sequence=sequence, sample_rate=16_000, samples=[0.0])  # type: ignore[arg-type]


@pytest.mark.parametrize("sample_rate", [7_999, 48_001])
def test_audio_chunk_rejects_unsupported_sample_rates(sample_rate: int) -> None:
    with pytest.raises(ValueError):
        AudioChunk(sequence=0, sample_rate=sample_rate, samples=[0.0])


def test_audio_metrics_and_gain_are_finite_and_clipped() -> None:
    samples = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    assert rms(samples) == pytest.approx(math.sqrt(1 / 6), rel=1e-6)
    assert peak_dbfs(samples) == pytest.approx(20 * math.log10(0.5), rel=1e-6)
    assert peak_dbfs(np.zeros(3, dtype=np.float32)) == -120.0
    np.testing.assert_allclose(apply_gain_db(samples, 6.020599913), [-1.0, 0.0, 1.0], atol=1e-5)


@pytest.mark.parametrize("gain_db", [math.nan, math.inf, -60.1, 30.1])
def test_apply_gain_rejects_invalid_gain(gain_db: float) -> None:
    with pytest.raises(ValueError):
        apply_gain_db([0.1], gain_db)


def test_resample_changes_rate_preserves_duration_and_returns_copy_for_same_rate() -> None:
    source = np.sin(np.linspace(0, 2 * np.pi, 800, endpoint=False)).astype(np.float32)
    converted = resample_mono(source, 8_000, 16_000)
    assert converted.size == 1_600
    assert converted.dtype == np.float32
    assert np.all(np.isfinite(converted))

    same_rate = resample_mono(source, 8_000, 8_000)
    same_rate[0] = 99.0
    assert source[0] != 99.0


@pytest.mark.parametrize(
    ("source_rate", "target_rate"),
    [(7_999, 16_000), (192_001, 16_000), (16_000, 7_999), (16_000, 48_001)],
)
def test_resample_rejects_rates_outside_supported_ranges(
    source_rate: int, target_rate: int
) -> None:
    with pytest.raises(ValueError):
        resample_mono([0.0], source_rate, target_rate)


def test_ring_buffer_keeps_only_the_newest_samples() -> None:
    buffer = AudioRingBuffer(5)
    assert len(buffer) == 0
    np.testing.assert_array_equal(buffer.to_array(), np.empty(0, dtype=np.float32))
    buffer.append(np.array([1, 2, 3], dtype=np.float32))
    buffer.append(np.array([4, 5, 6, 7], dtype=np.float32))
    np.testing.assert_array_equal(buffer.to_array(), [3, 4, 5, 6, 7])
    buffer.append(np.array([8, 9, 10, 11, 12, 13], dtype=np.float32))
    np.testing.assert_array_equal(buffer.to_array(), [9, 10, 11, 12, 13])
    assert len(buffer) == 5
    buffer.clear()
    assert len(buffer) == 0


@pytest.mark.parametrize("capacity", [True, 0, -1, 1.5])
def test_ring_buffer_rejects_invalid_capacity(capacity: object) -> None:
    with pytest.raises(ValueError):
        AudioRingBuffer(capacity)  # type: ignore[arg-type]


@pytest.mark.parametrize("samples", [[], [[1.0]], [math.nan], [math.inf]])
def test_ring_buffer_rejects_invalid_audio(samples: object) -> None:
    buffer = AudioRingBuffer(4)
    with pytest.raises(ValueError):
        buffer.append(samples)  # type: ignore[arg-type]
