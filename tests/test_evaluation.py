"""Unit tests for frozen Chinese normalization, CER, and degradations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from echoforge.evaluation.cer import EditCounts, character_error_rate, edit_counts
from echoforge.evaluation.degradations import add_noise_at_snr, telephone_channel
from echoforge.evaluation.normalize_zh import NORMALIZER_VERSION, normalize_zh


def test_normalize_zh_applies_nfkc_lowercase_and_character_filter() -> None:
    text = " ＡＢＣ１２３＿， Hello 世界！① "
    assert normalize_zh(text) == "abc123hello世界1"
    assert NORMALIZER_VERSION == "echoforge.zh-normalizer/v1"


def test_normalize_zh_requires_a_string() -> None:
    with pytest.raises(TypeError):
        normalize_zh(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("你好世界", "你好世界", EditCounts(0, 0, 0, 4, 4)),
        ("你好世界", "你好", EditCounts(0, 2, 0, 4, 2)),
        ("你好", "你啊", EditCounts(1, 0, 0, 2, 2)),
        ("你好", "你好啊", EditCounts(0, 0, 1, 2, 3)),
        ("abc", "", EditCounts(0, 3, 0, 3, 0)),
        ("ab", "ba", EditCounts(2, 0, 0, 2, 2)),
    ],
)
def test_edit_counts_reports_stable_operation_breakdown(
    reference: str, hypothesis: str, expected: EditCounts
) -> None:
    assert edit_counts(reference, hypothesis, normalize=False) == expected


def test_cer_normalizes_by_default_and_can_be_disabled() -> None:
    assert character_error_rate("ＡＢ！", "ab") == 0.0
    counts = edit_counts("A!", "a", normalize=False)
    assert counts.errors == 2
    assert character_error_rate("你好", "你啊") == pytest.approx(0.5)


def test_empty_normalized_reference_is_rejected_and_empty_cer_is_undefined() -> None:
    with pytest.raises(ValueError, match="empty"):
        edit_counts("，！", "anything")
    with pytest.raises(ValueError, match="undefined"):
        empty_counts = EditCounts(0, 0, 0, 0, 0)
        _ = empty_counts.cer


def test_add_noise_at_snr_is_deterministic_and_hits_target_for_stationary_noise() -> None:
    clean = np.full(1_000, 0.1, dtype=np.float32)
    noise = np.full(1_000, 0.05, dtype=np.float32)
    first = add_noise_at_snr(clean, noise, 20.0, seed=11)
    second = add_noise_at_snr(clean, noise, 20.0, seed=11)
    np.testing.assert_array_equal(first, second)
    measured_snr = 10 * math.log10(np.mean(clean**2) / np.mean((first - clean) ** 2))
    assert measured_snr == pytest.approx(20.0, abs=0.05)
    assert first.shape == clean.shape
    assert np.all(np.isfinite(first))


def test_add_noise_tiles_short_noise_and_bounds_clipped_output() -> None:
    clean = np.full(20, 0.9, dtype=np.float32)
    noisy = add_noise_at_snr(clean, [1.0], -10.0, seed=1)
    assert noisy.shape == clean.shape
    assert np.all(np.isfinite(noisy))
    assert np.max(np.abs(noisy)) <= 1.0


def test_add_noise_rejects_invalid_ranges_and_zero_energy() -> None:
    for snr_db in (math.nan, math.inf, -10.1, 60.1):
        with pytest.raises(ValueError):
            add_noise_at_snr([0.1], [0.1], snr_db)
    for clean, noise in (([0.0], [0.1]), ([0.1], [0.0]), ([], [0.1]), ([math.nan], [0.1])):
        with pytest.raises(ValueError):
            add_noise_at_snr(clean, noise, 10.0)


def test_add_noise_uses_the_selected_noise_segment_for_requested_snr() -> None:
    clean = np.full(100, 0.1, dtype=np.float32)
    noise = np.full(300, 0.01, dtype=np.float32)
    noise[148:248] = 0.4
    mixed = add_noise_at_snr(clean, noise, 20.0, seed=17)
    measured_snr = 10 * math.log10(np.mean(clean**2) / np.mean((mixed - clean) ** 2))
    assert measured_snr == pytest.approx(20.0, abs=1.0)


def test_telephone_channel_is_deterministic_16khz_band_limited_audio() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = 0.4 * np.sin(2 * np.pi * 1_000 * time) + 0.4 * np.sin(2 * np.pi * 6_000 * time)
    first = telephone_channel(audio)
    second = telephone_channel(audio)
    np.testing.assert_array_equal(first, second)
    assert first.shape == audio.shape
    assert np.all(np.isfinite(first))
    assert not np.allclose(first, audio)
    assert float(np.sqrt(np.mean(first**2))) < float(np.sqrt(np.mean(audio**2)))


@pytest.mark.parametrize("sample_rate", [8_000, 16_001])
def test_telephone_channel_requires_16khz(sample_rate: int) -> None:
    with pytest.raises(ValueError):
        telephone_channel([0.1, 0.2], sample_rate)


@pytest.mark.parametrize("samples", [[], [math.nan], [math.inf], [[0.1]]])
def test_telephone_channel_rejects_invalid_audio(samples: object) -> None:
    with pytest.raises(ValueError):
        telephone_channel(samples)  # type: ignore[arg-type]
