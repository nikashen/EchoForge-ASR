from __future__ import annotations

import numpy as np
import pytest

from echoforge.vad import EnergyVAD, EnergyVadConfig, VadEventType, VadState, energy_dbfs


def _tone(samples: int, amplitude: float = 0.1) -> np.ndarray:
    return np.full(samples, amplitude, dtype=np.float32)


def _config() -> EnergyVadConfig:
    return EnergyVadConfig(
        frame_ms=20,
        start_threshold_dbfs=-35.0,
        continue_threshold_dbfs=-42.0,
        min_speech_ms=40,
        min_silence_ms=60,
    )


def test_energy_dbfs_is_finite_and_has_floor() -> None:
    assert energy_dbfs(np.zeros(320, dtype=np.float32)) == -120.0
    assert energy_dbfs(_tone(320, 0.1)) == pytest.approx(-20.0, abs=1e-5)
    with pytest.raises(ValueError):
        energy_dbfs(np.array([np.nan], dtype=np.float32))


def test_vad_emits_debounced_start_and_end_with_boundaries() -> None:
    vad = EnergyVAD(_config())
    frame = _config().frame_samples
    silence = _tone(frame, 0.001)
    speech = _tone(frame, 0.1)

    assert vad.process(np.tile(silence, 2)).__len__() == 0
    # Start detection is debounced over two consecutive speech frames.
    vad = EnergyVAD(_config())
    assert vad.process(np.tile(silence, 2)) == ()
    start_events = vad.process(np.tile(speech, 2))
    assert len(start_events) == 1
    start = start_events[0]
    assert start.event_type is VadEventType.SPEECH_STARTED
    assert start.state is VadState.SPEECH
    assert start.sample_offset == 2 * frame
    assert start.observed_sample_offset == 4 * frame
    assert start.state_version == 1

    assert vad.process(np.tile(silence, 2)) == ()
    end_events = vad.process(np.tile(silence, 2))
    assert len(end_events) == 1
    end = end_events[0]
    assert end.event_type is VadEventType.SPEECH_ENDED
    assert end.state is VadState.SILENCE
    assert end.sample_offset == 4 * frame
    assert end.observed_sample_offset == 7 * frame
    assert end.state_version == 2


def test_vad_is_chunking_invariant_and_flush_closes_partial_frame() -> None:
    config = _config()
    frame = config.frame_samples
    signal = np.concatenate(
        (
            _tone(frame * 2, 0.001),
            _tone(frame * 3, 0.1),
            _tone(frame + 17, 0.001),
        )
    )
    whole = EnergyVAD(config)
    chunked = EnergyVAD(config)
    whole_events = whole.process(signal)
    chunked_events = []
    for chunk in (signal[:11], signal[11:777], signal[777:]):
        chunked_events.extend(chunked.process(chunk))
    assert tuple(chunked_events) == whole_events
    assert chunked.snapshot().pending_samples == 17
    flush_events = chunked.flush()
    assert len(flush_events) == 1
    assert flush_events[0].event_type is VadEventType.SPEECH_ENDED
    assert flush_events[0].reason == "flush"
    assert chunked.state is VadState.SILENCE
    assert chunked.snapshot().pending_samples == 0


def test_vad_hysteresis_keeps_speech_alive_between_thresholds() -> None:
    config = _config()
    frame = config.frame_samples
    vad = EnergyVAD(config)
    assert len(vad.process(_tone(frame * 2, 0.1))) == 1
    # -40 dBFS is below the start threshold but above the continuation threshold.
    assert vad.process(_tone(frame * 4, 0.01)) == ()
    assert vad.state is VadState.SPEECH
    assert len(vad.process(_tone(frame * 3, 0.001))) == 1
    assert vad.state is VadState.SILENCE


def test_vad_rejects_invalid_config_and_audio() -> None:
    with pytest.raises(ValueError):
        EnergyVadConfig(frame_ms=0)
    with pytest.raises(ValueError):
        EnergyVadConfig(start_threshold_dbfs=-40.0, continue_threshold_dbfs=-30.0)
    vad = EnergyVAD(_config())
    with pytest.raises(ValueError):
        vad.process(np.array([], dtype=np.float32))
    with pytest.raises(ValueError):
        vad.process(np.array([np.inf], dtype=np.float32))
    with pytest.raises(ValueError):
        vad.accept_audio(np.zeros(320, dtype=np.float32), 8_000)


def test_flush_discards_an_unfinished_debounce_candidate() -> None:
    config = _config()
    frame = config.frame_samples
    vad = EnergyVAD(config)
    assert vad.process(_tone(frame, 0.1)) == ()
    assert vad.flush() == ()
    # A new utterance still needs the complete debounce window.
    assert vad.process(_tone(frame, 0.1)) == ()
    assert len(vad.process(_tone(frame, 0.1))) == 1
