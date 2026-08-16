from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from echoforge.contracts.enums import StringEnum


class VadState(StringEnum):
    SILENCE = "silence"
    SPEECH = "speech"


class VadEventType(StringEnum):
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"


@dataclass(frozen=True, slots=True)
class EnergyVadConfig:
    """Frozen parameters for the deterministic energy VAD.

    The v1 wire contract is 16 kHz mono audio. Durations are converted to
    sample counts once, so processing the same samples in different chunk
    sizes produces byte-for-byte equivalent events.
    """

    sample_rate: int = 16_000
    frame_ms: int = 20
    start_threshold_dbfs: float = -42.0
    continue_threshold_dbfs: float = -48.0
    min_speech_ms: int = 80
    min_silence_ms: int = 240

    def __post_init__(self) -> None:
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if self.sample_rate != 16_000:
            raise ValueError("deterministic VAD v1 requires a 16 kHz sample rate")
        if (
            isinstance(self.frame_ms, bool)
            or not isinstance(self.frame_ms, int)
            or self.frame_ms < 1
        ):
            raise ValueError("frame_ms must be a positive integer")
        if self.frame_ms * self.sample_rate % 1000:
            raise ValueError("frame_ms must map to a whole number of samples")
        for name, value in (
            ("start_threshold_dbfs", self.start_threshold_dbfs),
            ("continue_threshold_dbfs", self.continue_threshold_dbfs),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < -120.0
                or value > 0.0
            ):
                raise ValueError(f"{name} must be finite and in [-120, 0] dBFS")
        if self.continue_threshold_dbfs > self.start_threshold_dbfs:
            raise ValueError("continue threshold must not exceed start threshold")
        for name, value in (
            ("min_speech_ms", self.min_speech_ms),
            ("min_silence_ms", self.min_silence_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def frame_samples(self) -> int:
        return self.frame_ms * self.sample_rate // 1000

    @property
    def min_speech_samples(self) -> int:
        return math.ceil(self.min_speech_ms * self.sample_rate / 1000)

    @property
    def min_silence_samples(self) -> int:
        return math.ceil(self.min_silence_ms * self.sample_rate / 1000)


@dataclass(frozen=True, slots=True)
class VadEvent:
    """A stable VAD transition.

    ``sample_offset`` is the best estimate of the transition boundary (the
    first frame that satisfied the debounce rule). ``observed_sample_offset``
    is where the server had enough audio to make that decision. The distinction
    makes debounce delay measurable without pretending the boundary was known
    earlier.
    """

    event_type: VadEventType
    state: VadState
    sample_offset: int
    observed_sample_offset: int
    frame_dbfs: float | None
    state_version: int
    reason: str

    @property
    def offset_ms(self) -> int:
        return round(self.sample_offset * 1000 / 16_000)

    @property
    def observed_offset_ms(self) -> int:
        return round(self.observed_sample_offset * 1000 / 16_000)

    def to_dict(self) -> dict[str, object]:
        return {
            "event_type": self.event_type.value,
            "state": self.state.value,
            "sample_offset": self.sample_offset,
            "observed_sample_offset": self.observed_sample_offset,
            "offset_ms": self.offset_ms,
            "observed_offset_ms": self.observed_offset_ms,
            "frame_dbfs": self.frame_dbfs,
            "state_version": self.state_version,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class VadSnapshot:
    state: VadState
    state_version: int
    total_samples: int
    pending_samples: int
    active_start_sample: int | None

    @property
    def total_ms(self) -> int:
        return round(self.total_samples * 1000 / 16_000)

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "state_version": self.state_version,
            "total_samples": self.total_samples,
            "pending_samples": self.pending_samples,
            "active_start_sample": self.active_start_sample,
            "total_ms": self.total_ms,
        }


FloatAudio = NDArray[np.float32]


def energy_dbfs(samples: NDArray[np.float32] | list[float]) -> float:
    """Return RMS level in dBFS with a deterministic -120 dB floor."""

    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("audio must be finite non-empty one-dimensional samples")
    level = float(np.sqrt(np.mean(np.square(array))))
    if level <= 0.0:
        return -120.0
    return max(-120.0, 20.0 * math.log10(level))


class EnergyVAD:
    """Deterministic frame-energy VAD with hysteresis and debounce."""

    _pending: FloatAudio

    def __init__(self, config: EnergyVadConfig | None = None) -> None:
        self.config = config or EnergyVadConfig()
        self.reset()

    @property
    def state(self) -> VadState:
        return self._state

    @property
    def is_speech(self) -> bool:
        return self._state is VadState.SPEECH

    def snapshot(self) -> VadSnapshot:
        return VadSnapshot(
            state=self._state,
            state_version=self._state_version,
            total_samples=self._processed_samples + self._pending.size,
            pending_samples=int(self._pending.size),
            active_start_sample=self._active_start_sample,
        )

    def reset(self) -> None:
        self._state = VadState.SILENCE
        self._state_version = 0
        self._processed_samples = 0
        self._pending = np.empty(0, dtype=np.float32)
        self._speech_run_samples = 0
        self._silence_run_samples = 0
        self._candidate_start_sample: int | None = None
        self._candidate_end_sample: int | None = None
        self._active_start_sample: int | None = None

    def process(self, samples: NDArray[np.float32] | list[float]) -> tuple[VadEvent, ...]:
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError("audio must be finite non-empty one-dimensional samples")
        if self._pending.size:
            array = np.concatenate((self._pending, array)).astype(np.float32, copy=False)
        frame_size = self.config.frame_samples
        full_size = (array.size // frame_size) * frame_size
        events: list[VadEvent] = []
        for start in range(0, full_size, frame_size):
            end = start + frame_size
            frame = array[start:end]
            events.extend(self._consume_frame(frame))
        self._pending = np.ascontiguousarray(array[full_size:], dtype=np.float32)
        return tuple(events)

    def accept_audio(
        self, samples: NDArray[np.float32] | list[float], sample_rate: int = 16_000
    ) -> tuple[VadEvent, ...]:
        """Recognizer-style alias for callers that share an audio pipeline."""

        if sample_rate != self.config.sample_rate:
            raise ValueError("VAD audio must use the configured sample rate")
        return self.process(samples)

    def flush(self) -> tuple[VadEvent, ...]:
        """Consume a final short frame and close an active speech segment."""

        events: list[VadEvent] = []
        if self._pending.size:
            events.extend(self._consume_frame(self._pending))
            self._pending = np.empty(0, dtype=np.float32)
        if self._state is VadState.SPEECH:
            total = self._processed_samples
            self._state = VadState.SILENCE
            self._state_version += 1
            events.append(
                VadEvent(
                    event_type=VadEventType.SPEECH_ENDED,
                    state=VadState.SILENCE,
                    sample_offset=total,
                    observed_sample_offset=total,
                    frame_dbfs=None,
                    state_version=self._state_version,
                    reason="flush",
                )
            )
            self._active_start_sample = None
            self._speech_run_samples = 0
            self._silence_run_samples = 0
            self._candidate_start_sample = None
            self._candidate_end_sample = None
        else:
            # A short, unfinished candidate must not leak into the next
            # utterance after an explicit flush.
            self._speech_run_samples = 0
            self._silence_run_samples = 0
            self._candidate_start_sample = None
            self._candidate_end_sample = None
        return tuple(events)

    def finalize(self) -> tuple[VadEvent, ...]:
        """Recognizer-style alias that closes the current speech segment."""

        return self.flush()

    def _consume_frame(self, frame: NDArray[np.float32]) -> tuple[VadEvent, ...]:
        frame_start = self._processed_samples
        self._processed_samples += int(frame.size)
        frame_end = self._processed_samples
        level = energy_dbfs(frame)
        if self._state is VadState.SILENCE:
            return self._consume_silence_frame(level, frame_start, frame_end)
        return self._consume_speech_frame(level, frame_start, frame_end)

    def _consume_silence_frame(self, level: float, start: int, end: int) -> tuple[VadEvent, ...]:
        if level < self.config.start_threshold_dbfs:
            self._speech_run_samples = 0
            self._candidate_start_sample = None
            return ()
        if self._speech_run_samples == 0:
            self._candidate_start_sample = start
        self._speech_run_samples += end - start
        if self._speech_run_samples < self.config.min_speech_samples:
            return ()
        self._state = VadState.SPEECH
        self._state_version += 1
        active_start = (
            self._candidate_start_sample if self._candidate_start_sample is not None else start
        )
        self._active_start_sample = active_start
        self._speech_run_samples = 0
        self._candidate_start_sample = None
        self._silence_run_samples = 0
        return (
            VadEvent(
                event_type=VadEventType.SPEECH_STARTED,
                state=VadState.SPEECH,
                sample_offset=active_start,
                observed_sample_offset=end,
                frame_dbfs=level,
                state_version=self._state_version,
                reason="energy",
            ),
        )

    def _consume_speech_frame(self, level: float, start: int, end: int) -> tuple[VadEvent, ...]:
        if level >= self.config.continue_threshold_dbfs:
            self._silence_run_samples = 0
            self._candidate_end_sample = None
            return ()
        if self._silence_run_samples == 0:
            self._candidate_end_sample = start
        self._silence_run_samples += end - start
        if self._silence_run_samples < self.config.min_silence_samples:
            return ()
        self._state = VadState.SILENCE
        self._state_version += 1
        end_offset = self._candidate_end_sample if self._candidate_end_sample is not None else start
        self._active_start_sample = None
        self._silence_run_samples = 0
        self._candidate_end_sample = None
        self._speech_run_samples = 0
        return (
            VadEvent(
                event_type=VadEventType.SPEECH_ENDED,
                state=VadState.SILENCE,
                sample_offset=end_offset,
                observed_sample_offset=end,
                frame_dbfs=level,
                state_version=self._state_version,
                reason="energy",
            ),
        )


# The all-caps alias is convenient for callers that use the common acronym.
DeterministicEnergyVAD = EnergyVAD
EnergyVad = EnergyVAD
EnergyVADConfig = EnergyVadConfig
