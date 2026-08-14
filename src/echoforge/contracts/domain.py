from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .enums import StringEnum


class RevisionStage(StringEnum):
    PARTIAL = "partial"
    STREAM_FINAL = "stream_final"
    DUAL_PASS_FINAL = "dual_pass_final"
    STREAM_ONLY = "stream_only"


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate: int = 16_000
    channels: int = 1
    sample_width: int = 2
    encoding: str = "pcm_s16le"

    def __post_init__(self) -> None:
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise ValueError("sample_rate must be an integer")  # noqa: TRY004
        if isinstance(self.channels, bool) or not isinstance(self.channels, int):
            raise ValueError("channels must be an integer")  # noqa: TRY004
        if isinstance(self.sample_width, bool) or not isinstance(self.sample_width, int):
            raise ValueError("sample_width must be an integer")  # noqa: TRY004
        if self.sample_rate != 16_000:
            raise ValueError("streaming v1 requires a 16 kHz sample rate")
        if self.channels != 1 or self.sample_width != 2 or self.encoding != "pcm_s16le":
            raise ValueError("streaming v1 requires mono PCM16LE")


@dataclass(frozen=True, slots=True)
class Hypothesis:
    text: str
    stage: RevisionStage
    model_id: str
    decoder_score: float | None = None
    audio_start_ms: int = 0
    audio_end_ms: int = 0
    degraded: bool = False
    degradation_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("hypothesis text must be a string")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if self.decoder_score is not None and (
            isinstance(self.decoder_score, bool)
            or not isinstance(self.decoder_score, (int, float))
            or not math.isfinite(self.decoder_score)
        ):
            raise ValueError("decoder_score must be finite")
        if self.audio_start_ms < 0 or self.audio_end_ms < self.audio_start_ms:
            raise ValueError("audio offsets must be monotonic")


@dataclass(frozen=True, slots=True)
class DiffSpan:
    operation: str
    fast_text: str
    verified_text: str
    fast_start: int
    fast_end: int
    verified_start: int
    verified_end: int


@dataclass(frozen=True, slots=True)
class TranscriptRevision:
    session_id: str
    utterance_id: str
    revision: int
    stage: RevisionStage
    lane: str
    text: str
    model_id: str
    stable_prefix_chars: int
    replaces_revision: int | None = None
    diff: tuple[DiffSpan, ...] = field(default_factory=tuple)
    decoder_score: float | None = None
    confidence_kind: str | None = None
    audio_start_ms: int = 0
    audio_end_ms: int = 0
    server_compute_ms: float | None = None
    endpoint_to_final_ms: float | None = None
    degraded: bool = False
    degradation_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, text_value in (
            ("session_id", self.session_id),
            ("utterance_id", self.utterance_id),
            ("lane", self.lane),
            ("model_id", self.model_id),
        ):
            if not isinstance(text_value, str) or not text_value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.revision < 0:
            raise ValueError("revision must be non-negative")
        if self.stable_prefix_chars < 0 or self.stable_prefix_chars > len(self.text):
            raise ValueError("stable prefix is outside text")
        if self.audio_start_ms < 0 or self.audio_end_ms < self.audio_start_ms:
            raise ValueError("audio offsets must be monotonic")
        if self.decoder_score is not None and (
            isinstance(self.decoder_score, bool)
            or not isinstance(self.decoder_score, (int, float))
            or not math.isfinite(self.decoder_score)
        ):
            raise ValueError("decoder_score must be finite")
        for metric in (self.server_compute_ms, self.endpoint_to_final_ms):
            if metric is not None and (
                isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or not math.isfinite(metric)
                or metric < 0
            ):
                raise ValueError("latency metrics must be finite and non-negative")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "utterance_id": self.utterance_id,
            "revision": self.revision,
            "stage": self.stage.value,
            "lane": self.lane,
            "text": self.text,
            "model_id": self.model_id,
            "stable_prefix_chars": self.stable_prefix_chars,
            "replaces_revision": self.replaces_revision,
            "diff": [
                span.__dict__
                if hasattr(span, "__dict__")
                else {
                    "operation": span.operation,
                    "fast_text": span.fast_text,
                    "verified_text": span.verified_text,
                    "fast_start": span.fast_start,
                    "fast_end": span.fast_end,
                    "verified_start": span.verified_start,
                    "verified_end": span.verified_end,
                }
                for span in self.diff
            ],
            "decoder_score": self.decoder_score,
            "confidence_kind": self.confidence_kind,
            "audio_start_ms": self.audio_start_ms,
            "audio_end_ms": self.audio_end_ms,
            "server_compute_ms": self.server_compute_ms,
            "endpoint_to_final_ms": self.endpoint_to_final_ms,
            "degraded": self.degraded,
            "degradation_code": self.degradation_code,
            "metadata": dict(self.metadata),
        }
