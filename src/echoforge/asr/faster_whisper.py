"""Optional endpoint verifier backed by faster-whisper.

This adapter is intentionally endpointed.  It never runs in the streaming
audio path and refuses model identifiers unless the caller explicitly opts in
to downloads.  Keeping that boundary visible prevents a demo from implying
that Whisper is a native low-latency decoder.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from echoforge.contracts.domain import Hypothesis, RevisionStage

FloatAudio = NDArray[np.float32]


class FasterWhisperUnavailable(RuntimeError):
    """Raised when faster-whisper or a local model is unavailable."""


class FasterWhisperFinalizer:
    """Endpoint-only ``EndpointFinalizer`` implementation."""

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        beam_size: int = 5,
        language: str = "zh",
        allow_download: bool = False,
        model_factory: Callable[[Any, str, str, str, int], Any] | None = None,
    ) -> None:
        if not isinstance(model_path, (str, os.PathLike)) or not str(model_path).strip():
            raise ValueError("model_path must be a local path or an explicit model id")
        self.model_path = str(model_path)
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.beam_size = beam_size
        self.language = language
        self.allow_download = allow_download
        self._model_factory = model_factory
        self._model: Any | None = None
        self.model_id = f"faster-whisper:{Path(self.model_path).name}"
        if cpu_threads < 1 or cpu_threads > 64:
            raise ValueError("cpu_threads must be in [1, 64]")
        if beam_size < 1 or beam_size > 20:
            raise ValueError("beam_size must be in [1, 20]")

    def _load(self) -> None:
        if self._model is not None:
            return
        path = Path(self.model_path).expanduser()
        if not path.exists() and not self.allow_download:
            raise FasterWhisperUnavailable(
                f"local Whisper model not found: {path}; set allow_download=True explicitly"
            )
        try:
            module = importlib.import_module("faster_whisper")
        except (ImportError, OSError) as exc:
            raise FasterWhisperUnavailable(
                "faster-whisper is not installed; install echoforge-asr[verifier]"
            ) from exc
        if self._model_factory is not None:
            model = self._model_factory(
                module, self.model_path, self.device, self.compute_type, self.cpu_threads
            )
        else:
            cls = getattr(module, "WhisperModel", None)
            if cls is None:
                raise FasterWhisperUnavailable("faster-whisper lacks WhisperModel")
            model = cls(
                self.model_path,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
        self._model = model

    def transcribe(self, samples: FloatAudio, sample_rate: int) -> Hypothesis:
        if sample_rate != 16_000:
            raise ValueError("faster-whisper verifier expects 16 kHz audio")
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError("audio samples must be finite non-empty mono float32")
        self._load()
        assert self._model is not None
        segments, _info = self._model.transcribe(
            array,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,
            word_timestamps=False,
        )
        parts: list[str] = []
        start_ms = 0
        end_ms = round(array.size * 1000 / sample_rate)
        scores: list[float] = []
        for segment in segments:
            first_segment = not parts
            text = str(getattr(segment, "text", "") or "").strip()
            if text:
                parts.append(text)
            start = getattr(segment, "start", None)
            end = getattr(segment, "end", None)
            if start is not None:
                start_ms = (
                    round(float(start) * 1000)
                    if first_segment
                    else min(start_ms, round(float(start) * 1000))
                )
            if end is not None:
                end_ms = max(end_ms, round(float(end) * 1000))
            logprob = getattr(segment, "avg_logprob", None)
            if logprob is not None and np.isfinite(logprob):
                scores.append(float(logprob))
        text = "".join(parts).strip()
        score = float(sum(scores) / len(scores)) if scores else None
        return Hypothesis(
            text=text,
            stage=RevisionStage.DUAL_PASS_FINAL,
            model_id=self.model_id,
            decoder_score=score,
            audio_start_ms=max(0, start_ms),
            audio_end_ms=max(max(0, start_ms), end_ms),
        )
