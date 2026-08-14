from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from echoforge.contracts.domain import Hypothesis, RevisionStage


class ScriptedStreamingRecognizer:
    """Deterministic protocol fixture. It is deliberately not an ASR model."""

    model_id = "deterministic-protocol-fixture"

    def __init__(
        self,
        partials: Sequence[str] = ("我们今天讨论图神经网路",),
        final_text: str = "我们今天讨论图神经网路",
        *,
        samples_per_partial: int = 3_200,
        fail_after_samples: int | None = None,
    ) -> None:
        if samples_per_partial < 1:
            raise ValueError("samples_per_partial must be positive")
        self.partials = tuple(partials)
        self.final_text = final_text
        self.samples_per_partial = samples_per_partial
        self.fail_after_samples = fail_after_samples
        self.reset()

    def reset(self) -> None:
        self._samples = 0
        self._emitted = 0

    def accept_audio(self, samples: NDArray[np.float32], sample_rate: int) -> Hypothesis | None:
        if sample_rate != 16_000:
            raise ValueError("fixture expects 16 kHz")
        if samples.ndim != 1 or not np.all(np.isfinite(samples)):
            raise ValueError("fixture audio must be finite mono samples")
        self._samples += samples.size
        if self.fail_after_samples is not None and self._samples >= self.fail_after_samples:
            raise RuntimeError("injected streaming backend failure")
        target = self._samples // self.samples_per_partial
        if self._emitted < len(self.partials) and target > self._emitted:
            text = self.partials[self._emitted]
            self._emitted += 1
            return Hypothesis(
                text=text,
                stage=RevisionStage.PARTIAL,
                model_id=self.model_id,
                audio_end_ms=round(self._samples * 1000 / sample_rate),
            )
        return None

    def finalize(self) -> Hypothesis:
        return Hypothesis(
            text=self.final_text,
            stage=RevisionStage.STREAM_FINAL,
            model_id=self.model_id,
            audio_end_ms=round(self._samples * 1000 / 16_000),
        )


class ScriptedFinalizer:
    """Endpoint fixture used for revision and failure-path tests only."""

    model_id = "deterministic-verifier-fixture"

    def __init__(self, text: str = "我们今天讨论图神经网络", *, fail: bool = False) -> None:
        self.text = text
        self.fail = fail

    def transcribe(self, samples: NDArray[np.float32], sample_rate: int) -> Hypothesis:
        if self.fail:
            raise RuntimeError("injected endpoint verifier failure")
        if sample_rate != 16_000 or samples.ndim != 1 or samples.size == 0:
            raise ValueError("verifier fixture expects non-empty 16 kHz mono audio")
        return Hypothesis(
            text=self.text,
            stage=RevisionStage.DUAL_PASS_FINAL,
            model_id=self.model_id,
            audio_end_ms=round(samples.size * 1000 / sample_rate),
        )
