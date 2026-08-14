from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from echoforge.contracts.domain import DiffSpan, Hypothesis, RevisionStage, TranscriptRevision


def stable_prefix_length(previous: str, current: str) -> int:
    length = 0
    for left, right in zip(previous, current):
        if left != right:
            break
        length += 1
    return length


def build_diff(fast_text: str, verified_text: str) -> tuple[DiffSpan, ...]:
    matcher = SequenceMatcher(a=fast_text, b=verified_text, autojunk=False)
    return tuple(
        DiffSpan(
            operation=operation,
            fast_text=fast_text[fast_start:fast_end],
            verified_text=verified_text[verified_start:verified_end],
            fast_start=fast_start,
            fast_end=fast_end,
            verified_start=verified_start,
            verified_end=verified_end,
        )
        for operation, fast_start, fast_end, verified_start, verified_end in matcher.get_opcodes()
        if operation != "equal"
    )


class RevisionLog:
    def __init__(self, session_id: str, utterance_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(utterance_id, str) or not utterance_id.strip():
            raise ValueError("utterance_id must be a non-empty string")
        self.session_id = session_id
        self.utterance_id = utterance_id
        self._items: list[TranscriptRevision] = []

    @property
    def items(self) -> tuple[TranscriptRevision, ...]:
        return tuple(self._items)

    @property
    def latest(self) -> TranscriptRevision | None:
        return self._items[-1] if self._items else None

    def append(
        self,
        hypothesis: Hypothesis,
        *,
        lane: str,
        server_compute_ms: float | None = None,
        endpoint_to_final_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TranscriptRevision:
        if not isinstance(lane, str) or not lane.strip():
            raise ValueError("revision lane must not be empty")
        if hypothesis.stage is RevisionStage.PARTIAL and not hypothesis.text.strip():
            raise ValueError("partial hypothesis text must not be empty")
        previous = self.latest
        allowed = {
            None: {RevisionStage.PARTIAL, RevisionStage.STREAM_FINAL},
            RevisionStage.PARTIAL: {RevisionStage.PARTIAL, RevisionStage.STREAM_FINAL},
            RevisionStage.STREAM_FINAL: {RevisionStage.DUAL_PASS_FINAL, RevisionStage.STREAM_ONLY},
            RevisionStage.DUAL_PASS_FINAL: set(),
            RevisionStage.STREAM_ONLY: set(),
        }
        previous_stage = previous.stage if previous else None
        if hypothesis.stage not in allowed[previous_stage]:
            raise ValueError(f"invalid revision transition: {previous_stage} -> {hypothesis.stage}")
        previous_text = previous.text if previous else ""
        revision = TranscriptRevision(
            session_id=self.session_id,
            utterance_id=self.utterance_id,
            revision=len(self._items),
            stage=hypothesis.stage,
            lane=lane,
            text=hypothesis.text,
            model_id=hypothesis.model_id,
            stable_prefix_chars=stable_prefix_length(previous_text, hypothesis.text),
            replaces_revision=previous.revision if previous else None,
            diff=build_diff(previous_text, hypothesis.text) if previous else (),
            decoder_score=hypothesis.decoder_score,
            confidence_kind="decoder_score" if hypothesis.decoder_score is not None else None,
            audio_start_ms=hypothesis.audio_start_ms,
            audio_end_ms=hypothesis.audio_end_ms,
            server_compute_ms=server_compute_ms,
            endpoint_to_final_ms=endpoint_to_final_ms,
            degraded=hypothesis.degraded,
            degradation_code=hypothesis.degradation_code,
            metadata=dict(metadata or {}),
        )
        self._items.append(revision)
        return revision
