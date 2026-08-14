from __future__ import annotations

import numpy as np
import pytest

from echoforge.asr.fake import ScriptedFinalizer, ScriptedStreamingRecognizer
from echoforge.cascade import RevisionLog
from echoforge.contracts.domain import Hypothesis, RevisionStage
from echoforge.contracts.errors import BackendFailureError, SequenceConflictError
from echoforge.streaming.session import SessionState, StreamingSession


def _hypothesis(text: str, stage: RevisionStage) -> Hypothesis:
    return Hypothesis(text=text, stage=stage, model_id="fixture", audio_end_ms=100)


def test_revision_log_enforces_monotonic_stages_and_serializes_slots() -> None:
    log = RevisionLog("s1", "utt-000000")
    partial = log.append(_hypothesis("你好", RevisionStage.PARTIAL), lane="streaming")
    assert partial.revision == 0
    assert partial.replaces_revision is None
    assert partial.to_dict()["diff"] == []
    assert (
        log.append(_hypothesis("你好世界", RevisionStage.PARTIAL), lane="streaming").revision == 1
    )
    final = log.append(_hypothesis("你好世界", RevisionStage.STREAM_FINAL), lane="streaming")
    assert final.revision == 2
    assert final.stable_prefix_chars == 4
    verified = log.append(
        _hypothesis("你好世界!", RevisionStage.DUAL_PASS_FINAL),
        lane="verifier",
        server_compute_ms=12.5,
        endpoint_to_final_ms=13.0,
        metadata={"source": "test"},
    )
    assert verified.server_compute_ms == 12.5
    assert verified.endpoint_to_final_ms == 13.0
    assert verified.metadata == {"source": "test"}
    assert verified.to_dict()["metadata"] == {"source": "test"}
    with pytest.raises(ValueError, match="partial hypothesis"):
        RevisionLog("s", "u").append(_hypothesis("", RevisionStage.PARTIAL), lane="streaming")
    with pytest.raises(ValueError, match="invalid revision transition"):
        log.append(_hypothesis("x", RevisionStage.PARTIAL), lane="streaming")


def test_streaming_flush_records_verifier_latency_without_mutating_frozen_revision() -> None:
    ticks = iter((10.0, 10.025))
    session = StreamingSession(
        "s1",
        ScriptedStreamingRecognizer(
            partials=("你好",), final_text="你好世界", samples_per_partial=320
        ),
        ScriptedFinalizer("你好世界"),
        clock=lambda: next(ticks),
    )
    audio = np.zeros(640, dtype=np.float32)
    first = session.ingest(0, audio[:320])
    assert first.revision is not None
    result = session.flush()
    assert result.verifier_degraded is False
    assert [item.stage for item in result.revisions] == [
        RevisionStage.PARTIAL,
        RevisionStage.STREAM_FINAL,
        RevisionStage.DUAL_PASS_FINAL,
    ]
    verifier = result.revisions[-1]
    assert verifier.server_compute_ms == pytest.approx(25.0)
    assert verifier.endpoint_to_final_ms == pytest.approx(25.0)
    assert verifier.to_dict()["server_compute_ms"] == pytest.approx(25.0)
    assert session.snapshot().state is SessionState.READY


def test_verifier_failure_preserves_stream_final_as_degraded_stream_only() -> None:
    session = StreamingSession(
        "s1",
        ScriptedStreamingRecognizer(partials=(), final_text="stream text"),
        ScriptedFinalizer(fail=True),
    )
    session.ingest(0, np.zeros(320, dtype=np.float32))
    result = session.flush()
    assert result.verifier_degraded is True
    assert [item.stage for item in result.revisions] == [
        RevisionStage.STREAM_FINAL,
        RevisionStage.STREAM_ONLY,
    ]
    fallback = result.revisions[-1]
    assert fallback.degraded is True
    assert fallback.degradation_code == "VERIFIER_FAILED"
    assert fallback.text == "stream text"


def test_streaming_backend_contract_failure_is_fail_closed() -> None:
    recognizer = ScriptedStreamingRecognizer(partials=(), final_text="x")
    # Deliberately violate the streaming accept contract.
    recognizer.accept_audio = lambda samples, sample_rate: _hypothesis(
        "x", RevisionStage.STREAM_FINAL
    )  # type: ignore[method-assign]
    session = StreamingSession("s1", recognizer)
    with pytest.raises(BackendFailureError):
        session.ingest(0, np.zeros(320, dtype=np.float32))
    snapshot = session.snapshot()
    assert snapshot.state is SessionState.FAILED
    assert snapshot.failed_code == "STREAMING_BACKEND_FAILED"


def test_duplicate_payload_is_idempotent_and_conflicting_duplicate_is_rejected() -> None:
    session = StreamingSession("s1", ScriptedStreamingRecognizer(partials=(), final_text="x"))
    samples = np.zeros(320, dtype=np.float32)
    session.ingest(0, samples)
    duplicate = session.ingest(0, samples.copy())
    assert duplicate.duplicate is True
    assert duplicate.highest_contiguous_sequence == 0
    with pytest.raises(SequenceConflictError):
        session.ingest(0, np.ones(320, dtype=np.float32))
