"""Unit tests for EchoForge's wire and domain contracts."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from echoforge.contracts.domain import (
    AudioFormat,
    DiffSpan,
    Hypothesis,
    RevisionStage,
    TranscriptRevision,
)
from echoforge.contracts.errors import ProtocolError
from echoforge.contracts.framing import (
    AUDIO_HEADER,
    AUDIO_MAGIC,
    MAX_AUDIO_FRAME_BYTES,
    MAX_AUDIO_SAMPLES,
    pack_audio_frame,
    unpack_audio_frame,
)
from echoforge.contracts.messages import (
    FlushCommand,
    HotwordsCommand,
    ResetCommand,
    StartCommand,
    StopCommand,
    parse_client_command,
)


def test_audio_format_accepts_only_the_v1_wire_format() -> None:
    assert AudioFormat() == AudioFormat(16_000, 1, 2, "pcm_s16le")
    invalid_formats = (
        {"sample_rate": 8_000},
        {"channels": 2},
        {"sample_width": 1},
        {"encoding": "float32"},
    )
    for values in invalid_formats:
        with pytest.raises(ValueError):
            AudioFormat(**values)


def test_hypothesis_validates_scores_and_audio_offsets() -> None:
    hypothesis = Hypothesis(
        text="你好",
        stage=RevisionStage.PARTIAL,
        model_id="fake-stream",
        decoder_score=-0.25,
        audio_start_ms=10,
        audio_end_ms=90,
    )
    assert hypothesis.audio_end_ms == 90
    for score in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            Hypothesis(text="x", stage=RevisionStage.PARTIAL, model_id="m", decoder_score=score)
    for start, end in ((-1, 1), (10, 9)):
        with pytest.raises(ValueError):
            Hypothesis(
                text="x",
                stage=RevisionStage.PARTIAL,
                model_id="m",
                audio_start_ms=start,
                audio_end_ms=end,
            )


def test_transcript_revision_serializes_slots_diff_and_copies_metadata() -> None:
    span = DiffSpan(
        operation="replace",
        fast_text="天气",
        verified_text="天气预报",
        fast_start=0,
        fast_end=2,
        verified_start=0,
        verified_end=4,
    )
    revision = TranscriptRevision(
        session_id="session-1",
        utterance_id="utt-1",
        revision=2,
        stage=RevisionStage.DUAL_PASS_FINAL,
        lane="verified",
        text="天气预报",
        model_id="verifier",
        stable_prefix_chars=4,
        replaces_revision=1,
        diff=(span,),
        decoder_score=0.8,
        confidence_kind="posterior",
        audio_start_ms=0,
        audio_end_ms=500,
        server_compute_ms=12.5,
        endpoint_to_final_ms=45.0,
        metadata={"source": "test"},
    )

    payload = revision.to_dict()
    assert payload["stage"] == "dual_pass_final"
    assert payload["diff"] == [
        {
            "operation": "replace",
            "fast_text": "天气",
            "verified_text": "天气预报",
            "fast_start": 0,
            "fast_end": 2,
            "verified_start": 0,
            "verified_end": 4,
        }
    ]
    payload["metadata"]["source"] = "mutated-copy"
    assert revision.metadata["source"] == "test"
    with pytest.raises(FrozenInstanceError):
        revision.text = "changed"  # type: ignore[misc]


def test_transcript_revision_rejects_invalid_revision_and_metrics() -> None:
    base = {
        "session_id": "s",
        "utterance_id": "u",
        "stage": RevisionStage.PARTIAL,
        "lane": "stream",
        "text": "abc",
        "model_id": "m",
        "stable_prefix_chars": 0,
    }
    for values in (
        {"revision": -1},
        {"revision": 0, "stable_prefix_chars": 4},
        {"revision": 0, "server_compute_ms": math.nan},
        {"revision": 0, "endpoint_to_final_ms": math.inf},
    ):
        candidate = {**base, **values}
        with pytest.raises(ValueError):
            TranscriptRevision(**candidate)


def test_protocol_error_exposes_wire_safe_details() -> None:
    error = ProtocolError("BAD_FRAME", "frame is invalid", retryable=True, close_code=1003)
    assert str(error) == "BAD_FRAME: frame is invalid"
    assert error.retryable is True
    assert error.close_code == 1003


def test_binary_audio_frame_round_trip() -> None:
    payload = b"\x00\x00\xff\x7f\x00\x80\x01\x00"
    encoded = pack_audio_frame(7, payload)
    decoded = unpack_audio_frame(encoded)
    assert decoded.sequence == 7
    assert decoded.sample_count == 4
    assert decoded.pcm16le == payload


@pytest.mark.parametrize("sequence", [True, -1, 2**32, 1.5])
def test_pack_audio_frame_rejects_invalid_sequences(sequence: object) -> None:
    with pytest.raises(ValueError):
        pack_audio_frame(sequence, b"\x00\x00")  # type: ignore[arg-type]


@pytest.mark.parametrize("payload", [b"", b"\x00", bytearray(b"\x00\x00")])
def test_pack_audio_frame_rejects_invalid_payloads(payload: object) -> None:
    with pytest.raises(ValueError):
        pack_audio_frame(0, payload)  # type: ignore[arg-type]
    if isinstance(payload, bytes) and len(payload) == 0:
        assert MAX_AUDIO_SAMPLES * 2 <= MAX_AUDIO_FRAME_BYTES


def test_pack_audio_frame_accepts_the_exact_size_limit() -> None:
    payload = b"\x00\x00" * MAX_AUDIO_SAMPLES
    encoded = pack_audio_frame(2**32 - 1, payload)
    assert len(encoded) == MAX_AUDIO_FRAME_BYTES
    assert unpack_audio_frame(encoded).sample_count == MAX_AUDIO_SAMPLES


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (b"", "AUDIO_FRAME_TRUNCATED"),
        (AUDIO_MAGIC + b"\x00", "AUDIO_FRAME_TRUNCATED"),
        (AUDIO_HEADER.pack(b"NOPE", 0, 1) + b"\x00\x00", "AUDIO_MAGIC_INVALID"),
        (AUDIO_HEADER.pack(AUDIO_MAGIC, 0, 0), "AUDIO_SAMPLE_COUNT_INVALID"),
        (AUDIO_HEADER.pack(AUDIO_MAGIC, 0, 2) + b"\x00\x00", "AUDIO_LENGTH_MISMATCH"),
        (AUDIO_HEADER.pack(AUDIO_MAGIC, 0, 1) + b"\x00" * 3, "AUDIO_LENGTH_MISMATCH"),
        (b"x" * (MAX_AUDIO_FRAME_BYTES + 1), "AUDIO_FRAME_TOO_LARGE"),
    ],
)
def test_unpack_audio_frame_reports_specific_protocol_errors(data: bytes, code: str) -> None:
    with pytest.raises(ProtocolError) as caught:
        unpack_audio_frame(data)
    assert caught.value.code == code


def test_start_command_normalizes_and_deduplicates_hotwords() -> None:
    raw = json.dumps(
        {
            "type": "session.start",
            "request_id": "req-1",
            "session_id": "demo:session",
            "hotwords": [" EchoForge ", "ASR", "EchoForge"],
        },
        ensure_ascii=False,
    )
    command = parse_client_command(raw)
    assert isinstance(command, StartCommand)
    assert command.mode == "dual_pass"
    assert command.hotwords == ("EchoForge", "ASR")


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        ({"type": "stream.flush", "request_id": "f", "expected_generation": 3}, FlushCommand),
        ({"type": "session.stop", "request_id": "s"}, StopCommand),
        (
            {
                "type": "session.reset",
                "request_id": "r",
                "reset_id": "reset-1",
                "expected_state_version": 2,
                "expected_generation": 4,
            },
            ResetCommand,
        ),
        ({"type": "hotwords.update", "request_id": "h", "hotwords": ["a", "a"]}, HotwordsCommand),
    ],
)
def test_parse_client_command_dispatches_all_supported_types(
    payload: dict[str, object], expected_type: type[object]
) -> None:
    command = parse_client_command(json.dumps(payload))
    assert isinstance(command, expected_type)


def test_parse_client_command_rejects_invalid_json_utf8_size_and_extra_fields() -> None:
    with pytest.raises(ValueError, match="valid UTF-8"):
        parse_client_command(b"\xff")
    with pytest.raises(ValueError, match="16 KiB"):
        parse_client_command("x" * (16 * 1024 + 1))
    with pytest.raises(ValueError):
        parse_client_command("not-json")
    with pytest.raises(ValidationError):
        parse_client_command(json.dumps({"type": "ping", "request_id": "p", "extra": 1}))
    with pytest.raises(ValidationError):
        parse_client_command(json.dumps({"type": "unknown", "request_id": "p"}))


def test_message_constraints_are_strict_and_models_are_frozen() -> None:
    with pytest.raises(ValidationError):
        StartCommand(type="session.start", request_id="", session_id="bad space")
    with pytest.raises(ValidationError):
        StartCommand(type="session.start", request_id="req", hotwords=[" "])  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        HotwordsCommand(type="hotwords.update", request_id="h", hotwords=["x"] * 33)
    command = StartCommand(type="session.start", request_id="req")
    with pytest.raises(ValidationError):
        command.request_id = "changed"  # type: ignore[misc]
