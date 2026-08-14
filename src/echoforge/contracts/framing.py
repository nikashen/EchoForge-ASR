from __future__ import annotations

import struct
from dataclasses import dataclass

from .errors import ProtocolError

AUDIO_MAGIC = b"EFA1"
AUDIO_HEADER = struct.Struct(">4sII")
MAX_AUDIO_FRAME_BYTES = 16 * 1024
MAX_AUDIO_SAMPLES = (MAX_AUDIO_FRAME_BYTES - AUDIO_HEADER.size) // 2


@dataclass(frozen=True, slots=True)
class BinaryAudioFrame:
    sequence: int
    sample_count: int
    pcm16le: bytes


def pack_audio_frame(sequence: int, pcm16le: bytes) -> bytes:
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 0
        or sequence > 2**32 - 1
    ):
        raise ValueError("sequence must be a uint32")
    if not isinstance(pcm16le, bytes) or not pcm16le or len(pcm16le) % 2:
        raise ValueError("PCM16LE payload must be non-empty and have an even byte length")
    sample_count = len(pcm16le) // 2
    if sample_count > MAX_AUDIO_SAMPLES:
        raise ValueError("audio frame exceeds the v1 sample limit")
    return AUDIO_HEADER.pack(AUDIO_MAGIC, sequence, sample_count) + pcm16le


def unpack_audio_frame(data: bytes) -> BinaryAudioFrame:
    if not isinstance(data, bytes) or len(data) < AUDIO_HEADER.size:
        raise ProtocolError(
            "AUDIO_FRAME_TRUNCATED", "binary audio frame header is incomplete", close_code=1003
        )
    if len(data) > MAX_AUDIO_FRAME_BYTES:
        raise ProtocolError(
            "AUDIO_FRAME_TOO_LARGE", "binary audio frame exceeds 16 KiB", close_code=1009
        )
    magic, sequence, sample_count = AUDIO_HEADER.unpack(data[: AUDIO_HEADER.size])
    if magic != AUDIO_MAGIC:
        raise ProtocolError(
            "AUDIO_MAGIC_INVALID", "binary audio frame magic is invalid", close_code=1003
        )
    payload = data[AUDIO_HEADER.size :]
    if sample_count == 0 or sample_count > MAX_AUDIO_SAMPLES:
        raise ProtocolError(
            "AUDIO_SAMPLE_COUNT_INVALID", "sample count is outside the v1 limit", close_code=1003
        )
    if len(payload) != sample_count * 2:
        raise ProtocolError(
            "AUDIO_LENGTH_MISMATCH", "sample count does not match PCM payload", close_code=1003
        )
    return BinaryAudioFrame(sequence=sequence, sample_count=sample_count, pcm16le=payload)
