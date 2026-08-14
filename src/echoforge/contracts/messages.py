from __future__ import annotations

import json
import re
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StartCommand(StrictModel):
    type: Literal["session.start"]
    request_id: str = Field(min_length=1, max_length=96)
    session_id: str | None = Field(default=None, max_length=96)
    language: Literal["zh-CN"] = "zh-CN"
    mode: Literal["streaming", "dual_pass"] = "dual_pass"
    sample_rate: Literal[16000] = 16000
    channels: Literal[1] = 1
    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    chunk_duration_ms: int = Field(default=40, ge=20, le=100)
    hotwords: tuple[str, ...] = Field(default_factory=tuple, max_length=32)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None) -> str | None:
        if value is not None and not SESSION_ID.fullmatch(value):
            raise ValueError("session_id contains unsupported characters")
        return value

    @field_validator("hotwords")
    @classmethod
    def validate_hotwords(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() or len(item) > 64 for item in values):
            raise ValueError("hotwords must be non-empty and at most 64 characters")
        return tuple(dict.fromkeys(item.strip() for item in values))


class FlushCommand(StrictModel):
    type: Literal["stream.flush"]
    request_id: str = Field(min_length=1, max_length=96)
    expected_generation: int = Field(default=0, ge=0)


class StopCommand(StrictModel):
    type: Literal["session.stop"]
    request_id: str = Field(min_length=1, max_length=96)


class ResetCommand(StrictModel):
    type: Literal["session.reset"]
    request_id: str = Field(min_length=1, max_length=96)
    reset_id: str = Field(min_length=1, max_length=96)
    confirm: Literal[True] = True
    expected_state_version: int = Field(default=0, ge=0)
    expected_generation: int = Field(default=0, ge=0)


class PingCommand(StrictModel):
    type: Literal["ping"]
    request_id: str = Field(min_length=1, max_length=96)


class HotwordsCommand(StrictModel):
    type: Literal["hotwords.update"]
    request_id: str = Field(min_length=1, max_length=96)
    hotwords: tuple[str, ...] = Field(max_length=32)

    @field_validator("hotwords")
    @classmethod
    def validate_hotwords(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() or len(item) > 64 for item in values):
            raise ValueError("hotwords must be non-empty and at most 64 characters")
        return tuple(dict.fromkeys(item.strip() for item in values))


ClientCommand: TypeAlias = Annotated[
    StartCommand | FlushCommand | StopCommand | ResetCommand | PingCommand | HotwordsCommand,
    Field(discriminator="type"),
]
COMMAND_ADAPTER: TypeAdapter[ClientCommand] = TypeAdapter(ClientCommand)


def parse_client_command(raw: str | bytes) -> ClientCommand:
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > 16 * 1024:
            raise ValueError("control message exceeds 16 KiB")
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("control message must be valid UTF-8 JSON") from exc
    return COMMAND_ADAPTER.validate_python(value)
