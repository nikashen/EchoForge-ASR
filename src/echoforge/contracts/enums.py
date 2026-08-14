from __future__ import annotations

from enum import Enum


class StringEnum(str, Enum):
    """Small StrEnum-compatible base that also works on Python 3.10."""

    def __str__(self) -> str:
        return str(self.value)
