from __future__ import annotations

from dataclasses import dataclass


class EchoForgeError(Exception):
    """Base class for expected domain failures."""


@dataclass(frozen=True, slots=True)
class ProtocolError(EchoForgeError):
    code: str
    message: str
    retryable: bool = False
    close_code: int | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class SessionStateError(EchoForgeError):
    pass


class SequenceConflictError(EchoForgeError):
    pass


class ResourceLimitError(EchoForgeError):
    pass


class BackendFailureError(EchoForgeError):
    pass


class GenerationConflictError(EchoForgeError):
    pass
