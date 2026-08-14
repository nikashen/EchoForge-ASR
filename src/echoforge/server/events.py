from __future__ import annotations

from typing import Any
from uuid import uuid4

SCHEMA_VERSION = "echoforge.ws/v1"


def event(
    event_type: str,
    *,
    session_id: str | None,
    generation: int,
    server_sequence: int,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common envelope used by every JSON server event."""

    return {
        "schema_version": SCHEMA_VERSION,
        "type": event_type,
        "event_id": uuid4().hex,
        "session_id": session_id,
        "generation": generation,
        "server_sequence": server_sequence,
        "payload": payload or {},
    }
