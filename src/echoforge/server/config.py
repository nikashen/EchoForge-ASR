from __future__ import annotations

import math
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Transport limits and policy for the HTTP/WebSocket service.

    Origins are deliberately an explicit allow-list.  A wildcard is not
    accepted because the stream can carry microphone audio and session data.
    """

    allowed_origins: tuple[str, ...] = field(
        default_factory=lambda: (
            "http://127.0.0.1:8090",
            "http://localhost:8090",
            "http://127.0.0.1:8000",
            "http://localhost:8000",
        )
    )
    subprotocol: str = "echoforge.v1"
    max_sessions: int = 4
    inbound_queue_size: int = 32
    enqueue_timeout_s: float = 0.25
    heartbeat_interval_s: float = 10.0
    heartbeat_timeout_s: float = 35.0
    receive_timeout_s: float | None = None
    backend_name: str = "deterministic-fake"

    def __post_init__(self) -> None:
        origins = tuple(dict.fromkeys(self.allowed_origins))
        if not origins:
            raise ValueError("allowed_origins must not be empty")
        for origin in origins:
            if not isinstance(origin, str) or origin == "*":
                raise ValueError("origins must be explicit absolute origins")
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"invalid origin: {origin!r}")
            if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
                raise ValueError(f"origin must not contain a path or query: {origin!r}")
        if self.subprotocol != "echoforge.v1":
            raise ValueError("streaming v1 requires the echoforge.v1 subprotocol")
        if isinstance(self.max_sessions, bool) or not 1 <= self.max_sessions <= 128:
            raise ValueError("max_sessions must be in [1, 128]")
        if isinstance(self.inbound_queue_size, bool) or not 1 <= self.inbound_queue_size <= 256:
            raise ValueError("inbound_queue_size must be in [1, 256]")
        if (
            isinstance(self.enqueue_timeout_s, bool)
            or not math.isfinite(self.enqueue_timeout_s)
            or self.enqueue_timeout_s <= 0
        ):
            raise ValueError("enqueue_timeout_s must be positive")
        if (
            isinstance(self.heartbeat_interval_s, bool)
            or isinstance(self.heartbeat_timeout_s, bool)
            or not math.isfinite(self.heartbeat_interval_s)
            or not math.isfinite(self.heartbeat_timeout_s)
            or self.heartbeat_interval_s <= 0
            or self.heartbeat_timeout_s <= self.heartbeat_interval_s
        ):
            raise ValueError("heartbeat timeout must be greater than heartbeat interval")
        if self.receive_timeout_s is not None and (
            isinstance(self.receive_timeout_s, bool)
            or not math.isfinite(self.receive_timeout_s)
            or self.receive_timeout_s <= 0
        ):
            raise ValueError("receive_timeout_s must be positive when set")
        object.__setattr__(self, "allowed_origins", origins)
