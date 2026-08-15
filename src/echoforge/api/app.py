from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from echoforge.asr.factory import BackendFactories, build_backend_factories
from echoforge.server.config import ServerConfig

if TYPE_CHECKING:
    from echoforge.server.app import FinalizerFactory, RecognizerFactory
else:
    FinalizerFactory = Any
    RecognizerFactory = Any


def create_app(
    *,
    factories: BackendFactories | None = None,
    recognizer_factory: RecognizerFactory | None = None,
    finalizer_factory: FinalizerFactory | None = None,
    config: ServerConfig | None = None,
    allowed_origins: Iterable[str] | None = None,
) -> Any:
    """Canonical app factory used by CLI, tests, and ASGI servers."""

    from echoforge.server.app import create_app as _create_server_app

    selected = factories or build_backend_factories("fake")
    if recognizer_factory is None:
        recognizer_factory = selected.recognizer_factory
    if finalizer_factory is None:
        finalizer_factory = selected.finalizer_factory
    if config is None:
        config = ServerConfig(
            allowed_origins=tuple(allowed_origins)
            if allowed_origins is not None
            else ServerConfig().allowed_origins,
            backend_name=selected.name,
        )
    elif allowed_origins is not None:
        config = ServerConfig(
            allowed_origins=tuple(allowed_origins),
            subprotocol=config.subprotocol,
            max_sessions=config.max_sessions,
            inbound_queue_size=config.inbound_queue_size,
            enqueue_timeout_s=config.enqueue_timeout_s,
            heartbeat_interval_s=config.heartbeat_interval_s,
            heartbeat_timeout_s=config.heartbeat_timeout_s,
            receive_timeout_s=config.receive_timeout_s,
            backend_name=config.backend_name,
        )
    return _create_server_app(
        recognizer_factory=recognizer_factory,
        finalizer_factory=finalizer_factory,
        config=config,
        startup_ready=selected.static_preflight_ok,
    )


__all__ = ["create_app"]
