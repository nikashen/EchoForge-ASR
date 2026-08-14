"""FastAPI transport for EchoForge's deterministic streaming core.

The configuration object is dependency-free; the FastAPI application is
loaded lazily so importing the base wheel does not require the ``serve`` extra.
"""

from .config import ServerConfig

__all__ = ["EchoForgeRuntime", "ServerConfig", "create_app"]


def __getattr__(name: str) -> object:
    if name in {"EchoForgeRuntime", "create_app"}:
        from .app import EchoForgeRuntime, create_app

        return {"EchoForgeRuntime": EchoForgeRuntime, "create_app": create_app}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
