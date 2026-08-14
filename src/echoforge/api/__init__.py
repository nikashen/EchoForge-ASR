"""Public application factory for the EchoForge HTTP/WebSocket API."""

from .app import create_app

__all__ = ["create_app"]
