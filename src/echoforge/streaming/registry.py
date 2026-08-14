from __future__ import annotations

import threading
from collections.abc import Callable

from echoforge.asr.base import EndpointFinalizer, StreamingRecognizer
from echoforge.contracts.errors import ResourceLimitError

from .session import StreamingSession


class SessionRegistry:
    def __init__(self, *, max_sessions: int = 4) -> None:
        if max_sessions < 1 or max_sessions > 128:
            raise ValueError("max_sessions must be in [1, 128]")
        self.max_sessions = max_sessions
        self._sessions: dict[str, StreamingSession] = {}
        self._lock = threading.RLock()

    def create(
        self,
        session_id: str,
        recognizer_factory: Callable[[], StreamingRecognizer],
        finalizer_factory: Callable[[], EndpointFinalizer | None],
    ) -> StreamingSession:
        with self._lock:
            if session_id in self._sessions:
                raise ValueError("session_id is already active")
            if len(self._sessions) >= self.max_sessions:
                raise ResourceLimitError("concurrent session limit reached")
            session = StreamingSession(
                session_id=session_id,
                recognizer=recognizer_factory(),
                finalizer=finalizer_factory(),
            )
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> StreamingSession:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError("session not found") from exc

    def remove(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                session.stop()

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def close_all(self) -> None:
        with self._lock:
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.stop()
