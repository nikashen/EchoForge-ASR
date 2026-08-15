from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from echoforge import __version__
from echoforge.asr.base import EndpointFinalizer, StreamingRecognizer
from echoforge.audio.pcm import pcm16le_to_float32
from echoforge.contracts.errors import (
    BackendFailureError,
    GenerationConflictError,
    ProtocolError,
    ResourceLimitError,
    SequenceConflictError,
    SessionStateError,
)
from echoforge.contracts.framing import unpack_audio_frame
from echoforge.contracts.messages import (
    FlushCommand,
    HotwordsCommand,
    PingCommand,
    ResetCommand,
    StartCommand,
    StopCommand,
    parse_client_command,
)
from echoforge.streaming.registry import SessionRegistry
from echoforge.streaming.session import StreamingSession
from echoforge.vad import EnergyVAD

from .config import ServerConfig
from .events import event

RecognizerFactory = Callable[[], StreamingRecognizer]
FinalizerFactory = Callable[[], EndpointFinalizer | None]


@dataclass(slots=True)
class EchoForgeRuntime:
    registry: SessionRegistry
    recognizer_factory: RecognizerFactory
    finalizer_factory: FinalizerFactory
    config: ServerConfig
    startup_ready: bool = True
    ready: bool = False
    streaming_model_load_verified: bool = False
    connections: set[Any] = field(default_factory=set)
    connection_tasks: set[asyncio.Task[Any]] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _Inbound:
    kind: Literal["text", "audio"]
    data: str | bytes


@dataclass(frozen=True, slots=True)
class _ConnectionFailure(Exception):
    code: str
    message: str
    close_code: int
    retryable: bool = False

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class _ClientGone(Exception):
    pass


class _GracefulStop(Exception):
    pass


def _default_recognizer_factory() -> StreamingRecognizer:
    # The fixture is imported lazily so importing the HTTP package never
    # initializes an optional model or downloads weights.
    from echoforge.asr.fake import ScriptedStreamingRecognizer

    return ScriptedStreamingRecognizer()


def _default_finalizer_factory() -> EndpointFinalizer | None:
    from echoforge.asr.fake import ScriptedFinalizer

    return ScriptedFinalizer()


def _requested_subprotocols(header: str | None) -> tuple[str, ...]:
    if not header:
        return ()
    return tuple(token.strip() for token in header.split(",") if token.strip())


def _origin_allowed(origin: str | None, config: ServerConfig) -> bool:
    # Header values are compared exactly after removing transport whitespace;
    # paths, wildcards, and implicit origins are not accepted.
    return origin is not None and origin.strip() in config.allowed_origins


class _StreamConnection:
    """One WebSocket connection with one ordered session consumer."""

    def __init__(self, websocket: WebSocket, runtime: EchoForgeRuntime) -> None:
        self.websocket = websocket
        self.runtime = runtime
        self.config = runtime.config
        self.queue: asyncio.Queue[_Inbound] = asyncio.Queue(maxsize=self.config.inbound_queue_size)
        self.session: StreamingSession | None = None
        self.session_id: str | None = None
        self._generation = 0
        self._server_sequence = 0
        self._last_activity = time.monotonic()
        self._starting = False
        self._busy = False
        self._io_lock = asyncio.Lock()
        self._closed = False
        self.vad = EnergyVAD()

    @property
    def generation(self) -> int:
        session = self.session
        return session.snapshot().generation if session is not None else self._generation

    async def _send_event_locked(
        self, event_type: str, payload: dict[str, Any] | None = None
    ) -> None:
        if self._closed:
            return
        message = event(
            event_type,
            session_id=self.session_id,
            generation=self.generation,
            server_sequence=self._server_sequence,
            payload=payload,
        )
        self._server_sequence += 1
        await self.websocket.send_json(message)

    async def send_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        async with self._io_lock:
            if self._closed:
                return
            try:
                await self._send_event_locked(event_type, payload)
            except Exception as exc:
                self._closed = True
                raise _ClientGone() from exc

    async def close(self, code: int = 1000, reason: str = "") -> None:
        async with self._io_lock:
            if self._closed:
                return
            self._closed = True
            with contextlib.suppress(Exception):
                await self.websocket.close(code=code, reason=reason[:120])

    async def error_close(self, failure: _ConnectionFailure) -> None:
        async with self._io_lock:
            if self._closed:
                return
            with contextlib.suppress(Exception):
                await self._send_event_locked(
                    "error",
                    {
                        "code": failure.code,
                        "message": failure.message,
                        "retryable": failure.retryable,
                    },
                )
            self._closed = True
            with contextlib.suppress(Exception):
                await self.websocket.close(code=failure.close_code, reason=failure.code[:120])

    async def receive_loop(self) -> None:
        while not self._closed:
            try:
                receive = self.websocket.receive()
                if self.config.receive_timeout_s is not None:
                    message = await asyncio.wait_for(receive, self.config.receive_timeout_s)
                else:
                    message = await receive
            except asyncio.TimeoutError as exc:
                if self._starting or self._busy:
                    continue
                raise _ConnectionFailure(
                    "RECEIVE_TIMEOUT",
                    "no client message was received before the receive deadline",
                    4408,
                    retryable=True,
                ) from exc
            except WebSocketDisconnect as exc:
                raise _ClientGone() from exc

            message_type = message.get("type")
            if message_type == "websocket.disconnect":
                raise _ClientGone()
            if message_type != "websocket.receive":
                raise _ConnectionFailure(
                    "MESSAGE_TYPE_UNSUPPORTED",
                    "unsupported WebSocket message type",
                    1003,
                )
            text = message.get("text")
            binary = message.get("bytes")
            if text is not None and binary is not None:
                raise _ConnectionFailure(
                    "MESSAGE_AMBIGUOUS", "message contains text and binary data", 1003
                )
            if text is None and binary is None:
                raise _ConnectionFailure("MESSAGE_EMPTY", "WebSocket message has no payload", 1003)
            if text is not None:
                if not isinstance(text, str):
                    raise _ConnectionFailure(
                        "MESSAGE_TEXT_INVALID", "text message is not a string", 1003
                    )
                inbound = _Inbound("text", text)
            else:
                if not isinstance(binary, bytes):
                    raise _ConnectionFailure(
                        "MESSAGE_BINARY_INVALID", "binary message is not bytes", 1003
                    )
                inbound = _Inbound("audio", binary)
            self._last_activity = time.monotonic()
            try:
                await asyncio.wait_for(self.queue.put(inbound), self.config.enqueue_timeout_s)
            except asyncio.TimeoutError as exc:
                raise _ConnectionFailure(
                    "BACKPRESSURE_OVERFLOW",
                    "the per-session inbound queue is full",
                    1013,
                    retryable=True,
                ) from exc

    async def heartbeat_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(self.config.heartbeat_interval_s)
            idle_s = time.monotonic() - self._last_activity
            if idle_s >= self.config.heartbeat_timeout_s and not self._starting and not self._busy:
                raise _ConnectionFailure(
                    "HEARTBEAT_TIMEOUT",
                    "client activity heartbeat timed out",
                    4408,
                    retryable=True,
                )
            if self.session is None:
                continue
            await self.send_event(
                "heartbeat",
                {
                    "idle_ms": round(idle_s * 1000),
                    "timeout_ms": round(self.config.heartbeat_timeout_s * 1000),
                },
            )

    async def consumer_loop(self) -> None:
        while not self._closed:
            inbound = await self.queue.get()
            self._busy = True
            try:
                if inbound.kind == "text":
                    await self.handle_control(inbound.data)  # type: ignore[arg-type]
                else:
                    await self.handle_audio(inbound.data)  # type: ignore[arg-type]
            finally:
                self._busy = False

    async def _require_session(self) -> StreamingSession:
        if self.session is None:
            raise _ConnectionFailure(
                "START_REQUIRED",
                "session.start must be the first client command",
                1008,
            )
        return self.session

    async def handle_control(self, raw: str) -> None:
        try:
            command = parse_client_command(raw)
        except Exception as exc:
            raise _ConnectionFailure(
                "CONTROL_MESSAGE_INVALID",
                "control message must be a valid supported JSON command",
                1008,
            ) from exc

        if isinstance(command, StartCommand):
            await self._handle_start(command)
            return
        if self.session is None:
            raise _ConnectionFailure(
                "START_REQUIRED",
                "session.start must be the first client command",
                1008,
            )
        if isinstance(command, FlushCommand):
            await self._handle_flush(command)
        elif isinstance(command, StopCommand):
            await self._handle_stop(command)
        elif isinstance(command, ResetCommand):
            await self._handle_reset(command)
        elif isinstance(command, PingCommand):
            await self.send_event("pong", {"request_id": command.request_id})
        elif isinstance(command, HotwordsCommand):
            await self._handle_hotwords(command)
        else:  # pragma: no cover - the discriminated adapter is exhaustive
            raise _ConnectionFailure("COMMAND_UNSUPPORTED", "unsupported command", 1008)

    async def _handle_start(self, command: StartCommand) -> None:
        if self.session is not None:
            raise _ConnectionFailure(
                "SESSION_ALREADY_STARTED",
                "only one session.start is allowed per connection",
                1008,
            )
        session_id = command.session_id or f"ws-{uuid4().hex}"
        finalizer_factory: FinalizerFactory
        if command.mode == "streaming":
            finalizer_factory = lambda: None
        else:
            finalizer_factory = self.runtime.finalizer_factory
        self._starting = True
        try:
            session = await asyncio.to_thread(
                self.runtime.registry.create,
                session_id,
                self.runtime.recognizer_factory,
                finalizer_factory,
            )
        except ResourceLimitError as exc:
            raise _ConnectionFailure("SESSION_CAPACITY", str(exc), 1013, retryable=True) from exc
        except ValueError as exc:
            if str(exc) == "session_id is already active":
                raise _ConnectionFailure("SESSION_ID_CONFLICT", str(exc), 1008) from exc
            raise _ConnectionFailure(
                "BACKEND_INIT_FAILED", "unable to initialize the ASR backend", 1011
            ) from exc
        except Exception as exc:
            raise _ConnectionFailure(
                "BACKEND_INIT_FAILED", "unable to initialize the ASR backend", 1011
            ) from exc
        finally:
            self._starting = False
        self.session = session
        self.session_id = session_id
        self._generation = session.generation
        self.vad.reset()
        hotwords_applied = False
        hotwords_reason: str | None = "not_requested"
        if command.hotwords:
            update = getattr(session.recognizer, "update_hotwords", None)
            hotwords_reason = "backend_not_supported"
            if callable(update):
                try:
                    await asyncio.to_thread(update, command.hotwords)
                    hotwords_applied = True
                    hotwords_reason = None
                except Exception:  # noqa: BLE001 - optional backend capability must not close the stream
                    hotwords_reason = "backend_rejected"
        await self.send_event(
            "session.started",
            {
                "request_id": command.request_id,
                "language": command.language,
                "mode": command.mode,
                "sample_rate": command.sample_rate,
                "channels": command.channels,
                "encoding": command.encoding,
                "chunk_duration_ms": command.chunk_duration_ms,
                "hotwords": list(command.hotwords),
                "hotwords_applied": hotwords_applied,
                "hotwords_reason": hotwords_reason,
                "model_id": getattr(session.recognizer, "model_id", "unknown"),
                "verifier_model_id": (
                    getattr(session.finalizer, "model_id", None)
                    if session.finalizer is not None
                    else None
                ),
            },
        )
        if command.hotwords:
            await self.send_event(
                "hotwords.updated",
                {
                    "request_id": command.request_id,
                    "hotwords": list(command.hotwords),
                    "applied": hotwords_applied,
                    "reason": hotwords_reason,
                },
            )

    async def handle_audio(self, raw: bytes) -> None:
        session = await self._require_session()
        try:
            frame = unpack_audio_frame(raw)
            samples = pcm16le_to_float32(frame.pcm16le)
        except ProtocolError as exc:
            raise _ConnectionFailure(
                exc.code, exc.message, exc.close_code or 1003, exc.retryable
            ) from exc
        except ValueError as exc:
            raise _ConnectionFailure("AUDIO_PAYLOAD_INVALID", str(exc), 1003) from exc
        try:
            result = await asyncio.to_thread(session.ingest, frame.sequence, samples)
        except SequenceConflictError as exc:
            raise _ConnectionFailure("SEQUENCE_CONFLICT", str(exc), 1008) from exc
        except ResourceLimitError as exc:
            raise _ConnectionFailure("AUDIO_LIMIT_EXCEEDED", str(exc), 1009) from exc
        except SessionStateError as exc:
            raise _ConnectionFailure("SESSION_STATE_INVALID", str(exc), 1008) from exc
        except BackendFailureError as exc:
            raise _ConnectionFailure("STREAMING_BACKEND_FAILED", str(exc), 1011) from exc
        except ValueError as exc:
            raise _ConnectionFailure("AUDIO_PAYLOAD_INVALID", str(exc), 1003) from exc
        if self.config.backend_name != "deterministic-fake":
            self.runtime.streaming_model_load_verified = True
        if not result.duplicate:
            for vad_event in self.vad.process(samples):
                await self.send_event("vad.event", {"event": vad_event.to_dict()})
        if result.revision is not None:
            await self.send_event("transcript.revision", {"revision": result.revision.to_dict()})
        await self.send_event(
            "audio.ack",
            {
                "sequence": frame.sequence,
                "highest_contiguous_sequence": result.highest_contiguous_sequence,
                "sample_count": frame.sample_count,
                "queued_ms": result.queued_ms,
                "duplicate": result.duplicate,
            },
        )

    async def _handle_flush(self, command: FlushCommand) -> None:
        session = await self._require_session()
        try:
            result = await asyncio.to_thread(
                session.flush, expected_generation=command.expected_generation
            )
        except GenerationConflictError as exc:
            raise _ConnectionFailure("GENERATION_CONFLICT", str(exc), 1008) from exc
        except SessionStateError as exc:
            raise _ConnectionFailure("SESSION_STATE_INVALID", str(exc), 1008) from exc
        except BackendFailureError as exc:
            raise _ConnectionFailure("STREAMING_BACKEND_FAILED", str(exc), 1011) from exc
        for vad_event in self.vad.flush():
            await self.send_event("vad.event", {"event": vad_event.to_dict()})
        for revision in result.revisions:
            await self.send_event("transcript.revision", {"revision": revision.to_dict()})
        await self.send_event(
            "stream.flushed",
            {
                "request_id": command.request_id,
                "utterance_id": result.utterance_id,
                "revision_count": len(result.revisions),
                "verifier_degraded": result.verifier_degraded,
            },
        )

    async def _handle_stop(self, command: StopCommand) -> None:
        session = await self._require_session()
        try:
            revisions = await asyncio.to_thread(session.stop)
        except BackendFailureError as exc:
            raise _ConnectionFailure("STREAMING_BACKEND_FAILED", str(exc), 1011) from exc
        for vad_event in self.vad.flush():
            await self.send_event("vad.event", {"event": vad_event.to_dict()})
        for revision in revisions:
            await self.send_event("transcript.revision", {"revision": revision.to_dict()})
        await self._remove_session(preserve_identity=True)
        try:
            await self.send_event("session.stopped", {"request_id": command.request_id})
        finally:
            self.session_id = None
        await self.close(1000, "normal closure")
        raise _GracefulStop()

    async def _handle_reset(self, command: ResetCommand) -> None:
        session = await self._require_session()
        try:
            result = await asyncio.to_thread(
                session.reset,
                command.reset_id,
                expected_state_version=command.expected_state_version,
                expected_generation=command.expected_generation,
            )
        except GenerationConflictError as exc:
            raise _ConnectionFailure("GENERATION_CONFLICT", str(exc), 1008) from exc
        except SessionStateError as exc:
            raise _ConnectionFailure("SESSION_STATE_INVALID", str(exc), 1008) from exc
        except ValueError as exc:
            raise _ConnectionFailure("RESET_INVALID", str(exc), 1008) from exc
        await self.send_event(
            "session.reset",
            {
                "request_id": command.request_id,
                "reset_id": result.reset_id,
                "generation": result.generation,
                "state_version": result.state_version,
                "removed_samples": result.removed_samples,
                "idempotent_replay": result.idempotent_replay,
            },
        )
        self.vad.reset()

    async def _handle_hotwords(self, command: HotwordsCommand) -> None:
        session = await self._require_session()
        update = getattr(session.recognizer, "update_hotwords", None)
        applied = False
        reason: str | None = "backend_not_supported"
        if callable(update):
            try:
                await asyncio.to_thread(update, command.hotwords)
                applied = True
                reason = None
            except Exception:  # noqa: BLE001 - report unsupported capability without failing ASR
                reason = "backend_rejected"
        await self.send_event(
            "hotwords.updated",
            {
                "request_id": command.request_id,
                "hotwords": list(command.hotwords),
                "applied": applied,
                "reason": reason,
            },
        )

    async def _remove_session(self, *, preserve_identity: bool = False) -> None:
        session_id = self.session_id
        if self.session is not None:
            self._generation = self.session.snapshot().generation
        self.session = None
        if not preserve_identity:
            self.session_id = None
        if session_id is not None:
            await asyncio.to_thread(self.runtime.registry.remove, session_id)

    async def cleanup(self) -> None:
        with contextlib.suppress(Exception):
            await self._remove_session()

    async def run(self) -> None:
        tasks = {
            asyncio.create_task(self.receive_loop(), name="echoforge-ws-receive"),
            asyncio.create_task(self.consumer_loop(), name="echoforge-ws-consumer"),
            asyncio.create_task(self.heartbeat_loop(), name="echoforge-ws-heartbeat"),
        }
        failure: _ConnectionFailure | None = None
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    task.result()
                except _ConnectionFailure as exc:
                    failure = failure or exc
                except (_ClientGone, WebSocketDisconnect, _GracefulStop):
                    pass
                except asyncio.CancelledError:
                    pass
                except Exception as exc:  # noqa: BLE001 - transport must fail closed on unknown backend errors
                    failure = failure or _ConnectionFailure(
                        "INTERNAL_SERVER_ERROR",
                        "unexpected WebSocket server failure",
                        1011,
                    )
                    _ = exc
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if failure is not None:
                await self.error_close(failure)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.cleanup()


def create_app(
    *,
    recognizer_factory: RecognizerFactory | None = None,
    finalizer_factory: FinalizerFactory | None = None,
    config: ServerConfig | None = None,
    allowed_origins: Iterable[str] | None = None,
    startup_ready: bool = True,
) -> FastAPI:
    """Create an isolated FastAPI application with injected ASR factories."""

    if config is None:
        selected_config = ServerConfig(
            allowed_origins=tuple(allowed_origins)
            if allowed_origins is not None
            else ServerConfig().allowed_origins
        )
    elif allowed_origins is None:
        selected_config = config
    else:
        selected_config = ServerConfig(
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
    runtime = EchoForgeRuntime(
        registry=SessionRegistry(max_sessions=selected_config.max_sessions),
        recognizer_factory=recognizer_factory or _default_recognizer_factory,
        finalizer_factory=finalizer_factory or _default_finalizer_factory,
        config=selected_config,
        startup_ready=startup_ready,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime.ready = runtime.startup_ready
        try:
            yield
        finally:
            runtime.ready = False
            current = asyncio.current_task()
            active_connections = tuple(runtime.connections)
            for connection in active_connections:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(connection.close(1001, "server shutdown"), 1.0)
            active_tasks = tuple(runtime.connection_tasks)
            for task in active_tasks:
                if task is not current and not task.done():
                    task.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            with contextlib.suppress(Exception):
                await asyncio.to_thread(runtime.registry.close_all)

    app = FastAPI(title="EchoForge-ASR", version=__version__, lifespan=lifespan)
    app.state.echoforge_runtime = runtime

    web_dir = Path(__file__).resolve().parent.parent / "web"
    if web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=web_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(web_dir / "index.html")

    @app.get("/api/v1/health")
    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "echoforge-asr",
            "version": __version__,
            "schema_version": "echoforge.http/v1",
        }

    @app.get("/api/v1/readiness")
    @app.get("/readiness")
    async def readiness() -> JSONResponse:
        is_fixture = selected_config.backend_name == "deterministic-fake"
        model_status = (
            "fixture"
            if is_fixture
            else (
                "static_preflight_failed"
                if not runtime.startup_ready
                else (
                    "streaming_load_verified"
                    if runtime.streaming_model_load_verified
                    else "static_preflight_passed"
                )
            )
        )
        payload = {
            "status": "ready" if runtime.ready else "not_ready",
            "service": "echoforge-asr",
            "backend": selected_config.backend_name,
            "static_preflight": (
                "not_required" if is_fixture else ("passed" if runtime.startup_ready else "failed")
            ),
            "model_status": model_status,
            "streaming_model_load_verified": runtime.streaming_model_load_verified,
            "active_sessions": runtime.registry.active_count(),
            "max_sessions": selected_config.max_sessions,
        }
        return JSONResponse(payload, status_code=200 if runtime.ready else 503)

    @app.websocket("/api/v1/stream")
    async def stream(websocket: WebSocket) -> None:
        if not _origin_allowed(websocket.headers.get("origin"), selected_config):
            with contextlib.suppress(Exception):
                await websocket.close(code=1008, reason="ORIGIN_NOT_ALLOWED")
            return
        requested = _requested_subprotocols(websocket.headers.get("sec-websocket-protocol"))
        if selected_config.subprotocol not in requested:
            with contextlib.suppress(Exception):
                await websocket.close(code=1008, reason="SUBPROTOCOL_REQUIRED")
            return
        await websocket.accept(subprotocol=selected_config.subprotocol)
        if not runtime.ready:
            with contextlib.suppress(Exception):
                await websocket.close(code=1013, reason="SERVICE_NOT_READY")
            return
        connection = _StreamConnection(websocket, runtime)
        runtime.connections.add(connection)
        if not runtime.ready:
            runtime.connections.discard(connection)
            await connection.close(1013, "server shutdown")
            return
        current = asyncio.current_task()
        if current is not None:
            runtime.connection_tasks.add(current)
        try:
            await connection.run()
        finally:
            runtime.connections.discard(connection)
            if current is not None:
                runtime.connection_tasks.discard(current)

    return app


__all__ = [
    "EchoForgeRuntime",
    "FinalizerFactory",
    "RecognizerFactory",
    "ServerConfig",
    "create_app",
]
