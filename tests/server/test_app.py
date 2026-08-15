from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from echoforge.asr.fake import ScriptedFinalizer, ScriptedStreamingRecognizer
from echoforge.audio.pcm import float32_to_pcm16le
from echoforge.contracts.framing import pack_audio_frame
from echoforge.server import ServerConfig, create_app

ORIGIN = "http://testserver"


@pytest.fixture()
def config() -> ServerConfig:
    return ServerConfig(
        allowed_origins=(ORIGIN,),
        heartbeat_interval_s=0.05,
        heartbeat_timeout_s=0.20,
    )


@pytest.fixture()
def client(config: ServerConfig) -> Iterator[TestClient]:
    app = create_app(config=config)
    with TestClient(app) as value:
        yield value


def connect(client: TestClient):
    return client.websocket_connect(
        "/api/v1/stream",
        headers={"Origin": ORIGIN},
        subprotocols=["echoforge.v1"],
    )


def frame(sequence: int, count: int = 3200) -> bytes:
    return pack_audio_frame(sequence, float32_to_pcm16le(np.zeros(count, dtype=np.float32)))


def test_health_and_readiness(client: TestClient, config: ServerConfig) -> None:
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    readiness = client.get("/api/v1/readiness")
    assert readiness.status_code == 200
    assert readiness.json()["status"] == "ready"
    assert readiness.json()["model_status"] == "fixture"
    assert readiness.json()["streaming_model_load_verified"] is False

    not_ready_app = create_app(config=config, startup_ready=False)
    with TestClient(not_ready_app) as not_ready_client:
        not_ready = not_ready_client.get("/api/v1/readiness")
        assert not_ready.status_code == 503
        assert not_ready.json()["status"] == "not_ready"

    real_config = ServerConfig(allowed_origins=(ORIGIN,), backend_name="sherpa-onnx")
    failed_real_app = create_app(config=real_config, startup_ready=False)
    with TestClient(failed_real_app) as failed_real_client:
        failed_real = failed_real_client.get("/api/v1/readiness")
        assert failed_real.status_code == 503
        assert failed_real.json()["static_preflight"] == "failed"
        assert failed_real.json()["model_status"] == "static_preflight_failed"


@pytest.mark.parametrize(
    ("headers", "subprotocols", "reason"),
    [
        ({"Origin": "http://evil.example"}, ["echoforge.v1"], "ORIGIN_NOT_ALLOWED"),
        ({"Origin": ORIGIN}, [], "SUBPROTOCOL_REQUIRED"),
    ],
)
def test_handshake_policy_is_strict(
    client: TestClient,
    headers: dict[str, str],
    subprotocols: list[str],
    reason: str,
) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as raised,
        client.websocket_connect("/api/v1/stream", headers=headers, subprotocols=subprotocols),
    ):
        pass
    assert raised.value.code == 1008
    assert raised.value.reason == reason


def test_start_is_required_before_audio(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as raised, connect(client) as websocket:
        websocket.send_bytes(frame(0))
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["payload"]["code"] == "START_REQUIRED"
        websocket.receive_json()
    assert raised.value.code == 1008


def test_start_hotwords_report_backend_support_explicitly(client: TestClient) -> None:
    with connect(client) as websocket:
        websocket.send_json(
            {
                "type": "session.start",
                "request_id": "start",
                "session_id": "hotwords",
                "hotwords": ["EchoForge"],
            }
        )
        assert websocket.receive_json()["type"] == "session.started"
        updated = websocket.receive_json()
        assert updated["type"] == "hotwords.updated"
        assert updated["payload"] == {
            "request_id": "start",
            "hotwords": ["EchoForge"],
            "applied": False,
            "reason": "backend_not_supported",
        }


def test_stream_lifecycle_and_registry_cleanup(client: TestClient) -> None:
    app = client.app
    with connect(client) as websocket:
        websocket.send_json(
            {
                "type": "session.start",
                "request_id": "start-1",
                "session_id": "demo-session",
                "mode": "dual_pass",
            }
        )
        started = websocket.receive_json()
        assert started["type"] == "session.started"
        assert started["session_id"] == "demo-session"

        websocket.send_bytes(frame(0))
        first = websocket.receive_json()
        second = websocket.receive_json()
        assert {first["type"], second["type"]} == {"transcript.revision", "audio.ack"}
        assert any(
            message.get("payload", {}).get("revision", {}).get("stage") == "partial"
            for message in (first, second)
            if message["type"] == "transcript.revision"
        )

        websocket.send_json(
            {"type": "stream.flush", "request_id": "flush-1", "expected_generation": 0}
        )
        events = [websocket.receive_json() for _ in range(4)]
        assert events[-1]["type"] == "stream.flushed"
        assert events[-1]["payload"]["verifier_degraded"] is False

        websocket.send_json({"type": "session.stop", "request_id": "stop-1"})
        assert websocket.receive_json()["type"] == "session.stopped"
    assert app.state.echoforge_runtime.registry.active_count() == 0


def test_duplicate_audio_is_idempotent_and_gap_closes(client: TestClient) -> None:
    with connect(client) as websocket:
        websocket.send_json({"type": "session.start", "request_id": "start", "session_id": "seq"})
        websocket.receive_json()
        payload = frame(0)
        websocket.send_bytes(payload)
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_bytes(payload)
        duplicate = websocket.receive_json()
        assert duplicate["type"] == "audio.ack"
        assert duplicate["payload"]["duplicate"] is True
        websocket.send_bytes(frame(2))
        error = websocket.receive_json()
        assert error["payload"]["code"] == "SEQUENCE_CONFLICT"
        with pytest.raises(WebSocketDisconnect) as raised:
            websocket.receive_json()
    assert raised.value.code == 1008


def test_invalid_binary_frame_uses_protocol_close_code(client: TestClient) -> None:
    with connect(client) as websocket:
        websocket.send_json({"type": "session.start", "request_id": "start", "session_id": "bad"})
        websocket.receive_json()
        websocket.send_bytes(b"bad")
        error = websocket.receive_json()
        assert error["payload"]["code"] == "AUDIO_FRAME_TRUNCATED"
        with pytest.raises(WebSocketDisconnect) as raised:
            websocket.receive_json()
    assert raised.value.code == 1003


def test_unsupported_hotwords_are_reported_without_closing_session(client: TestClient) -> None:
    with connect(client) as websocket:
        websocket.send_json(
            {
                "type": "session.start",
                "request_id": "start-hotwords",
                "session_id": "hotwords",
                "hotwords": ["EchoForge"],
            }
        )
        started = websocket.receive_json()
        assert started["payload"]["hotwords_applied"] is False
        initial = websocket.receive_json()
        assert initial["type"] == "hotwords.updated"
        assert initial["payload"]["reason"] == "backend_not_supported"
        websocket.send_json(
            {
                "type": "hotwords.update",
                "request_id": "update-hotwords",
                "hotwords": ["图神经网络"],
            }
        )
        updated = websocket.receive_json()
        assert updated["type"] == "hotwords.updated"
        assert updated["payload"]["applied"] is False
        websocket.send_json({"type": "ping", "request_id": "still-open"})
        assert websocket.receive_json()["type"] == "pong"


def test_heartbeat_timeout_closes_idle_connection(client: TestClient) -> None:
    with connect(client) as websocket:
        websocket.send_json({"type": "session.start", "request_id": "start", "session_id": "idle"})
        assert websocket.receive_json()["type"] == "session.started"
        heartbeats = []
        disconnected: WebSocketDisconnect | None = None
        for _ in range(8):
            try:
                message = websocket.receive_json()
            except WebSocketDisconnect as exc:
                disconnected = exc
                break
            heartbeats.append(message)
        assert any(item["type"] == "heartbeat" for item in heartbeats)
        timeout_errors = [item for item in heartbeats if item["type"] == "error"]
        if timeout_errors:
            assert timeout_errors[-1]["payload"]["code"] == "HEARTBEAT_TIMEOUT"
        assert disconnected is not None
    assert disconnected.code == 4408


def test_backend_failure_is_fail_closed(client: TestClient) -> None:
    def failing_factory() -> ScriptedStreamingRecognizer:
        return ScriptedStreamingRecognizer(fail_after_samples=1)

    app = create_app(
        recognizer_factory=failing_factory,
        finalizer_factory=lambda: ScriptedFinalizer(),
        config=client.app.state.echoforge_runtime.config,
    )
    with TestClient(app) as local_client:
        with connect(local_client) as websocket:
            websocket.send_json(
                {"type": "session.start", "request_id": "start", "session_id": "fail"}
            )
            websocket.receive_json()
            websocket.send_bytes(frame(0, count=3200))
            error = websocket.receive_json()
            assert error["payload"]["code"] == "STREAMING_BACKEND_FAILED"
            with pytest.raises(WebSocketDisconnect) as raised:
                websocket.receive_json()
        assert raised.value.code == 1011


def test_lifespan_closes_active_connection() -> None:
    config = ServerConfig(
        allowed_origins=(ORIGIN,), heartbeat_interval_s=0.05, heartbeat_timeout_s=0.20
    )
    app = create_app(config=config)
    with TestClient(app) as local_client:
        websocket = local_client.websocket_connect(
            "/api/v1/stream", headers={"Origin": ORIGIN}, subprotocols=["echoforge.v1"]
        )
        websocket.__enter__()
        websocket.send_json(
            {"type": "session.start", "request_id": "start", "session_id": "shutdown"}
        )
        assert websocket.receive_json()["type"] == "session.started"
    assert app.state.echoforge_runtime.registry.active_count() == 0
    assert app.state.echoforge_runtime.ready is False
