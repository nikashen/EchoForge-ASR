from __future__ import annotations

from fastapi.testclient import TestClient

from echoforge.api.app import create_app
from echoforge.cli import main


def test_cli_smoke_and_preflight_report_contracts(capsys) -> None:
    assert main(["smoke", "--json"]) == 0
    output = capsys.readouterr().out
    assert '"dual_pass_final"' in output
    assert '"deterministic-fake"' in output
    assert main(["preflight", "--backend", "fake", "--json"]) == 0
    output = capsys.readouterr().out
    assert '"verification_level": "fixture"' in output
    assert main(["preflight", "--backend", "sherpa-onnx", "--json"]) == 1
    output = capsys.readouterr().out
    assert '"verification_level": "static"' in output
    assert '"model_load_verified": false' in output


def test_app_serves_voice_lab_static_shell() -> None:
    app = create_app(allowed_origins=("http://testserver",))
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "EchoForge-ASR" in page.text
        assert "This page does not run transformed audio through ASR." in page.text
        script = client.get("/static/app.js")
        assert script.status_code == 200
        assert "Deterministic replay" in script.text
        assert "FIXTURE ONLY" in script.text
        assert "0.06" not in script.text
