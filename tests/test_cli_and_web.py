from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from echoforge import cli
from echoforge.api.app import create_app
from echoforge.cli import main


def test_cli_smoke_report_contract(capsys) -> None:
    assert main(["smoke", "--json"]) == 0
    output = capsys.readouterr().out
    assert '"dual_pass_final"' in output
    assert '"deterministic-fake"' in output


def test_preflight_cli_exit_code_contract(capsys) -> None:
    assert main(["preflight", "--backend", "fake", "--json"]) == 0
    output = capsys.readouterr().out
    assert '"verification_level": "fixture"' in output
    assert main(["preflight", "--backend", "sherpa-onnx", "--json"]) == 1
    output = capsys.readouterr().out
    assert '"verification_level": "static"' in output
    assert '"model_load_verified": false' in output
    with pytest.raises(SystemExit) as raised:
        main(["preflight", "--provider", "not-a-provider"])
    assert raised.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_preflight_cli_prints_cuda_as_unverified(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "run_preflight",
        lambda *_args, **_kwargs: {
            "ok": True,
            "static_requirements_ok": True,
            "backend": "sherpa-onnx",
            "verification_level": "static",
            "model_load_verified": False,
            "runtime_probe_required": True,
            "checks": [
                {
                    "name": "cuda_compatibility",
                    "ok": None,
                    "status": "not_verified",
                    "detail": "runtime probe required",
                }
            ],
        },
    )

    assert main(["preflight", "--provider", "cuda"]) == 0
    output = capsys.readouterr().out
    assert "static prerequisites PASSED (model load unverified)" in output
    assert "cuda_compatibility=NOT_VERIFIED" in output


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
        assert "STATIC ONLY / MODEL UNVERIFIED" in script.text
        assert "0.06" not in script.text
