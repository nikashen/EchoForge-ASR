from __future__ import annotations

from pathlib import Path

import pytest

from echoforge.asr import preflight


def _write_streaming_model(path: Path) -> None:
    path.mkdir()
    for name in (
        "tokens.txt",
        "encoder-epoch-99-avg-1.onnx",
        "decoder-epoch-99-avg-1.onnx",
        "joiner-epoch-99-avg-1.onnx",
    ):
        (path / name).write_bytes(b"fixture")


def test_fake_preflight_is_ready() -> None:
    result = preflight.run_preflight("fake")
    assert result["ok"] is True
    assert result["backend"] == "deterministic-fake"
    assert result["verification_level"] == "fixture"
    assert result["model_load_verified"] is False
    assert result["runtime_probe_required"] is False


def test_real_preflight_fails_closed_for_missing_paths() -> None:
    result = preflight.run_preflight(
        "sherpa-onnx",
        model_dir=Path("missing-streaming-model"),
        verifier_model=Path("missing-verifier-model"),
    )
    assert result["ok"] is False
    assert any(check["name"] == "model_dir" and not check["ok"] for check in result["checks"])


def test_real_preflight_checks_expected_files(monkeypatch, tmp_path: Path) -> None:
    stream = tmp_path / "stream"
    verifier = tmp_path / "verifier"
    _write_streaming_model(stream)
    verifier.mkdir()
    (verifier / "model.bin").write_bytes(b"fixture")
    (verifier / "config.json").write_text("{}", encoding="utf-8")
    (verifier / "tokenizer.json").write_text("{}", encoding="utf-8")
    (verifier / "vocabulary.txt").write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(
        preflight,
        "_runtime_check",
        lambda _name, _symbols: (True, "test runtime available"),
    )
    monkeypatch.setattr(
        preflight,
        "_ctranslate2_check",
        lambda: (True, "test ctranslate2 runtime available"),
    )

    result = preflight.run_preflight("sherpa-onnx", model_dir=stream, verifier_model=verifier)
    assert result["ok"] is True
    assert result["static_requirements_ok"] is True
    assert result["verification_level"] == "static"
    assert result["model_load_verified"] is False
    assert result["runtime_probe_required"] is True
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["verifier_tokenizer_json"]["status"] == "passed"
    assert checks["verifier_vocabulary"]["status"] == "passed"


@pytest.mark.parametrize("provider", ["cpu", "cuda"])
def test_static_provider_checks_do_not_overclaim_cuda(
    monkeypatch, tmp_path: Path, provider: str
) -> None:
    stream = tmp_path / "stream"
    _write_streaming_model(stream)
    monkeypatch.setattr(
        preflight,
        "_runtime_check",
        lambda _name, _symbols: (True, "test runtime available"),
    )

    result = preflight.run_preflight(
        "sherpa-onnx",
        model_dir=stream,
        provider=provider,
        dual_pass=False,
    )

    assert result["ok"] is True
    assert result["static_requirements_ok"] is True
    assert result["verification_level"] == "static"
    assert result["model_load_verified"] is False
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["provider"]["status"] == "passed"
    if provider == "cuda":
        assert checks["cuda_compatibility"]["ok"] is None
        assert checks["cuda_compatibility"]["status"] == "not_verified"
    else:
        assert "cuda_compatibility" not in checks


@pytest.mark.parametrize(
    ("version", "expected_ok"),
    [("4.5.0", True), ("4.8.1", False)],
)
def test_windows_ctranslate2_version_policy(monkeypatch, version: str, expected_ok: bool) -> None:
    monkeypatch.setattr(preflight.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        preflight,
        "_runtime_check",
        lambda _name, _symbols: (True, "test runtime available"),
    )
    monkeypatch.setattr(preflight.importlib.metadata, "version", lambda _name: version)
    ok, detail = preflight._ctranslate2_check()
    assert ok is expected_ok
    assert version in detail
