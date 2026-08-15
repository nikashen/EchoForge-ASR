from __future__ import annotations

from pathlib import Path

from echoforge.asr import preflight


def test_fake_preflight_is_ready() -> None:
    result = preflight.run_preflight("fake")
    assert result["ok"] is True
    assert result["backend"] == "deterministic-fake"


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
    stream.mkdir()
    verifier.mkdir()
    for name in (
        "tokens.txt",
        "encoder-epoch-99-avg-1.onnx",
        "decoder-epoch-99-avg-1.onnx",
        "joiner-epoch-99-avg-1.onnx",
    ):
        (stream / name).write_bytes(b"fixture")
    (verifier / "model.bin").write_bytes(b"fixture")
    (verifier / "config.json").write_text("{}", encoding="utf-8")
    (verifier / "tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        preflight,
        "_runtime_check",
        lambda _name, _symbols: (True, "test runtime available"),
    )

    result = preflight.run_preflight("sherpa-onnx", model_dir=stream, verifier_model=verifier)
    assert result["ok"] is True
