from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import echoforge.asr.factory as factory_module
from echoforge.asr.factory import build_backend_factories
from echoforge.asr.faster_whisper import FasterWhisperFinalizer, FasterWhisperUnavailable
from echoforge.asr.sherpa_onnx import SherpaOnnxConfig, SherpaOnnxStreamingRecognizer
from echoforge.contracts.domain import RevisionStage


class _FakeStream:
    def __init__(self) -> None:
        self.text = ""
        self.finished = False

    def accept_waveform(self, _sample_rate: int, samples: np.ndarray) -> None:
        if samples.size:
            self.text = "你好"

    def input_finished(self) -> None:
        self.finished = True
        self.text = "你好世界"


class _FakeSherpaRecognizer:
    def __init__(self) -> None:
        self.streams: list[_FakeStream] = []

    def create_stream(self) -> _FakeStream:
        stream = _FakeStream()
        self.streams.append(stream)
        return stream

    def is_ready(self, _stream: _FakeStream) -> bool:
        return False

    def decode_stream(self, _stream: _FakeStream) -> None:
        raise AssertionError("fake stream is always ready")


def test_sherpa_adapter_is_lazy_and_emits_monotonic_hypotheses(tmp_path, monkeypatch) -> None:
    module = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    config = SherpaOnnxConfig(model_dir=tmp_path)
    recognizer = _FakeSherpaRecognizer()
    adapter = SherpaOnnxStreamingRecognizer(
        config,
        recognizer_factory=lambda _module, _config: recognizer,
    )
    assert adapter._recognizer is None  # type: ignore[attr-defined]
    partial = adapter.accept_audio(np.ones(320, dtype=np.float32), 16_000)
    assert partial is not None
    assert partial.stage is RevisionStage.PARTIAL
    final = adapter.finalize()
    assert final.stage is RevisionStage.STREAM_FINAL
    assert final.text == "你好世界"
    adapter.reset()
    assert len(recognizer.streams) == 2


def test_sherpa_config_rejects_missing_model_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        SherpaOnnxConfig(model_dir=tmp_path / "missing")


def test_faster_whisper_requires_explicit_local_model(tmp_path) -> None:
    verifier = FasterWhisperFinalizer(tmp_path / "missing")
    with pytest.raises(FasterWhisperUnavailable, match="local Whisper model"):
        verifier.transcribe(np.zeros(320, dtype=np.float32), 16_000)
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "model.bin").write_bytes(b"fixture")
    (incomplete / "config.json").write_text("{}", encoding="utf-8")
    (incomplete / "vocabulary.json").write_text("{}", encoding="utf-8")
    verifier = FasterWhisperFinalizer(incomplete)
    with pytest.raises(FasterWhisperUnavailable, match="tokenizer.json"):
        verifier.transcribe(np.zeros(320, dtype=np.float32), 16_000)


def test_faster_whisper_disables_implicit_downloads(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "whisper-model"
    model_path.mkdir()
    (model_path / "model.bin").write_bytes(b"fixture")
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_path / "vocabulary.txt").write_text("fixture", encoding="utf-8")
    captured: dict[str, object] = {}

    class Model:
        def __init__(self, path: str, **kwargs: object) -> None:
            captured["path"] = path
            captured.update(kwargs)

        def transcribe(self, _audio, **_kwargs):
            return iter(()), object()

    module = types.SimpleNamespace(WhisperModel=Model)
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    verifier = FasterWhisperFinalizer(model_path)
    verifier.transcribe(np.zeros(320, dtype=np.float32), 16_000)

    assert captured["path"] == str(model_path)
    assert captured["local_files_only"] is True


def test_faster_whisper_adapter_uses_endpoint_segments(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "whisper-model"
    model_path.mkdir()

    class Segment:
        text = " 你好"
        start = 0.0
        end = 0.4
        avg_logprob = -0.2

    class Model:
        def transcribe(self, _audio, **_kwargs):
            return iter((Segment(),)), object()

    module = types.SimpleNamespace(WhisperModel=object)
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    verifier = FasterWhisperFinalizer(
        model_path,
        model_factory=lambda *_args: Model(),
    )
    result = verifier.transcribe(np.zeros(6_400, dtype=np.float32), 16_000)
    assert result.stage is RevisionStage.DUAL_PASS_FINAL
    assert result.text == "你好"
    assert result.decoder_score == pytest.approx(-0.2)


def test_fake_factory_remains_deterministic_and_no_optional_imports() -> None:
    factories = build_backend_factories("fake")
    assert factories.name == "deterministic-fake"
    assert factories.evidence["streaming_model"] == "deterministic-protocol-fixture"


def test_real_factory_preserves_static_only_verification(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "stream"
    model_dir.mkdir()
    monkeypatch.setattr(
        factory_module,
        "run_preflight",
        lambda *_args, **_kwargs: {
            "ok": True,
            "static_requirements_ok": True,
            "verification_level": "static",
            "model_load_verified": False,
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

    factories = build_backend_factories(
        "sherpa-onnx",
        model_dir=model_dir,
        provider="cuda",
        dual_pass=False,
    )

    assert factories.static_preflight_ok is True
    assert factories.evidence["preflight"] == "static_requirements_passed"
    assert factories.evidence["verification_level"] == "static"
    assert factories.evidence["model_load_verified"] == "false"
    recognizer = factories.recognizer_factory()
    assert isinstance(recognizer, SherpaOnnxStreamingRecognizer)
    assert recognizer.config.provider == "cuda"
    assert recognizer._recognizer is None  # type: ignore[attr-defined]
