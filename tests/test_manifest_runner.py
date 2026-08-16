from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

import echoforge.evaluation.runner as runner_module
from echoforge.asr.factory import BackendFactories
from echoforge.asr.fake import ScriptedFinalizer, ScriptedStreamingRecognizer
from echoforge.evaluation.runner import ManifestRunnerError, run_manifest


def _write_wav(path: Path, *, frames: int = 3200) -> str:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(b"\x00\x00" * frames)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, audio_sha256: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "echoforge.eval-manifest/v1",
                "dataset": {
                    "name": "unit",
                    "source": "generated fixture",
                    "source_url": "https://example.invalid/unit",
                    "license": "Reviewed-Test-License",
                    "license_page_sha256": "b" * 64,
                    "download_manifest_sha256": "c" * 64,
                    "extraction_marker_sha256": "e" * 64,
                    "extraction_inventory_sha256": "f" * 64,
                    "transcript_sha256": "d" * 64,
                    "speaker_policy": "speaker-disjoint",
                    "audio_protocol": "16 kHz mono PCM16LE",
                    "raw_audio_in_repository": False,
                    "selection": {
                        "splits": ["dev"],
                        "speaker_limit_per_split": None,
                        "utterances_per_speaker": None,
                        "extraction_speaker_limit_per_split": None,
                        "selected_speakers": {"dev": ["speaker-1"]},
                        "rows": 1,
                    },
                },
                "protocol_id": "unit-protocol-v1",
                "evaluation_authorized": True,
                "frozen": False,
                "normalization": "echoforge.zh-normalizer/v1",
                "rows": [
                    {
                        "id": "utterance-1",
                        "speaker_id": "speaker-1",
                        "split": "dev",
                        "audio_relpath": "utterance-1.wav",
                        "audio_sha256": audio_sha256,
                        "reference": "你好世界",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class _ConfiguredRecognizer(ScriptedStreamingRecognizer):
    def __init__(self) -> None:
        super().__init__(partials=("你好",), final_text="你好世界")
        self.config = SimpleNamespace(
            model_type="zipformer",
            provider="cpu",
            num_threads=2,
            decoding_method="greedy_search",
            sample_rate=16_000,
        )


class _ConfiguredFinalizer(ScriptedFinalizer):
    def __init__(self) -> None:
        super().__init__(text="你好世界")
        self.device = "cpu"
        self.compute_type = "int8"
        self.cpu_threads = 4
        self.beam_size = 5
        self.language = "zh"


def _configured_factories(*args: object, **kwargs: object) -> BackendFactories:
    return BackendFactories(
        recognizer_factory=_ConfiguredRecognizer,
        finalizer_factory=_ConfiguredFinalizer,
        name="sherpa-onnx+faster-whisper",
        evidence={},
    )


def _real_run_kwargs(tmp_path: Path) -> dict[str, object]:
    _write_wav(tmp_path / "warmup.wav", frames=1600)
    return {
        "backend": "sherpa-onnx",
        "model_dir": tmp_path,
        "verifier_model": tmp_path,
        "streaming_revision": "2025-06-30",
        "verifier_revision": "Systran/faster-whisper-small@536b0662",
        "streaming_source_url": "https://example.invalid/models/streaming-v1",
        "streaming_license": "Reviewed-Test-License",
        "streaming_license_reviewed": True,
        "verifier_source_url": "https://example.invalid/models/verifier-v1",
        "verifier_license": "MIT",
        "verifier_license_reviewed": True,
        "warmup_audio": tmp_path / "warmup.wav",
    }


def test_fake_runner_records_complete_timing_contract_and_warmup(tmp_path: Path) -> None:
    audio_hash = _write_wav(tmp_path / "utterance-1.wav")
    manifest_path = tmp_path / "prepared.json"
    output_path = tmp_path / "result.json"
    _write_manifest(manifest_path, audio_hash)

    result = run_manifest(
        manifest_path,
        output_path,
        audio_root=tmp_path,
        backend="fake",
        warmup_audio=tmp_path / "utterance-1.wav",
    )

    assert result["protocol_id"] == "unit-protocol-v1"
    assert result["frozen"] is False
    assert result["model"]["streaming_config"] is None
    assert result["runner"]["warmup"]["audio_sha256"] == audio_hash
    definitions = result["runner"]["timing_definitions"]
    assert set(definitions) == {
        "first_partial_audio_ms",
        "first_partial_wall_ms",
        "stream_compute_ms",
        "stream_finalize_ms",
        "verifier_compute_ms",
        "endpoint_to_final_ms",
        "total_compute_ms",
        "utterance_rtf",
        "aggregate_utterance_rtf",
    }
    row = result["rows"][0]
    assert row["endpoint_to_final_ms"] >= row["stream_finalize_ms"]
    assert row["total_compute_ms"] >= row["stream_compute_ms"]


def test_real_runner_hashes_models_before_factory_and_records_actual_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_hash = _write_wav(tmp_path / "utterance-1.wav")
    manifest_path = tmp_path / "prepared.json"
    output_path = tmp_path / "result.json"
    _write_manifest(manifest_path, audio_hash)
    events: list[str] = []
    artifact = {"label": "encoder", "name": "encoder.onnx", "bytes": 10, "sha256": "a" * 64}

    def artifacts(*args: object, **kwargs: object) -> list[dict[str, object]]:
        events.append("artifacts")
        return [artifact]

    def factories(*args: object, **kwargs: object) -> BackendFactories:
        events.append("factory")
        return _configured_factories()

    monkeypatch.setattr(runner_module, "_model_artifacts", artifacts)
    monkeypatch.setattr(runner_module, "build_backend_factories", factories)

    result = run_manifest(
        manifest_path,
        output_path,
        audio_root=tmp_path,
        **_real_run_kwargs(tmp_path),  # type: ignore[arg-type]
    )

    assert events == ["artifacts", "factory", "artifacts"]
    assert result["frozen"] is True
    assert result["model"]["streaming_config"] == {
        "model_type": "zipformer",
        "provider": "cpu",
        "num_threads": 2,
        "decoding_method": "greedy_search",
        "sample_rate": 16_000,
    }
    assert result["model"]["verifier_config"] == {
        "device": "cpu",
        "compute_type": "int8",
        "cpu_threads": 4,
        "beam_size": 5,
        "language": "zh",
    }
    assert result["model"]["verifier_revision"] == ("Systran/faster-whisper-small@536b0662")


def test_real_runner_rejects_model_mutation_before_writing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_hash = _write_wav(tmp_path / "utterance-1.wav")
    manifest_path = tmp_path / "prepared.json"
    output_path = tmp_path / "result.json"
    _write_manifest(manifest_path, audio_hash)
    calls = 0

    def artifacts(*args: object, **kwargs: object) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        return [
            {
                "label": "encoder",
                "name": "encoder.onnx",
                "bytes": 10,
                "sha256": ("a" if calls == 1 else "b") * 64,
            }
        ]

    monkeypatch.setattr(runner_module, "_model_artifacts", artifacts)
    monkeypatch.setattr(runner_module, "build_backend_factories", _configured_factories)

    with pytest.raises(ManifestRunnerError, match="model artifacts changed"):
        run_manifest(
            manifest_path,
            output_path,
            audio_root=tmp_path,
            **_real_run_kwargs(tmp_path),  # type: ignore[arg-type]
        )
    assert not output_path.exists()


def test_runner_rejects_noncanonical_audio_relpath_before_inference(tmp_path: Path) -> None:
    audio_hash = _write_wav(tmp_path / "utterance-1.wav")
    manifest_path = tmp_path / "prepared.json"
    _write_manifest(manifest_path, audio_hash)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"][0]["audio_relpath"] = "sub/../utterance-1.wav"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestRunnerError, match="audio_relpath"):
        run_manifest(
            manifest_path,
            tmp_path / "result.json",
            audio_root=tmp_path,
            backend="fake",
        )


@pytest.mark.parametrize("truncated_bytes", [1, 2, 3])
def test_runner_rejects_truncated_wav_data(tmp_path: Path, truncated_bytes: int) -> None:
    wav_path = tmp_path / "utterance-1.wav"
    _write_wav(wav_path, frames=2)
    wav_path.write_bytes(wav_path.read_bytes()[:-truncated_bytes])
    audio_hash = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "prepared.json"
    _write_manifest(manifest_path, audio_hash)

    with pytest.raises(ManifestRunnerError, match="truncated"):
        run_manifest(
            manifest_path,
            tmp_path / "result.json",
            audio_root=tmp_path,
            backend="fake",
        )


def test_runner_rejects_ambiguous_or_non_utf8_json(tmp_path: Path) -> None:
    audio_hash = _write_wav(tmp_path / "utterance-1.wav")
    manifest_path = tmp_path / "prepared.json"
    _write_manifest(manifest_path, audio_hash)
    duplicate = manifest_path.read_text(encoding="utf-8").replace(
        '"frozen": false,',
        '"frozen": false, "frozen": true,',
    )
    manifest_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ManifestRunnerError, match="duplicate JSON object key"):
        run_manifest(
            manifest_path,
            tmp_path / "duplicate-result.json",
            audio_root=tmp_path,
            backend="fake",
        )

    manifest_path.write_bytes(b"\xff")
    with pytest.raises(ManifestRunnerError, match="not valid UTF-8"):
        run_manifest(
            manifest_path,
            tmp_path / "utf8-result.json",
            audio_root=tmp_path,
            backend="fake",
        )


@pytest.mark.parametrize("revision", [None, "latest", "deterministic-fixture", "bad revision"])
def test_real_runner_rejects_unpinned_streaming_revision(
    tmp_path: Path, revision: str | None
) -> None:
    audio_hash = _write_wav(tmp_path / "utterance-1.wav")
    manifest_path = tmp_path / "prepared.json"
    _write_manifest(manifest_path, audio_hash)

    with pytest.raises(ManifestRunnerError, match="streaming_revision"):
        run_manifest(
            manifest_path,
            tmp_path / "result.json",
            audio_root=tmp_path,
            backend="sherpa-onnx",
            streaming_revision=revision,
            verifier_revision="536b0662742c02347bc0e980a01041f333bce120",
        )
