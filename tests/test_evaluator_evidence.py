from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from echoforge.evaluation.normalize_zh import NORMALIZER_VERSION
from scripts.evaluate_manifest import REQUIRED_TIMING_DEFINITIONS, evaluate


def _artifact(label: str, name: str, marker: str, size: int) -> dict[str, object]:
    return {"label": label, "name": name, "bytes": size, "sha256": marker * 64}


def _valid_manifest(*, dual_pass: bool = True) -> dict[str, Any]:
    artifacts = [
        _artifact("tokens", "tokens.txt", "a", 625_503),
        _artifact("encoder", "encoder.int8.onnx", "b", 80_423_711),
        _artifact("decoder", "decoder.int8.onnx", "c", 4_304_921),
        _artifact("joiner", "joiner.int8.onnx", "d", 5_206_733),
    ]
    if dual_pass:
        artifacts.extend(
            [
                _artifact("verifier_model.bin", "model.bin", "e", 461_307_911),
                _artifact("verifier_config.json", "config.json", "f", 2_183),
                _artifact("verifier_tokenizer.json", "tokenizer.json", "1", 2_481_944),
                _artifact("verifier_vocabulary", "vocabulary.json", "2", 912_884),
                _artifact(
                    "verifier_preprocessor_config.json",
                    "preprocessor_config.json",
                    "3",
                    339,
                ),
            ]
        )
    backend = "sherpa-onnx+faster-whisper" if dual_pass else "sherpa-onnx"
    stream_final = "泥号" if dual_pass else "你号"
    return {
        "schema_version": "echoforge.eval-manifest/v1",
        "dataset": {
            "name": "AISHELL-1",
            "source": "OpenSLR 33",
            "source_url": "https://www.openslr.org/33/",
            "license": "Apache-2.0",
            "license_page_sha256": "6" * 64,
            "download_manifest_sha256": "7" * 64,
            "extraction_marker_sha256": "0" * 64,
            "extraction_inventory_sha256": "a" * 64,
            "transcript_sha256": "8" * 64,
            "speaker_policy": "speaker-disjoint",
            "audio_protocol": "16 kHz mono PCM16LE",
            "raw_audio_in_repository": False,
            "selection": {
                "splits": ["dev"],
                "speaker_limit_per_split": None,
                "utterances_per_speaker": None,
                "extraction_speaker_limit_per_split": None,
                "selected_speakers": {"dev": ["S0002"]},
                "rows": 1,
            },
        },
        "protocol_id": "aishell1-dev-clean-v1",
        "evaluation_authorized": True,
        "frozen": True,
        "normalization": NORMALIZER_VERSION,
        "model": {
            "backend": backend,
            "streaming_revision": "sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
            "verifier_revision": ("Systran/faster-whisper-small@536b0662" if dual_pass else None),
            "provenance": {
                "streaming": {
                    "source_url": "https://example.invalid/models/streaming/release-v1",
                    "revision": "sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
                    "license": "Reviewed-Test-License",
                    "license_reviewed": True,
                },
                "verifier": (
                    {
                        "source_url": "https://huggingface.co/Systran/faster-whisper-small",
                        "revision": "Systran/faster-whisper-small@536b0662",
                        "license": "MIT",
                        "license_reviewed": True,
                    }
                    if dual_pass
                    else None
                ),
            },
            "streaming_config": {
                "model_type": "zipformer",
                "provider": "cpu",
                "num_threads": 2,
                "decoding_method": "greedy_search",
                "sample_rate": 16_000,
            },
            "verifier_config": (
                {
                    "device": "cpu",
                    "compute_type": "int8",
                    "cpu_threads": 4,
                    "beam_size": 5,
                    "language": "zh",
                }
                if dual_pass
                else None
            ),
            "artifacts": artifacts,
        },
        "runner": {
            "schema_version": "echoforge.runner/v1",
            "generated_at_utc": "2026-08-15T12:00:00+00:00",
            "source_manifest_sha256": "4" * 64,
            "authorization_requested": True,
            "authorization_effective": True,
            "chunk_ms": 200,
            "clock": ("time.perf_counter wall time; offline unpaced sequential single process"),
            "rtf_basis": "sequential ASR wall time / decoded utterance duration",
            "timing_definitions": {
                name: f"Frozen definition for {name}." for name in REQUIRED_TIMING_DEFINITIONS
            },
            "warmup": {
                "performed": True,
                "audio_sha256": "9" * 64,
                "audio_duration_s": 1.0,
                "total_compute_ms": 125.0,
            },
            "packages": {
                "echoforge-asr": "0.1.2",
                "numpy": "2.2.6",
                "sherpa-onnx": "1.13.5",
                "faster-whisper": "1.2.1" if dual_pass else None,
                "ctranslate2": "4.5.0" if dual_pass else None,
                "onnxruntime": None,
            },
            "device": {
                "platform": "Windows-11-10.0.26100-SP0",
                "processor": "Intel64 Family 6 Model 154",
                "python": "3.11.9",
                "logical_cpus": 16,
            },
        },
        "runtime_summary": {
            "rows": 1,
            "total_audio_s": 2.0,
            "total_compute_ms": 150.0 if dual_pass else 100.0,
            "aggregate_utterance_rtf": 0.075 if dual_pass else 0.05,
        },
        "rows": [
            {
                "id": "BAC009S0002W0122",
                "speaker_id": "S0002",
                "split": "dev",
                "audio_relpath": "wav/dev/S0002/BAC009S0002W0122.wav",
                "audio_sha256": "5" * 64,
                "reference": "你好",
                "hypothesis": "你号",
                "stream_final": stream_final,
                "verified_final": "你号" if dual_pass else None,
                "final_stage": "dual_pass_final" if dual_pass else "stream_final",
                "decoder_score": -0.42 if dual_pass else None,
                "partial_updates": 2,
                "first_partial_audio_ms": 200,
                "first_partial_wall_ms": 5.25,
                "stream_compute_ms": 100.0,
                "stream_finalize_ms": 10.0,
                "verifier_compute_ms": 50.0 if dual_pass else None,
                "endpoint_to_final_ms": 60.0 if dual_pass else 10.0,
                "total_compute_ms": 150.0 if dual_pass else 100.0,
                "audio_duration_s": 2.0,
                "utterance_rtf": 0.075 if dual_pass else 0.05,
            }
        ],
    }


def _evaluate_payload(tmp_path: Path, payload: object) -> dict[str, Any]:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return evaluate(path)


def test_structured_dual_pass_evidence_is_scored(tmp_path: Path) -> None:
    report = _evaluate_payload(tmp_path, _valid_manifest())

    assert report["status"] == "evaluated"
    assert report["split"] == "dev"
    assert report["protocol_id"] == "aishell1-dev-clean-v1"
    assert report["counts"]["substitutions"] == 1
    assert report["cer"] == pytest.approx(0.5)
    assert report["evidence"]["dataset"]["source_url"] == "https://www.openslr.org/33/"
    assert report["evidence"]["model"]["provenance"]["verifier"]["license"] == "MIT"
    assert report["evidence"]["dataset"]["selection"]["speakers"] == 1
    encoded_evidence = json.dumps(report["evidence"], ensure_ascii=False)
    assert '"reference":' not in encoded_evidence
    assert '"hypothesis":' not in encoded_evidence
    assert '"audio_relpath":' not in encoded_evidence


def test_structured_streaming_only_evidence_is_scored(tmp_path: Path) -> None:
    report = _evaluate_payload(tmp_path, _valid_manifest(dual_pass=False))

    assert report["status"] == "evaluated"
    assert report["cer"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda manifest: manifest.pop("dataset"),
            "manifest has no dataset provenance",
        ),
        (
            lambda manifest: manifest["model"].__setitem__("backend", "deterministic-fake"),
            "manifest model backend is not an allowed real ASR backend",
        ),
        (
            lambda manifest: manifest["model"].pop("provenance"),
            "model has no structured provenance",
        ),
        (
            lambda manifest: manifest["model"]["provenance"]["streaming"].__setitem__(
                "license_reviewed", False
            ),
            "model streaming license_reviewed is not true",
        ),
        (
            lambda manifest: manifest.__setitem__("normalization", "unknown-normalizer/v0"),
            f"normalization must equal {NORMALIZER_VERSION}",
        ),
        (
            lambda manifest: manifest.__setitem__("protocol_id", " non canonical "),
            "manifest has no canonical protocol_id",
        ),
        (
            lambda manifest: manifest["runner"].__setitem__("authorization_effective", False),
            "runner authorization_effective is not true",
        ),
        (
            lambda manifest: manifest["rows"][0].__setitem__("id", "bad id"),
            "row 0 has a non-canonical id",
        ),
        (
            lambda manifest: manifest["rows"][0].__setitem__("speaker_id", " speaker "),
            "row 0 has a non-canonical speaker_id",
        ),
        (
            lambda manifest: manifest["rows"][0].__setitem__("split", "smoke"),
            "row 0 has an invalid evaluation split",
        ),
    ],
)
def test_core_authorization_evidence_fails_closed(tmp_path: Path, mutation, reason: str) -> None:
    manifest = _valid_manifest()
    mutation(manifest)

    report = _evaluate_payload(tmp_path, manifest)

    assert report["status"] == "not_yet_evaluated"
    assert reason in report["reasons"]


def test_mixed_splits_and_duplicate_audio_evidence_fail_closed(tmp_path: Path) -> None:
    manifest = _valid_manifest()
    duplicate = copy.deepcopy(manifest["rows"][0])
    duplicate.update(
        {
            "id": "BAC009S0003W0123",
            "speaker_id": "S0003",
            "split": "test",
            "audio_relpath": "WAV/DEV/S0002/BAC009S0002W0122.WAV",
        }
    )
    manifest["rows"].append(duplicate)
    manifest["runtime_summary"]["rows"] = 2
    manifest["runtime_summary"]["total_audio_s"] = 4.0
    manifest["runtime_summary"]["total_compute_ms"] = 300.0

    report = _evaluate_payload(tmp_path, manifest)

    assert report["status"] == "not_yet_evaluated"
    assert "manifest must contain exactly one dev or test split" in report["reasons"]
    assert "row 1 reuses an audio_sha256" in report["reasons"]
    assert "row 1 reuses an audio_relpath" in report["reasons"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        (
            "final_stage",
            "stream_final",
            "row 0 final_stage does not match dual-pass backend",
        ),
        (
            "hypothesis",
            "不同文本",
            "row 0 hypothesis does not match verified_final",
        ),
    ],
)
def test_dual_pass_final_text_chain_must_be_consistent(
    tmp_path: Path, field: str, value: object, reason: str
) -> None:
    manifest = _valid_manifest()
    manifest["rows"][0][field] = value

    report = _evaluate_payload(tmp_path, manifest)

    assert report["status"] == "not_yet_evaluated"
    assert reason in report["reasons"]


def test_artifact_labels_and_immutable_revisions_are_required(tmp_path: Path) -> None:
    manifest = _valid_manifest()
    manifest["model"]["streaming_revision"] = "deterministic-fixture"
    manifest["model"]["artifacts"] = [
        artifact
        for artifact in manifest["model"]["artifacts"]
        if artifact["label"] != "verifier_model.bin"
    ]

    report = _evaluate_payload(tmp_path, manifest)

    assert report["status"] == "not_yet_evaluated"
    assert "model has an invalid or fixture streaming_revision" in report["reasons"]
    assert "model artifacts are missing required labels: verifier_model.bin" in report["reasons"]


def test_floating_revision_tiny_artifacts_and_impossible_timing_fail_closed(
    tmp_path: Path,
) -> None:
    manifest = _valid_manifest()
    manifest["model"]["streaming_revision"] = "main"
    for artifact in manifest["model"]["artifacts"]:
        artifact["bytes"] = 1
    manifest["rows"][0]["verifier_compute_ms"] = 1_000_000.0

    report = _evaluate_payload(tmp_path, manifest)

    assert report["status"] == "not_yet_evaluated"
    assert "model has an invalid or fixture streaming_revision" in report["reasons"]
    assert any("implausibly small" in reason for reason in report["reasons"])
    assert "row 0 verifier_compute_ms exceeds total_compute_ms" in report["reasons"]
    assert "row 0 verifier_compute_ms exceeds endpoint_to_final_ms" in report["reasons"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        (
            "speaker_limit_per_split",
            1,
            "authorized dataset selection must not limit speakers",
        ),
        (
            "utterances_per_speaker",
            1,
            "authorized dataset selection must not limit utterances",
        ),
        (
            "extraction_speaker_limit_per_split",
            1,
            "authorized dataset extraction must not limit speakers",
        ),
    ],
)
def test_authorized_selection_must_represent_a_full_split(
    tmp_path: Path, field: str, value: object, reason: str
) -> None:
    manifest = _valid_manifest()
    manifest["dataset"]["selection"][field] = value

    report = _evaluate_payload(tmp_path, manifest)

    assert report["status"] == "not_yet_evaluated"
    assert reason in report["reasons"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda manifest: manifest["runner"]["packages"].pop("sherpa-onnx"),
            "runner package evidence is missing: sherpa-onnx",
        ),
        (
            lambda manifest: manifest["runner"]["device"].pop("logical_cpus"),
            "runner device logical_cpus is invalid",
        ),
        (
            lambda manifest: manifest["runner"].__setitem__("clock", "wall clock"),
            "runner has an unsupported latency clock",
        ),
        (
            lambda manifest: manifest["runner"]["timing_definitions"].pop("endpoint_to_final_ms"),
            "runner timing definitions are missing: endpoint_to_final_ms",
        ),
        (
            lambda manifest: manifest["runner"]["warmup"].__setitem__("performed", False),
            "authorized runner must perform a separate warmup",
        ),
        (
            lambda manifest: manifest["runner"]["warmup"].__setitem__(
                "audio_sha256", manifest["rows"][0]["audio_sha256"]
            ),
            "runner warmup reuses an evaluation audio hash",
        ),
        (
            lambda manifest: manifest["rows"][0].__setitem__("utterance_rtf", 0.5),
            "row 0 utterance_rtf is inconsistent with timing",
        ),
    ],
)
def test_runtime_provenance_and_timing_are_complete(tmp_path: Path, mutation, reason: str) -> None:
    manifest = _valid_manifest()
    mutation(manifest)

    report = _evaluate_payload(tmp_path, manifest)

    assert report["status"] == "not_yet_evaluated"
    assert reason in report["reasons"]


def test_non_object_manifest_fails_closed(tmp_path: Path) -> None:
    report = _evaluate_payload(tmp_path, [])

    assert report == {
        "schema_version": "echoforge.report/v1",
        "status": "not_yet_evaluated",
        "reasons": ["manifest root is not an object"],
        "rows": 0,
    }


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ('{"frozen": false, "frozen": true}', "duplicate JSON object key"),
        ('{"value": NaN}', "non-finite JSON number"),
    ],
)
def test_ambiguous_nonstandard_json_fails_closed(tmp_path: Path, raw: str, message: str) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(raw, encoding="utf-8")

    report = evaluate(path)

    assert report["status"] == "not_yet_evaluated"
    assert message in report["reasons"][0]
