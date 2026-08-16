"""Recompute CER from a frozen row manifest, or fail closed."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from echoforge.evaluation.cer import edit_counts
from echoforge.evaluation.evidence_contract import (
    canonical_identifier,
    valid_audio_relpath,
    valid_revision,
    valid_sha256,
    validate_dataset_provenance,
    validate_dataset_selection,
)
from echoforge.evaluation.evidence_io import (
    EvidenceJsonError,
    normalized_path,
    strict_json_dumps,
    strict_json_loads,
    write_json_new,
)
from echoforge.evaluation.normalize_zh import NORMALIZER_VERSION

MANIFEST_SCHEMA = "echoforge.eval-manifest/v1"
RUNNER_SCHEMA = "echoforge.runner/v1"
REPORT_SCHEMA = "echoforge.report/v1"
ALLOWED_BACKENDS = {"sherpa-onnx", "sherpa-onnx+faster-whisper"}
ALLOWED_SPLITS = {"dev", "test"}
REQUIRED_PACKAGES = {
    "echoforge-asr",
    "numpy",
    "sherpa-onnx",
    "faster-whisper",
    "ctranslate2",
    "onnxruntime",
}
REQUIRED_TIMING_DEFINITIONS = {
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
BASE_ARTIFACT_LABELS = {"tokens", "encoder", "decoder"}
VERIFIER_ARTIFACT_LABELS = {
    "verifier_model.bin",
    "verifier_config.json",
    "verifier_tokenizer.json",
    "verifier_vocabulary",
}
OPTIONAL_VERIFIER_ARTIFACT_LABELS = {"verifier_preprocessor_config.json"}
MINIMUM_ARTIFACT_BYTES = {
    "tokens": 128,
    "encoder": 100_000,
    "decoder": 10_000,
    "joiner": 10_000,
    "verifier_model.bin": 1_000_000,
    "verifier_config.json": 32,
    "verifier_tokenizer.json": 1_000,
    "verifier_vocabulary": 1_000,
    "verifier_preprocessor_config.json": 16,
}


def _valid_hash(value: object) -> bool:
    return valid_sha256(value)


def _canonical_identifier(value: object) -> bool:
    return canonical_identifier(value)


def _valid_revision(value: object) -> bool:
    return valid_revision(value)


def _valid_number(value: object, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and (number > 0 if positive else number >= 0)


def _valid_audio_relpath(value: object) -> bool:
    return valid_audio_relpath(value)


def _valid_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or value != value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _not_evaluated(reasons: list[str], *, rows: int) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "not_yet_evaluated",
        "reasons": sorted(set(reasons)),
        "rows": rows,
    }


def _validate_dataset(payload: dict[str, Any], reasons: list[str]) -> dict[str, Any] | None:
    dataset_reasons, selection = validate_dataset_provenance(payload.get("dataset"))
    reasons.extend(dataset_reasons)
    return selection


def _validate_dataset_selection(
    selection: dict[str, Any] | None,
    reasons: list[str],
    *,
    splits: set[str],
    row_count: int,
    speakers: set[str],
) -> None:
    reasons.extend(
        validate_dataset_selection(
            selection,
            splits=splits,
            row_count=row_count,
            speakers=speakers,
        )
    )


def _validate_provenance_entry(
    value: object,
    reasons: list[str],
    *,
    label: str,
    expected_revision: object,
) -> None:
    if not isinstance(value, dict):
        reasons.append(f"model has no reviewed {label} provenance")
        return
    source_url = value.get("source_url")
    if (
        not isinstance(source_url, str)
        or source_url != source_url.strip()
        or not source_url.startswith("https://")
    ):
        reasons.append(f"model {label} provenance has an invalid source_url")
    license_name = value.get("license")
    if (
        not isinstance(license_name, str)
        or license_name != license_name.strip()
        or not license_name
        or license_name.casefold() in {"unknown", "none", "n/a"}
    ):
        reasons.append(f"model {label} provenance has an invalid license")
    if value.get("license_reviewed") is not True:
        reasons.append(f"model {label} license_reviewed is not true")
    if value.get("revision") != expected_revision:
        reasons.append(f"model {label} provenance revision is inconsistent")


def _validate_model(payload: dict[str, Any], reasons: list[str]) -> str | None:
    model = payload.get("model")
    if not isinstance(model, dict):
        reasons.append("manifest has no model evidence")
        return None

    backend = model.get("backend")
    if backend not in ALLOWED_BACKENDS:
        reasons.append("manifest model backend is not an allowed real ASR backend")
        return None
    assert isinstance(backend, str)

    if not _valid_revision(model.get("streaming_revision")):
        reasons.append("model has an invalid or fixture streaming_revision")
    dual_pass = backend == "sherpa-onnx+faster-whisper"
    verifier_revision = model.get("verifier_revision")
    if dual_pass:
        if not _valid_revision(verifier_revision):
            reasons.append("dual-pass model has an invalid or fixture verifier_revision")
    elif verifier_revision is not None:
        reasons.append("streaming-only model must not declare a verifier_revision")

    provenance = model.get("provenance")
    if not isinstance(provenance, dict):
        reasons.append("model has no structured provenance")
    else:
        _validate_provenance_entry(
            provenance.get("streaming"),
            reasons,
            label="streaming",
            expected_revision=model.get("streaming_revision"),
        )
        if dual_pass:
            _validate_provenance_entry(
                provenance.get("verifier"),
                reasons,
                label="verifier",
                expected_revision=verifier_revision,
            )
        elif provenance.get("verifier") is not None:
            reasons.append("streaming-only model must not declare verifier provenance")

    streaming_config = model.get("streaming_config")
    model_type: object = None
    if not isinstance(streaming_config, dict):
        reasons.append("model has no streaming decoder configuration")
    else:
        model_type = streaming_config.get("model_type")
        if model_type not in {"zipformer", "paraformer", "transducer"}:
            reasons.append("streaming decoder has an invalid model_type")
        if streaming_config.get("provider") != "cpu":
            reasons.append("authorized streaming decoder provider must be cpu")
        threads = streaming_config.get("num_threads")
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 64:
            reasons.append("streaming decoder has an invalid num_threads")
        if streaming_config.get("decoding_method") not in {
            "greedy_search",
            "modified_beam_search",
            "fast_beam_search",
        }:
            reasons.append("streaming decoder has an invalid decoding_method")
        if streaming_config.get("sample_rate") != 16_000:
            reasons.append("streaming decoder sample_rate must be 16000")

    verifier_config = model.get("verifier_config")
    if dual_pass:
        if not isinstance(verifier_config, dict):
            reasons.append("dual-pass model has no verifier decoder configuration")
        else:
            if verifier_config.get("device") not in {"cpu", "cuda"}:
                reasons.append("verifier decoder has an invalid device")
            compute_type = verifier_config.get("compute_type")
            if (
                not isinstance(compute_type, str)
                or compute_type != compute_type.strip()
                or not compute_type
            ):
                reasons.append("verifier decoder has an invalid compute_type")
            cpu_threads = verifier_config.get("cpu_threads")
            if (
                isinstance(cpu_threads, bool)
                or not isinstance(cpu_threads, int)
                or not 1 <= cpu_threads <= 64
            ):
                reasons.append("verifier decoder has an invalid cpu_threads")
            beam_size = verifier_config.get("beam_size")
            if (
                isinstance(beam_size, bool)
                or not isinstance(beam_size, int)
                or not 1 <= beam_size <= 20
            ):
                reasons.append("verifier decoder has an invalid beam_size")
            if verifier_config.get("language") != "zh":
                reasons.append("verifier decoder language must be zh")
    elif verifier_config is not None:
        reasons.append("streaming-only model must not declare verifier_config")

    artifacts_value = model.get("artifacts")
    artifacts = artifacts_value if isinstance(artifacts_value, list) else []
    if not artifacts:
        reasons.append("manifest has no model artifact evidence")
    labels: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            reasons.append(f"model artifact {index} is not an object")
            continue
        label = artifact.get("label")
        if not isinstance(label, str) or not label or label != label.strip():
            reasons.append(f"model artifact {index} has an invalid label")
        elif label in labels:
            reasons.append(f"model artifact {index} has a duplicate label")
        else:
            labels.add(label)
        name = artifact.get("name")
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or PurePosixPath(name).name != name
            or PureWindowsPath(name).name != name
        ):
            reasons.append(f"model artifact {index} has an invalid name")
        if not _valid_hash(artifact.get("sha256")):
            reasons.append(f"model artifact {index} has an invalid sha256")
        size = artifact.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            reasons.append(f"model artifact {index} has an invalid byte count")
        elif isinstance(label, str):
            minimum = MINIMUM_ARTIFACT_BYTES.get(label)
            if minimum is not None and size < minimum:
                reasons.append(f"model artifact {index} is implausibly small for {label}")

    required_labels = set(BASE_ARTIFACT_LABELS)
    if model_type != "paraformer":
        required_labels.add("joiner")
    if dual_pass:
        required_labels.update(VERIFIER_ARTIFACT_LABELS)
    missing_labels = sorted(required_labels - labels)
    if missing_labels:
        reasons.append("model artifacts are missing required labels: " + ", ".join(missing_labels))
    if not dual_pass and labels.intersection(
        VERIFIER_ARTIFACT_LABELS | OPTIONAL_VERIFIER_ARTIFACT_LABELS
    ):
        reasons.append("streaming-only model contains verifier artifact labels")

    return backend


def _validate_runner(
    payload: dict[str, Any], reasons: list[str], *, backend: str | None
) -> str | None:
    runner = payload.get("runner")
    if not isinstance(runner, dict) or runner.get("schema_version") != RUNNER_SCHEMA:
        reasons.append("manifest has no supported runner evidence")
        return None
    if not _valid_hash(runner.get("source_manifest_sha256")):
        reasons.append("runner has an invalid source_manifest_sha256")
    if runner.get("authorization_requested") is not True:
        reasons.append("runner did not record authorized evaluation input")
    if runner.get("authorization_effective") is not True:
        reasons.append("runner authorization_effective is not true")
    if not _valid_utc_timestamp(runner.get("generated_at_utc")):
        reasons.append("runner has an invalid generated_at_utc")
    chunk_ms = runner.get("chunk_ms")
    if isinstance(chunk_ms, bool) or not isinstance(chunk_ms, int) or not 20 <= chunk_ms <= 1000:
        reasons.append("runner has an invalid chunk_ms")
    if runner.get("clock") != (
        "time.perf_counter wall time; offline unpaced sequential single process"
    ):
        reasons.append("runner has an unsupported latency clock")
    if runner.get("rtf_basis") != "sequential ASR wall time / decoded utterance duration":
        reasons.append("runner has an unsupported RTF basis")

    definitions = runner.get("timing_definitions")
    if not isinstance(definitions, dict):
        reasons.append("runner has no timing definitions")
    else:
        missing = sorted(REQUIRED_TIMING_DEFINITIONS - definitions.keys())
        if missing:
            reasons.append("runner timing definitions are missing: " + ", ".join(missing))
        for name in REQUIRED_TIMING_DEFINITIONS.intersection(definitions):
            description = definitions[name]
            if (
                not isinstance(description, str)
                or description != description.strip()
                or not description
            ):
                reasons.append(f"runner timing definition {name} is invalid")

    packages = runner.get("packages")
    if not isinstance(packages, dict):
        reasons.append("runner has no package version evidence")
    else:
        missing = sorted(REQUIRED_PACKAGES - packages.keys())
        if missing:
            reasons.append("runner package evidence is missing: " + ", ".join(missing))
        relevant = {"echoforge-asr", "numpy", "sherpa-onnx"}
        if backend == "sherpa-onnx+faster-whisper":
            relevant.update({"faster-whisper", "ctranslate2"})
        for name in relevant:
            version = packages.get(name)
            if not isinstance(version, str) or version != version.strip() or not version:
                reasons.append(f"runner package {name} has no version")
        for name in REQUIRED_PACKAGES.intersection(packages):
            version = packages[name]
            if version is not None and (
                not isinstance(version, str) or version != version.strip() or not version
            ):
                reasons.append(f"runner package {name} has an invalid version")

    device = runner.get("device")
    if not isinstance(device, dict):
        reasons.append("runner has no device evidence")
    else:
        for name in ("platform", "python"):
            value = device.get(name)
            if not isinstance(value, str) or value != value.strip() or not value:
                reasons.append(f"runner device {name} is invalid")
        if not isinstance(device.get("processor"), str):
            reasons.append("runner device processor is invalid")
        logical_cpus = device.get("logical_cpus")
        if isinstance(logical_cpus, bool) or not isinstance(logical_cpus, int) or logical_cpus <= 0:
            reasons.append("runner device logical_cpus is invalid")

    warmup = runner.get("warmup")
    if not isinstance(warmup, dict) or not isinstance(warmup.get("performed"), bool):
        reasons.append("runner has no explicit warmup evidence")
        return None
    if warmup["performed"] is not True:
        reasons.append("authorized runner must perform a separate warmup")
        return None
    warmup_hash = warmup.get("audio_sha256")
    if not _valid_hash(warmup_hash):
        reasons.append("runner warmup has an invalid audio_sha256")
    if not _valid_number(warmup.get("audio_duration_s"), positive=True):
        reasons.append("runner warmup has an invalid audio_duration_s")
    if not _valid_number(warmup.get("total_compute_ms")):
        reasons.append("runner warmup has an invalid total_compute_ms")
    return str(warmup_hash).lower() if _valid_hash(warmup_hash) else None


def _validate_row_timing(
    row: dict[str, Any], index: int, reasons: list[str], *, dual_pass: bool
) -> tuple[float, float] | None:
    partial_updates = row.get("partial_updates")
    if (
        isinstance(partial_updates, bool)
        or not isinstance(partial_updates, int)
        or partial_updates < 0
    ):
        reasons.append(f"row {index} has an invalid partial_updates")
        partial_updates = None
    first_audio = row.get("first_partial_audio_ms")
    first_wall = row.get("first_partial_wall_ms")
    if partial_updates == 0:
        if first_audio is not None or first_wall is not None:
            reasons.append(f"row {index} has partial timing without a partial update")
    elif (
        isinstance(partial_updates, int)
        and partial_updates > 0
        and (not _valid_number(first_audio) or not _valid_number(first_wall))
    ):
        reasons.append(f"row {index} is missing first-partial timing")

    required_nonnegative = (
        "stream_compute_ms",
        "stream_finalize_ms",
        "endpoint_to_final_ms",
        "total_compute_ms",
    )
    for field in required_nonnegative:
        if not _valid_number(row.get(field)):
            reasons.append(f"row {index} has an invalid {field}")
    verifier_compute = row.get("verifier_compute_ms")
    if dual_pass:
        if not _valid_number(verifier_compute):
            reasons.append(f"row {index} has an invalid verifier_compute_ms")
    elif verifier_compute is not None:
        reasons.append(f"row {index} streaming-only result has verifier_compute_ms")

    duration = row.get("audio_duration_s")
    rtf = row.get("utterance_rtf")
    if not _valid_number(duration, positive=True):
        reasons.append(f"row {index} has an invalid audio_duration_s")
    if not _valid_number(rtf):
        reasons.append(f"row {index} has an invalid utterance_rtf")

    timing_values = [row.get(field) for field in required_nonnegative]
    if all(_valid_number(value) for value in timing_values):
        stream_compute, stream_finalize, endpoint_to_final, total_compute = map(
            float, timing_values
        )
        if stream_finalize > stream_compute:
            reasons.append(f"row {index} stream_finalize_ms exceeds stream_compute_ms")
        if endpoint_to_final > total_compute:
            reasons.append(f"row {index} endpoint_to_final_ms exceeds total_compute_ms")
        if total_compute < stream_compute:
            reasons.append(f"row {index} total_compute_ms is below stream_compute_ms")
        if dual_pass:
            if endpoint_to_final < stream_finalize:
                reasons.append(f"row {index} endpoint_to_final_ms is below stream_finalize_ms")
            if _valid_number(verifier_compute):
                verifier_value = float(verifier_compute)
                if verifier_value > total_compute:
                    reasons.append(f"row {index} verifier_compute_ms exceeds total_compute_ms")
                if verifier_value > endpoint_to_final:
                    reasons.append(f"row {index} verifier_compute_ms exceeds endpoint_to_final_ms")
                if stream_compute + verifier_value > total_compute + 0.002:
                    reasons.append(
                        f"row {index} total_compute_ms is below stream plus verifier compute"
                    )
                if stream_finalize + verifier_value > endpoint_to_final + 0.002:
                    reasons.append(
                        f"row {index} endpoint_to_final_ms is below finalize plus verifier compute"
                    )
        elif not math.isclose(endpoint_to_final, stream_finalize, abs_tol=0.001):
            reasons.append(
                f"row {index} streaming-only endpoint_to_final_ms differs from stream_finalize_ms"
            )

    if _valid_number(duration, positive=True):
        duration_ms = float(duration) * 1000
        if _valid_number(first_audio) and float(first_audio) > duration_ms + 0.001:
            reasons.append(f"row {index} first_partial_audio_ms exceeds audio duration")
    if (
        _valid_number(first_wall)
        and _valid_number(row.get("stream_compute_ms"))
        and float(first_wall) > float(row["stream_compute_ms"]) + 0.001
    ):
        reasons.append(f"row {index} first_partial_wall_ms exceeds stream_compute_ms")

    if (
        _valid_number(duration, positive=True)
        and _valid_number(row.get("total_compute_ms"))
        and _valid_number(rtf)
    ):
        expected_rtf = round((float(row["total_compute_ms"]) / 1000) / float(duration), 6)
        if not math.isclose(float(rtf), expected_rtf, abs_tol=0.000001):
            reasons.append(f"row {index} utterance_rtf is inconsistent with timing")
        return float(duration), float(row["total_compute_ms"])
    return None


def _validate_runtime_summary(
    payload: dict[str, Any], reasons: list[str], *, rows: int, totals: list[tuple[float, float]]
) -> None:
    summary = payload.get("runtime_summary")
    if not isinstance(summary, dict):
        reasons.append("manifest has no runtime summary")
        return
    summary_rows = summary.get("rows")
    if isinstance(summary_rows, bool) or not isinstance(summary_rows, int) or summary_rows != rows:
        reasons.append("runtime summary row count is inconsistent")
    audio_s = summary.get("total_audio_s")
    compute_ms = summary.get("total_compute_ms")
    aggregate_rtf = summary.get("aggregate_utterance_rtf")
    if not _valid_number(audio_s, positive=True):
        reasons.append("runtime summary has an invalid total_audio_s")
    if not _valid_number(compute_ms):
        reasons.append("runtime summary has an invalid total_compute_ms")
    if not _valid_number(aggregate_rtf):
        reasons.append("runtime summary has an invalid aggregate_utterance_rtf")
    if len(totals) != rows:
        return
    expected_audio = round(sum(item[0] for item in totals), 6)
    expected_compute = round(sum(item[1] for item in totals), 3)
    if _valid_number(audio_s, positive=True) and not math.isclose(
        float(audio_s), expected_audio, abs_tol=0.000001
    ):
        reasons.append("runtime summary total_audio_s is inconsistent with rows")
    if _valid_number(compute_ms) and not math.isclose(
        float(compute_ms), expected_compute, abs_tol=0.001
    ):
        reasons.append("runtime summary total_compute_ms is inconsistent with rows")
    if _valid_number(aggregate_rtf):
        expected_rtf = round((expected_compute / 1000) / expected_audio, 6)
        if not math.isclose(float(aggregate_rtf), expected_rtf, abs_tol=0.000001):
            reasons.append("runtime summary aggregate_utterance_rtf is inconsistent with rows")


def _sanitized_report_evidence(payload: dict[str, Any], *, speaker_count: int) -> dict[str, Any]:
    dataset = payload["dataset"]
    model = payload["model"]
    runner = payload["runner"]
    selection = dataset["selection"]
    return {
        "dataset": {
            field: dataset[field]
            for field in (
                "name",
                "source",
                "source_url",
                "license",
                "license_page_sha256",
                "download_manifest_sha256",
                "extraction_marker_sha256",
                "extraction_inventory_sha256",
                "transcript_sha256",
                "speaker_policy",
                "audio_protocol",
                "raw_audio_in_repository",
            )
        }
        | {
            "selection": {
                "splits": selection["splits"],
                "rows": selection["rows"],
                "speakers": speaker_count,
                "speaker_limit_per_split": selection["speaker_limit_per_split"],
                "utterances_per_speaker": selection["utterances_per_speaker"],
                "extraction_speaker_limit_per_split": selection[
                    "extraction_speaker_limit_per_split"
                ],
            }
        },
        "model": {
            field: model[field]
            for field in (
                "backend",
                "streaming_revision",
                "verifier_revision",
                "streaming_config",
                "verifier_config",
                "provenance",
                "artifacts",
            )
        },
        "runner": {
            field: runner[field]
            for field in (
                "schema_version",
                "generated_at_utc",
                "source_manifest_sha256",
                "authorization_requested",
                "authorization_effective",
                "chunk_ms",
                "clock",
                "rtf_basis",
                "timing_definitions",
                "warmup",
                "packages",
                "device",
            )
        },
        "runtime_summary": payload["runtime_summary"],
    }


def evaluate(manifest_path: Path) -> dict[str, Any]:
    manifest_path = normalized_path(manifest_path)
    source_bytes = manifest_path.read_bytes()
    try:
        decoded = strict_json_loads(source_bytes)
    except EvidenceJsonError as exc:
        return _not_evaluated([f"manifest is not valid strict JSON: {exc}"], rows=0)
    if not isinstance(decoded, dict):
        return _not_evaluated(["manifest root is not an object"], rows=0)
    payload: dict[str, Any] = decoded
    reasons: list[str] = []
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        reasons.append("unsupported manifest schema")
    if payload.get("evaluation_authorized") is not True:
        reasons.append("manifest is not authorized for quality evaluation")
    if payload.get("frozen") is not True:
        reasons.append("manifest is not marked frozen")
    if payload.get("normalization") != NORMALIZER_VERSION:
        reasons.append(f"normalization must equal {NORMALIZER_VERSION}")
    if not _canonical_identifier(payload.get("protocol_id")):
        reasons.append("manifest has no canonical protocol_id")

    dataset_selection = _validate_dataset(payload, reasons)
    backend = _validate_model(payload, reasons)
    warmup_hash = _validate_runner(payload, reasons, backend=backend)

    rows_value = payload.get("rows")
    if not isinstance(rows_value, list) or not rows_value:
        reasons.append("manifest has no evaluation rows")
        rows: list[object] = []
    else:
        rows = rows_value
    seen_ids: set[str] = set()
    seen_audio_hashes: set[str] = set()
    seen_audio_paths: set[str] = set()
    speakers: set[str] = set()
    splits: set[str] = set()
    totals: list[tuple[float, float]] = []
    counts = {
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "reference_units": 0,
        "hypothesis_units": 0,
    }
    dual_pass = backend == "sherpa-onnx+faster-whisper"
    for index, value in enumerate(rows):
        if not isinstance(value, dict):
            reasons.append(f"row {index} is not an object")
            continue
        row: dict[str, Any] = value
        row_id = row.get("id")
        if not _canonical_identifier(row_id):
            reasons.append(f"row {index} has a non-canonical id")
        elif row_id in seen_ids:
            reasons.append(f"row {index} has a duplicate id")
        else:
            seen_ids.add(row_id)
        speaker_id = row.get("speaker_id")
        if not _canonical_identifier(speaker_id):
            reasons.append(f"row {index} has a non-canonical speaker_id")
        else:
            assert isinstance(speaker_id, str)
            speakers.add(speaker_id)
        split = row.get("split")
        if split not in ALLOWED_SPLITS:
            reasons.append(f"row {index} has an invalid evaluation split")
        else:
            assert isinstance(split, str)
            splits.add(split)

        audio_hash = row.get("audio_sha256")
        if not _valid_hash(audio_hash):
            reasons.append(f"row {index} has an invalid audio_sha256")
        else:
            normalized_hash = str(audio_hash).lower()
            if normalized_hash in seen_audio_hashes:
                reasons.append(f"row {index} reuses an audio_sha256")
            else:
                seen_audio_hashes.add(normalized_hash)
        audio_relpath = row.get("audio_relpath")
        if not _valid_audio_relpath(audio_relpath):
            reasons.append(f"row {index} has an invalid audio_relpath")
        else:
            normalized_audio_path = str(audio_relpath).casefold()
            if normalized_audio_path in seen_audio_paths:
                reasons.append(f"row {index} reuses an audio_relpath")
            else:
                seen_audio_paths.add(normalized_audio_path)

        reference = row.get("reference")
        hypothesis = row.get("hypothesis")
        stream_final = row.get("stream_final")
        verified_final = row.get("verified_final")
        if not isinstance(reference, str) or not isinstance(hypothesis, str):
            reasons.append(f"row {index} is missing reference/hypothesis")
        else:
            try:
                edit = edit_counts(reference, hypothesis)
            except (TypeError, ValueError) as exc:
                reasons.append(f"row {index} has invalid text normalization: {exc}")
            else:
                for key in counts:
                    counts[key] += getattr(edit, key)
        if not isinstance(stream_final, str):
            reasons.append(f"row {index} is missing stream_final")
        if dual_pass:
            if row.get("final_stage") != "dual_pass_final":
                reasons.append(f"row {index} final_stage does not match dual-pass backend")
            if not isinstance(verified_final, str):
                reasons.append(f"row {index} is missing verified_final")
            elif hypothesis != verified_final:
                reasons.append(f"row {index} hypothesis does not match verified_final")
        elif backend == "sherpa-onnx":
            if row.get("final_stage") != "stream_final":
                reasons.append(f"row {index} final_stage does not match streaming backend")
            if verified_final is not None:
                reasons.append(f"row {index} streaming-only result contains verified_final")
            if isinstance(stream_final, str) and hypothesis != stream_final:
                reasons.append(f"row {index} hypothesis does not match stream_final")

        timing = _validate_row_timing(row, index, reasons, dual_pass=dual_pass)
        if timing is not None:
            totals.append(timing)

    if len(splits) != 1:
        reasons.append("manifest must contain exactly one dev or test split")
    if warmup_hash is not None and warmup_hash in seen_audio_hashes:
        reasons.append("runner warmup reuses an evaluation audio hash")
    _validate_dataset_selection(
        dataset_selection,
        reasons,
        splits=splits,
        row_count=len(rows),
        speakers=speakers,
    )
    if counts["reference_units"] == 0:
        reasons.append("normalized references contain no evaluation units")
    _validate_runtime_summary(payload, reasons, rows=len(rows), totals=totals)

    if reasons:
        return _not_evaluated(reasons, rows=len(rows))
    cer = (counts["substitutions"] + counts["deletions"] + counts["insertions"]) / counts[
        "reference_units"
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "evaluated",
        "rows": len(rows),
        "split": next(iter(splits)),
        "protocol_id": payload["protocol_id"],
        "counts": counts,
        "cer": cer,
        "manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "normalization": NORMALIZER_VERSION,
        "evidence": _sanitized_report_evidence(payload, speaker_count=len(speakers)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = evaluate(args.manifest)
        encoded = strict_json_dumps(report)
    except (EvidenceJsonError, OSError) as exc:
        print(f"unable to read evaluation evidence: {exc}", file=sys.stderr)
        return 1
    if args.output:
        source = normalized_path(args.manifest)
        destination = normalized_path(args.output)
        if source == destination:
            print("report output must differ from the source manifest", file=sys.stderr)
            return 1
        try:
            write_json_new(destination, report)
        except (EvidenceJsonError, FileExistsError, OSError) as exc:
            print(f"unable to publish evaluation report: {exc}", file=sys.stderr)
            return 1
    print(encoded, end="")
    return 0 if report["status"] == "evaluated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
