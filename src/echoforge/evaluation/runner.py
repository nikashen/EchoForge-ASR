"""Hash-verified offline ASR runner for prepared local evaluation manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import os
import platform
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from echoforge import __version__
from echoforge.asr.base import EndpointFinalizer, StreamingRecognizer
from echoforge.asr.factory import build_backend_factories
from echoforge.asr.faster_whisper import (
    LOCAL_MODEL_OPTIONAL_FILES,
    LOCAL_MODEL_REQUIRED_FILES,
    LOCAL_MODEL_VOCABULARY_FILES,
)
from echoforge.asr.sherpa_onnx import (
    DECODER_CANDIDATES,
    ENCODER_CANDIDATES,
    JOINER_CANDIDATES,
    TOKENS_CANDIDATES,
    resolve_model_file,
)
from echoforge.audio.pcm import pcm16le_to_float32

from .evidence_contract import (
    canonical_identifier,
    valid_audio_relpath,
    valid_revision,
    valid_sha256,
    validate_dataset_provenance,
    validate_dataset_selection,
)
from .evidence_io import (
    EvidenceJsonError,
    ensure_new_json_path,
    normalized_path,
    strict_json_loads,
    write_json_new,
)
from .normalize_zh import NORMALIZER_VERSION, normalize_zh

FloatAudio = NDArray[np.float32]
ALLOWED_SPLITS = {"dev", "test", "smoke"}


class ManifestRunnerError(ValueError):
    """Raised when prepared evaluation evidence is incomplete or inconsistent."""


def _hash_file(path: Path) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise ManifestRunnerError(f"model artifact must be a regular non-link file: {path}")
    path_before = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        handle_before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
        handle_after = os.fstat(handle.fileno())
    path_after = path.stat()

    def signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if (
        size != handle_before.st_size
        or signature(path_before) != signature(handle_before)
        or signature(handle_before) != signature(handle_after)
        or signature(handle_after) != signature(path_after)
    ):
        raise ManifestRunnerError(f"file changed while it was being hashed: {path}")
    return size, digest.hexdigest()


def _decode_pcm16_wav_bytes(raw: bytes, path: Path) -> FloatAudio:
    try:
        with wave.open(io.BytesIO(raw), "rb") as source:
            sample_rate = source.getframerate()
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            compression = source.getcomptype()
            declared_frames = source.getnframes()
            frames = source.readframes(declared_frames)
    except (EOFError, wave.Error) as exc:
        raise ManifestRunnerError(f"invalid WAV file: {path}") from exc
    if sample_rate != 16_000 or channels != 1 or sample_width != 2 or compression != "NONE":
        raise ManifestRunnerError(f"WAV must be uncompressed 16 kHz mono PCM16LE: {path}")
    expected_audio_bytes = declared_frames * channels * sample_width
    if len(frames) != expected_audio_bytes:
        raise ManifestRunnerError(
            f"WAV audio data is truncated: expected {expected_audio_bytes} bytes, "
            f"got {len(frames)}: {path}"
        )
    if not frames:
        raise ManifestRunnerError(f"WAV contains no audio frames: {path}")
    try:
        return pcm16le_to_float32(frames)
    except ValueError as exc:
        raise ManifestRunnerError(f"WAV contains invalid PCM16LE audio: {path}") from exc


def _load_hashed_pcm16_wav(path: Path) -> tuple[FloatAudio, str]:
    raw = path.read_bytes()
    return _decode_pcm16_wav_bytes(raw, path), hashlib.sha256(raw).hexdigest()


def _resolve_audio_path(audio_root: Path, relative_value: object) -> Path:
    if not valid_audio_relpath(relative_value):
        raise ManifestRunnerError("row audio_relpath must be a canonical POSIX relative path")
    assert isinstance(relative_value, str)
    relative = Path(relative_value)
    root = audio_root.expanduser().resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ManifestRunnerError(f"audio path escapes audio_root: {relative_value}")
    if not path.is_file():
        raise ManifestRunnerError(f"audio file does not exist: {path}")
    return path


def _validate_manifest(payload: object) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise ManifestRunnerError("manifest root must be an object")
    manifest = dict(payload)
    if manifest.get("schema_version") != "echoforge.eval-manifest/v1":
        raise ManifestRunnerError("unsupported manifest schema")
    if manifest.get("frozen") is not False:
        raise ManifestRunnerError("runner input must be explicitly marked frozen=false")
    if not isinstance(manifest.get("evaluation_authorized"), bool):
        raise ManifestRunnerError("manifest evaluation_authorized must be an explicit boolean")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or not dataset:
        raise ManifestRunnerError("manifest dataset metadata is required")
    for field in ("name", "source"):
        value = dataset.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ManifestRunnerError(f"manifest dataset {field} is required")
    if manifest.get("normalization") != NORMALIZER_VERSION:
        raise ManifestRunnerError(f"normalization must equal {NORMALIZER_VERSION}")
    rows_value = manifest.get("rows")
    if not isinstance(rows_value, list) or not rows_value:
        raise ManifestRunnerError("manifest must contain at least one row")

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_audio_hashes: set[str] = set()
    seen_audio_paths: set[str] = set()
    speaker_splits: dict[str, set[str]] = {}
    for index, value in enumerate(rows_value):
        if not isinstance(value, dict):
            raise ManifestRunnerError(f"row {index} must be an object")
        row = dict(value)
        row_id = row.get("id")
        if (
            not isinstance(row_id, str)
            or row_id != row_id.strip()
            or not canonical_identifier(row_id)
            or row_id in seen_ids
        ):
            raise ManifestRunnerError(f"row {index} has a missing or duplicate id")
        seen_ids.add(row_id)
        speaker_id = row.get("speaker_id")
        split = row.get("split")
        if (
            not isinstance(speaker_id, str)
            or speaker_id != speaker_id.strip()
            or not canonical_identifier(speaker_id)
        ):
            raise ManifestRunnerError(f"row {index} is missing speaker_id")
        if not isinstance(split, str) or split not in ALLOWED_SPLITS:
            raise ManifestRunnerError(f"row {index} is missing split")
        speaker_splits.setdefault(speaker_id, set()).add(split)
        audio_hash = row.get("audio_sha256")
        if not valid_sha256(audio_hash):
            raise ManifestRunnerError(f"row {index} has an invalid audio_sha256")
        normalized_hash = str(audio_hash).lower()
        if normalized_hash in seen_audio_hashes:
            raise ManifestRunnerError(f"row {index} reuses an audio_sha256")
        seen_audio_hashes.add(normalized_hash)
        audio_relpath = row.get("audio_relpath")
        normalized_audio_path = audio_relpath.casefold() if isinstance(audio_relpath, str) else ""
        if not valid_audio_relpath(audio_relpath) or normalized_audio_path in seen_audio_paths:
            raise ManifestRunnerError(f"row {index} has an invalid or duplicate audio_relpath")
        seen_audio_paths.add(normalized_audio_path)
        reference = row.get("reference")
        if not isinstance(reference, str) or not normalize_zh(reference):
            raise ManifestRunnerError(f"row {index} has an empty normalized reference")
        if "hypothesis" in row:
            raise ManifestRunnerError(f"row {index} already contains a hypothesis")
        rows.append(row)

    overlap = sorted(speaker for speaker, splits in speaker_splits.items() if len(splits) > 1)
    if overlap:
        raise ManifestRunnerError("speakers occur in multiple splits: " + ", ".join(overlap))
    selected_splits = {str(row["split"]) for row in rows}
    if manifest["evaluation_authorized"] is True:
        protocol_id = manifest.get("protocol_id")
        if (
            not isinstance(protocol_id, str)
            or protocol_id != protocol_id.strip()
            or not canonical_identifier(protocol_id)
        ):
            raise ManifestRunnerError("authorized manifest requires a canonical protocol_id")
        if len(selected_splits) != 1 or not selected_splits.issubset({"dev", "test"}):
            raise ManifestRunnerError("authorized manifest must contain one dev or test split")
        dataset_reasons, selection = validate_dataset_provenance(manifest.get("dataset"))
        dataset_reasons.extend(
            validate_dataset_selection(
                selection,
                splits=selected_splits,
                row_count=len(rows),
                speakers=set(speaker_splits),
            )
        )
        if dataset_reasons:
            raise ManifestRunnerError(
                "authorized manifest has invalid dataset evidence: " + "; ".join(dataset_reasons)
            )
    return manifest, rows


def _artifact(label: str, path: Path) -> dict[str, object]:
    size, sha256 = _hash_file(path)
    return {
        "label": label,
        "name": path.name,
        "bytes": size,
        "sha256": sha256,
    }


def _model_root(path: Path | None, *, label: str) -> Path | None:
    if path is None:
        return None
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ManifestRunnerError(f"{label} must not be a symbolic link: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_dir():
        raise ManifestRunnerError(f"{label} directory does not exist: {resolved}")
    return resolved


def _resolve_required(
    model_dir: Path, label: str, candidates: tuple[str, ...]
) -> dict[str, object]:
    path = resolve_model_file(model_dir, None, candidates=candidates)
    if path is None:
        raise ManifestRunnerError(f"unable to resolve {label} in {model_dir}")
    return _artifact(label, path)


def _model_artifacts(
    backend: str,
    *,
    model_dir: Path | None,
    verifier_model: Path | None,
    model_type: str,
    dual_pass: bool,
) -> list[dict[str, object]]:
    if backend == "fake":
        return []
    if model_dir is None:
        raise ManifestRunnerError("model_dir is required for a real backend")
    artifacts = [_resolve_required(model_dir, "tokens", TOKENS_CANDIDATES)]
    artifacts.append(_resolve_required(model_dir, "encoder", ENCODER_CANDIDATES))
    artifacts.append(_resolve_required(model_dir, "decoder", DECODER_CANDIDATES))
    if model_type != "paraformer":
        artifacts.append(_resolve_required(model_dir, "joiner", JOINER_CANDIDATES))
    if dual_pass:
        if verifier_model is None:
            raise ManifestRunnerError("verifier_model is required when dual_pass is enabled")
        for filename in LOCAL_MODEL_REQUIRED_FILES:
            path = verifier_model / filename
            if not path.is_file():
                raise ManifestRunnerError(f"verifier model file is missing: {path}")
            artifacts.append(_artifact(f"verifier_{filename}", path))
        vocabulary_path = next(
            (
                verifier_model / name
                for name in LOCAL_MODEL_VOCABULARY_FILES
                if (verifier_model / name).is_file()
            ),
            None,
        )
        if vocabulary_path is None:
            raise ManifestRunnerError("verifier model vocabulary.json/txt is missing")
        artifacts.append(_artifact("verifier_vocabulary", vocabulary_path))
        for filename in LOCAL_MODEL_OPTIONAL_FILES:
            path = verifier_model / filename
            if path.is_file():
                artifacts.append(_artifact(f"verifier_{filename}", path))
    return artifacts


def _actual_decoder_configs(
    backend: str,
    recognizer: StreamingRecognizer,
    finalizer: EndpointFinalizer | None,
    *,
    dual_pass: bool,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Read the settings from the adapter instances that perform inference."""

    if backend == "fake":
        return None, None

    config = getattr(recognizer, "config", None)
    required_streaming = (
        "model_type",
        "provider",
        "num_threads",
        "decoding_method",
        "sample_rate",
    )
    if config is None or any(not hasattr(config, field) for field in required_streaming):
        raise ManifestRunnerError("streaming adapter does not expose its decoder configuration")
    streaming_config: dict[str, object] = {
        field: getattr(config, field) for field in required_streaming
    }

    if not dual_pass:
        if finalizer is not None:
            raise ManifestRunnerError("single-pass run unexpectedly created a verifier")
        return streaming_config, None
    if finalizer is None:
        raise ManifestRunnerError("dual-pass run did not create a verifier")
    required_verifier = (
        "device",
        "compute_type",
        "cpu_threads",
        "beam_size",
        "language",
    )
    if any(not hasattr(finalizer, field) for field in required_verifier):
        raise ManifestRunnerError("verifier adapter does not expose its decoder configuration")
    verifier_config: dict[str, object] = {
        field: getattr(finalizer, field) for field in required_verifier
    }
    return streaming_config, verifier_config


def _validated_revision(value: str | None, *, label: str) -> str:
    revision = (value or "").strip()
    if not valid_revision(revision):
        raise ManifestRunnerError(f"{label} must not be a fixture or floating placeholder")
    return revision


def _provenance_entry(
    *,
    label: str,
    source_url: str | None,
    revision: str,
    license_name: str | None,
    license_reviewed: bool,
    required: bool,
) -> dict[str, object]:
    if not isinstance(license_reviewed, bool):
        raise ManifestRunnerError(f"{label}_license_reviewed must be a boolean")
    normalized_url = (source_url or "").strip()
    normalized_license = (license_name or "").strip()
    if source_url is not None and (
        normalized_url != source_url or not normalized_url.startswith("https://")
    ):
        raise ManifestRunnerError(f"{label}_source_url must be a canonical HTTPS URL")
    if license_name is not None and (
        normalized_license != license_name
        or not normalized_license
        or normalized_license.casefold() in {"unknown", "none", "n/a"}
    ):
        raise ManifestRunnerError(f"{label}_license must name a reviewed model-weight license")
    if license_reviewed and (not normalized_url or not normalized_license):
        raise ManifestRunnerError(
            f"{label} license review requires source_url and license metadata"
        )
    if required and (not license_reviewed or not normalized_url or not normalized_license):
        raise ManifestRunnerError(
            f"authorized evaluation requires reviewed {label} model provenance"
        )
    return {
        "source_url": normalized_url or None,
        "revision": revision,
        "license": normalized_license or None,
        "license_reviewed": license_reviewed,
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"echoforge-asr": __version__}
    for distribution in (
        "numpy",
        "sherpa-onnx",
        "faster-whisper",
        "ctranslate2",
        "onnxruntime",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _run_utterance(
    recognizer: StreamingRecognizer,
    finalizer: EndpointFinalizer | None,
    audio: FloatAudio,
    *,
    chunk_samples: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    partial_count = 0
    first_partial_audio_ms: int | None = None
    first_partial_wall_ms: float | None = None
    try:
        for offset in range(0, int(audio.size), chunk_samples):
            partial = recognizer.accept_audio(audio[offset : offset + chunk_samples], 16_000)
            if partial is not None:
                partial_count += 1
                if first_partial_audio_ms is None:
                    first_partial_audio_ms = partial.audio_end_ms
                    first_partial_wall_ms = (time.perf_counter() - started) * 1000
        endpoint_started = time.perf_counter()
        stream_final = recognizer.finalize()
        stream_finalize_ms = (time.perf_counter() - endpoint_started) * 1000
        stream_elapsed_ms = (time.perf_counter() - started) * 1000
        endpoint_to_final_ms = stream_finalize_ms
        verifier_compute_ms: float | None = None
        verified_text: str | None = None
        decoder_score: float | None = None
        hypothesis = stream_final.text
        final_stage = stream_final.stage.value
        if finalizer is not None:
            verifier_started = time.perf_counter()
            verified = finalizer.transcribe(audio, 16_000)
            verifier_compute_ms = (time.perf_counter() - verifier_started) * 1000
            endpoint_to_final_ms = (time.perf_counter() - endpoint_started) * 1000
            if verified.degraded:
                raise ManifestRunnerError(
                    f"verifier returned a degraded result: {verified.degradation_code or 'unknown'}"
                )
            verified_text = verified.text
            decoder_score = verified.decoder_score
            hypothesis = verified.text
            final_stage = verified.stage.value
        total_compute_ms = (time.perf_counter() - started) * 1000
    finally:
        recognizer.reset()

    audio_duration_s = round(int(audio.size) / 16_000, 6)
    rounded_total_compute_ms = round(total_compute_ms, 3)
    return {
        "hypothesis": hypothesis,
        "stream_final": stream_final.text,
        "verified_final": verified_text,
        "final_stage": final_stage,
        "decoder_score": decoder_score,
        "partial_updates": partial_count,
        "first_partial_audio_ms": first_partial_audio_ms,
        "first_partial_wall_ms": (
            round(first_partial_wall_ms, 3) if first_partial_wall_ms is not None else None
        ),
        "stream_compute_ms": round(stream_elapsed_ms, 3),
        "stream_finalize_ms": round(stream_finalize_ms, 3),
        "verifier_compute_ms": (
            round(verifier_compute_ms, 3) if verifier_compute_ms is not None else None
        ),
        "endpoint_to_final_ms": round(endpoint_to_final_ms, 3),
        "total_compute_ms": rounded_total_compute_ms,
        "audio_duration_s": audio_duration_s,
        "utterance_rtf": round(
            (rounded_total_compute_ms / 1000) / audio_duration_s,
            6,
        ),
    }


def run_manifest(
    manifest_path: Path,
    output_path: Path,
    *,
    audio_root: Path,
    backend: str = "fake",
    model_dir: Path | None = None,
    verifier_model: Path | None = None,
    model_type: str = "zipformer",
    provider: str = "cpu",
    dual_pass: bool = True,
    chunk_ms: int = 200,
    streaming_revision: str | None = None,
    verifier_revision: str | None = None,
    streaming_source_url: str | None = None,
    streaming_license: str | None = None,
    streaming_license_reviewed: bool = False,
    verifier_source_url: str | None = None,
    verifier_license: str | None = None,
    verifier_license_reviewed: bool = False,
    warmup_audio: Path | None = None,
) -> dict[str, Any]:
    """Run all prepared rows and atomically emit a frozen evaluation manifest."""

    if isinstance(chunk_ms, bool) or not isinstance(chunk_ms, int) or not 20 <= chunk_ms <= 1000:
        raise ManifestRunnerError("chunk_ms must be an integer in [20, 1000]")
    if not isinstance(dual_pass, bool):
        raise ManifestRunnerError("dual_pass must be a boolean")
    if model_type not in {"zipformer", "paraformer", "transducer"}:
        raise ManifestRunnerError("model_type must be zipformer, paraformer, or transducer")
    if provider not in {"cpu", "cuda"}:
        raise ManifestRunnerError("provider must be cpu or cuda")
    if not isinstance(backend, str):
        raise ManifestRunnerError("backend must be fake or sherpa-onnx")
    normalized_backend = backend.strip().lower()
    if normalized_backend not in {"fake", "sherpa", "sherpa-onnx"}:
        raise ManifestRunnerError("backend must be fake or sherpa-onnx")
    if not dual_pass and (
        verifier_model is not None
        or (verifier_revision or "").strip()
        or verifier_source_url is not None
        or verifier_license is not None
        or verifier_license_reviewed
    ):
        raise ManifestRunnerError(
            "verifier model and provenance arguments must be omitted for a single-pass run"
        )
    if normalized_backend == "fake" and any(
        value is not None
        for value in (
            model_dir,
            verifier_model,
            streaming_revision,
            verifier_revision,
            streaming_source_url,
            streaming_license,
            verifier_source_url,
            verifier_license,
        )
    ):
        raise ManifestRunnerError("fake backend must not declare real model or provenance data")
    if normalized_backend == "fake" and (streaming_license_reviewed or verifier_license_reviewed):
        raise ManifestRunnerError("fake backend must not declare reviewed model provenance")
    resolved_streaming_revision = "deterministic-fixture"
    resolved_verifier_revision: str | None = "deterministic-fixture" if dual_pass else None
    if normalized_backend != "fake":
        resolved_streaming_revision = _validated_revision(
            streaming_revision,
            label="streaming_revision",
        )
        resolved_verifier_revision = (
            _validated_revision(verifier_revision, label="verifier_revision") if dual_pass else None
        )

    manifest_path = normalized_path(manifest_path)
    output_path = normalized_path(output_path)
    if output_path == manifest_path:
        raise ManifestRunnerError("output path must differ from the source manifest")
    try:
        ensure_new_json_path(output_path)
    except FileExistsError as exc:
        raise ManifestRunnerError(f"output already exists or is reserved: {output_path}") from exc
    source_bytes = manifest_path.read_bytes()
    try:
        decoded = strict_json_loads(source_bytes)
    except EvidenceJsonError as exc:
        raise ManifestRunnerError(f"manifest is not valid strict JSON: {exc}") from exc
    manifest, rows = _validate_manifest(decoded)
    root = audio_root.expanduser().resolve()
    if not root.is_dir():
        raise ManifestRunnerError(f"audio_root directory does not exist: {root}")
    authorized_real_run = manifest["evaluation_authorized"] is True and normalized_backend != "fake"
    if authorized_real_run and provider != "cpu":
        raise ManifestRunnerError(
            "authorized evaluation currently supports CPU only; CUDA device evidence is incomplete"
        )
    streaming_provenance: dict[str, object] | None = None
    verifier_provenance: dict[str, object] | None = None
    if normalized_backend != "fake":
        streaming_provenance = _provenance_entry(
            label="streaming",
            source_url=streaming_source_url,
            revision=resolved_streaming_revision,
            license_name=streaming_license,
            license_reviewed=streaming_license_reviewed,
            required=authorized_real_run,
        )
        if dual_pass:
            assert resolved_verifier_revision is not None
            verifier_provenance = _provenance_entry(
                label="verifier",
                source_url=verifier_source_url,
                revision=resolved_verifier_revision,
                license_name=verifier_license,
                license_reviewed=verifier_license_reviewed,
                required=authorized_real_run,
            )
    if authorized_real_run and warmup_audio is None:
        raise ManifestRunnerError("authorized real evaluation requires separate warmup audio")
    warmup_samples: FloatAudio | None = None
    warmup_hash: str | None = None
    if warmup_audio is not None:
        warmup_path = warmup_audio.expanduser().resolve()
        if not warmup_path.is_file():
            raise ManifestRunnerError(f"warmup audio does not exist: {warmup_path}")
        warmup_samples, warmup_hash = _load_hashed_pcm16_wav(warmup_path)
        evaluation_hashes = {str(row["audio_sha256"]).lower() for row in rows}
        if authorized_real_run and warmup_hash in evaluation_hashes:
            raise ManifestRunnerError("authorized warmup audio must not reuse an evaluation row")

    # Freeze a content snapshot before preflight, adapter construction, or lazy
    # model loading.  A second snapshot below makes any mutation during the run
    # a hard failure instead of silently publishing mixed-model evidence.
    model_dir = _model_root(model_dir, label="model_dir")
    verifier_model = _model_root(verifier_model, label="verifier_model")
    artifacts = _model_artifacts(
        normalized_backend,
        model_dir=model_dir,
        verifier_model=verifier_model,
        model_type=model_type,
        dual_pass=dual_pass,
    )
    try:
        factories = build_backend_factories(
            normalized_backend,
            model_dir=model_dir,
            verifier_model=verifier_model,
            model_type=model_type,
            provider=provider,
            dual_pass=dual_pass,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise ManifestRunnerError(f"backend preflight failed: {exc}") from exc
    recognizer = factories.recognizer_factory()
    finalizer = factories.finalizer_factory()
    streaming_config, verifier_config = _actual_decoder_configs(
        normalized_backend,
        recognizer,
        finalizer,
        dual_pass=dual_pass,
    )
    chunk_samples = round(chunk_ms * 16)

    warmup: dict[str, object] = {"performed": False}
    if warmup_samples is not None and warmup_hash is not None:
        warmup_result = _run_utterance(
            recognizer, finalizer, warmup_samples, chunk_samples=chunk_samples
        )
        warmup = {
            "performed": True,
            "audio_sha256": warmup_hash,
            "audio_duration_s": warmup_result["audio_duration_s"],
            "total_compute_ms": warmup_result["total_compute_ms"],
        }

    output_rows: list[dict[str, Any]] = []
    total_audio_s = 0.0
    total_compute_ms = 0.0
    for index, row in enumerate(rows):
        path = _resolve_audio_path(root, row.get("audio_relpath"))
        audio, actual_hash = _load_hashed_pcm16_wav(path)
        expected_hash = str(row["audio_sha256"]).lower()
        if actual_hash != expected_hash:
            raise ManifestRunnerError(
                f"row {index} audio hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
        result = _run_utterance(recognizer, finalizer, audio, chunk_samples=chunk_samples)
        total_audio_s += float(result["audio_duration_s"])
        total_compute_ms += float(result["total_compute_ms"])
        output_rows.append(
            {
                "id": row["id"],
                "speaker_id": row["speaker_id"],
                "split": row["split"],
                "audio_relpath": row["audio_relpath"],
                "audio_sha256": actual_hash,
                "reference": row["reference"],
                **result,
            }
        )

    try:
        final_artifacts = _model_artifacts(
            normalized_backend,
            model_dir=model_dir,
            verifier_model=verifier_model,
            model_type=model_type,
            dual_pass=dual_pass,
        )
    except (OSError, ValueError) as exc:
        raise ManifestRunnerError("model artifacts became unreadable during inference") from exc
    if final_artifacts != artifacts:
        raise ManifestRunnerError("model artifacts changed during inference")
    effective_authorization = (
        manifest["evaluation_authorized"] is True and normalized_backend != "fake"
    )
    output: dict[str, Any] = {
        "schema_version": "echoforge.eval-manifest/v1",
        "dataset": manifest["dataset"],
        "protocol_id": manifest.get("protocol_id"),
        "evaluation_authorized": effective_authorization,
        "frozen": effective_authorization,
        "model": {
            "backend": factories.name,
            "streaming_revision": resolved_streaming_revision,
            "verifier_revision": resolved_verifier_revision,
            "streaming_config": streaming_config,
            "verifier_config": verifier_config,
            "provenance": {
                "streaming": streaming_provenance,
                "verifier": verifier_provenance,
            },
            "artifacts": artifacts,
            "declared_provenance": manifest.get("model"),
        },
        "normalization": manifest.get("normalization", "echoforge.zh-normalizer/v1"),
        "runner": {
            "schema_version": "echoforge.runner/v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "authorization_requested": manifest["evaluation_authorized"],
            "authorization_effective": effective_authorization,
            "chunk_ms": chunk_ms,
            "clock": ("time.perf_counter wall time; offline unpaced sequential single process"),
            "rtf_basis": "sequential ASR wall time / decoded utterance duration",
            "timing_definitions": {
                "first_partial_audio_ms": (
                    "audio consumed by the streaming decoder when its first non-empty partial "
                    "was emitted"
                ),
                "first_partial_wall_ms": (
                    "offline unpaced perf_counter wall time from the first streaming call until "
                    "the first non-empty partial; not real-time playback TTFT"
                ),
                "stream_compute_ms": (
                    "offline processing wall time for all streaming accepts plus "
                    "recognizer.finalize"
                ),
                "stream_finalize_ms": (
                    "perf_counter wall time spent inside recognizer.finalize after input ended"
                ),
                "verifier_compute_ms": (
                    "perf_counter wall time spent in endpoint verifier transcription; null when "
                    "disabled"
                ),
                "endpoint_to_final_ms": (
                    "perf_counter wall time from immediately before streaming finalize until the "
                    "selected final result"
                ),
                "total_compute_ms": (
                    "offline processing wall time for streaming accepts, streaming finalize, "
                    "and the optional verifier"
                ),
                "utterance_rtf": "total_compute_ms / (1000 * decoded utterance duration seconds)",
                "aggregate_utterance_rtf": (
                    "sum of utterance total_compute_ms / (1000 * sum of decoded audio seconds)"
                ),
            },
            "warmup": warmup,
            "packages": _package_versions(),
            "device": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python": sys.version.split()[0],
                "logical_cpus": os.cpu_count(),
            },
        },
        "runtime_summary": {
            "rows": len(output_rows),
            "total_audio_s": round(total_audio_s, 6),
            "total_compute_ms": round(total_compute_ms, 3),
            "aggregate_utterance_rtf": round((total_compute_ms / 1000) / total_audio_s, 6),
        },
        "rows": output_rows,
    }
    try:
        write_json_new(output_path, output)
    except (EvidenceJsonError, FileExistsError, OSError) as exc:
        raise ManifestRunnerError(f"unable to publish output evidence: {exc}") from exc
    return output


__all__ = ["ManifestRunnerError", "run_manifest"]
