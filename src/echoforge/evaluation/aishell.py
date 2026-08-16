"""Deterministic AISHELL-1 prepared-manifest construction."""

from __future__ import annotations

import hashlib
import io
import os
import re
import wave
from pathlib import Path
from typing import Any

from .aishell_extract import TRANSCRIPT_NAME, AishellExtractionError, validate_aishell_extraction
from .evidence_io import (
    EvidenceJsonError,
    ensure_new_json_path,
    normalized_path,
    strict_json_loads,
    write_json_new,
)
from .normalize_zh import normalize_zh
from .runner import ManifestRunnerError

OPENSLR_ARCHIVES = {
    "audio": "https://www.openslr.org/resources/33/data_aishell.tgz",
    "resources": "https://www.openslr.org/resources/33/resource_aishell.tgz",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _load_download_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        payload = strict_json_loads(raw)
    except EvidenceJsonError as exc:
        raise ManifestRunnerError(f"download manifest is not valid strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestRunnerError("download manifest root must be an object")
    if payload.get("schema_version") != "echoforge.download/v1":
        raise ManifestRunnerError("unsupported download manifest schema")
    dry_run = payload.get("dry_run")
    if payload.get("dataset") != "AISHELL-1" or dry_run is not False:
        raise ManifestRunnerError(
            "download manifest does not describe a completed AISHELL-1 download"
        )
    if not _valid_sha256(payload.get("license_text_sha256")):
        raise ManifestRunnerError("download manifest has an invalid license-page SHA-256")
    license_declared = payload.get("license_declared")
    if (
        not isinstance(license_declared, str)
        or license_declared != license_declared.strip()
        or "Apache-2.0" not in license_declared
    ):
        raise ManifestRunnerError("download manifest has no reviewed Apache-2.0 declaration")
    archives = payload.get("archives")
    if not isinstance(archives, list) or len(archives) != len(OPENSLR_ARCHIVES):
        raise ManifestRunnerError("download manifest must contain exactly audio and resources")
    seen_names: set[str] = set()
    for index, value in enumerate(archives):
        if not isinstance(value, dict):
            raise ManifestRunnerError(f"download archive record {index} must be an object")
        name = value.get("name")
        if not isinstance(name, str) or name not in OPENSLR_ARCHIVES or name in seen_names:
            raise ManifestRunnerError(
                "download manifest must contain one audio and one resources archive"
            )
        seen_names.add(name)
        if value.get("url") != OPENSLR_ARCHIVES[name]:
            raise ManifestRunnerError(f"download archive {name} has an unexpected OpenSLR URL")
        if not _valid_sha256(value.get("sha256")):
            raise ManifestRunnerError(f"download archive {name} has an invalid SHA-256")
        byte_count = value.get("bytes")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
            raise ManifestRunnerError(f"download archive {name} has an invalid byte count")
    if seen_names != set(OPENSLR_ARCHIVES):
        raise ManifestRunnerError(
            "download manifest must contain one audio and one resources archive"
        )
    return dict(payload), hashlib.sha256(raw).hexdigest()


def _load_transcripts(path: Path) -> tuple[dict[str, str], str]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestRunnerError("transcript file is not valid UTF-8") from exc
    transcripts: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ManifestRunnerError(f"invalid transcript line {line_number}")
        utterance_id, text = parts
        reference = "".join(text.split())
        if utterance_id in transcripts:
            raise ManifestRunnerError(f"duplicate transcript id: {utterance_id}")
        if not normalize_zh(reference):
            raise ManifestRunnerError(f"empty normalized transcript: {utterance_id}")
        transcripts[utterance_id] = reference
    if not transcripts:
        raise ManifestRunnerError("transcript file contains no records")
    return transcripts, hashlib.sha256(raw).hexdigest()


def _wav_evidence(path: Path) -> tuple[int, float, str]:
    raw = path.read_bytes()
    try:
        with wave.open(io.BytesIO(raw), "rb") as source:
            sample_rate = source.getframerate()
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            compression = source.getcomptype()
            frames = source.getnframes()
            audio_bytes = source.readframes(frames)
    except (EOFError, wave.Error) as exc:
        raise ManifestRunnerError(f"invalid WAV file: {path}") from exc
    if sample_rate != 16_000 or channels != 1 or sample_width != 2 or compression != "NONE":
        raise ManifestRunnerError(f"AISHELL WAV must be 16 kHz mono PCM16LE: {path}")
    if frames <= 0:
        raise ManifestRunnerError(f"AISHELL WAV contains no frames: {path}")
    expected_audio_bytes = frames * channels * sample_width
    if len(audio_bytes) != expected_audio_bytes:
        raise ManifestRunnerError(
            f"AISHELL WAV audio data is truncated: expected {expected_audio_bytes} bytes, "
            f"got {len(audio_bytes)}: {path}"
        )
    return frames, frames / sample_rate, hashlib.sha256(raw).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _resolve_directory(path: Path, *, root: Path | None, label: str) -> Path:
    if path.is_symlink():
        raise ManifestRunnerError(f"{label} must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ManifestRunnerError(f"{label} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise ManifestRunnerError(f"{label} is not a directory: {path}")
    if root is not None and not _is_within(resolved, root):
        raise ManifestRunnerError(f"{label} escapes wav_root: {path}")
    return resolved


def _resolve_wav(path: Path, *, speaker_root: Path, wav_root: Path) -> Path:
    if path.is_symlink():
        raise ManifestRunnerError(f"AISHELL WAV must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ManifestRunnerError(f"AISHELL WAV does not exist: {path}") from exc
    if not resolved.is_file():
        raise ManifestRunnerError(f"AISHELL WAV is not a regular file: {path}")
    if not _is_within(resolved, wav_root) or not _is_within(resolved, speaker_root):
        raise ManifestRunnerError(f"AISHELL WAV escapes its speaker directory: {path}")
    return resolved


def prepare_aishell_manifest(
    output_path: Path,
    *,
    wav_root: Path,
    transcript_path: Path,
    download_manifest_path: Path,
    splits: tuple[str, ...] = ("dev",),
    speaker_limit: int | None = None,
    utterances_per_speaker: int | None = None,
    evaluation_authorized: bool = False,
    protocol_id: str | None = None,
    extraction_root: Path | None = None,
) -> dict[str, Any]:
    """Create a deterministic, unfrozen AISHELL manifest without running ASR."""

    if not isinstance(evaluation_authorized, bool):
        raise ManifestRunnerError("evaluation_authorized must be a boolean")
    if not splits or any(split not in {"dev", "test"} for split in splits):
        raise ManifestRunnerError("splits must contain only dev and/or test")
    if len(set(splits)) != len(splits):
        raise ManifestRunnerError("splits must not contain duplicates")
    if evaluation_authorized and len(splits) != 1:
        raise ManifestRunnerError("authorized evaluation must select one dev or test split")
    if evaluation_authorized and (speaker_limit is not None or utterances_per_speaker is not None):
        raise ManifestRunnerError("authorized evaluation must use the complete selected split")
    if evaluation_authorized and extraction_root is None:
        raise ManifestRunnerError("authorized evaluation requires verified extraction evidence")
    if speaker_limit is not None and (isinstance(speaker_limit, bool) or speaker_limit < 1):
        raise ManifestRunnerError("speaker_limit must be positive when set")
    if utterances_per_speaker is not None and (
        isinstance(utterances_per_speaker, bool) or utterances_per_speaker < 1
    ):
        raise ManifestRunnerError("utterances_per_speaker must be positive when set")
    if evaluation_authorized and protocol_id is None:
        raise ManifestRunnerError("protocol_id is required when evaluation is authorized")
    if protocol_id is not None and (
        not isinstance(protocol_id, str)
        or protocol_id != protocol_id.strip()
        or IDENTIFIER_PATTERN.fullmatch(protocol_id) is None
    ):
        raise ManifestRunnerError("protocol_id must be a canonical identifier")

    output_path = normalized_path(output_path)
    try:
        ensure_new_json_path(output_path)
    except FileExistsError as exc:
        raise ManifestRunnerError(f"output already exists or is reserved: {output_path}") from exc

    root_input = Path(os.path.abspath(wav_root.expanduser()))
    root = _resolve_directory(root_input, root=None, label="wav_root")
    transcript = transcript_path.expanduser().resolve()
    download_manifest = download_manifest_path.expanduser().resolve()
    if not transcript.is_file():
        raise ManifestRunnerError(f"transcript file does not exist: {transcript}")
    if not download_manifest.is_file():
        raise ManifestRunnerError(f"download manifest does not exist: {download_manifest}")

    download_payload, download_manifest_sha256 = _load_download_manifest(download_manifest)
    extraction_evidence: dict[str, Any] | None = None
    extraction_path: Path | None = None
    if extraction_root is not None:
        extraction_path = extraction_root.expanduser().resolve()
        try:
            validated_extraction = validate_aishell_extraction(
                extraction_path,
                expected_download_manifest_sha256=download_manifest_sha256,
            )
        except AishellExtractionError as exc:
            raise ManifestRunnerError(f"invalid AISHELL extraction evidence: {exc}") from exc
        expected_wav_root = (extraction_path / "wav").resolve()
        expected_transcript = (extraction_path / "transcript" / TRANSCRIPT_NAME).resolve()
        if root != expected_wav_root or transcript != expected_transcript:
            raise ManifestRunnerError(
                "wav_root and transcript must be the paths inside the verified extraction root"
            )
        marker_splits = validated_extraction.get("splits")
        if not isinstance(marker_splits, list) or not set(splits).issubset(set(marker_splits)):
            raise ManifestRunnerError("requested splits are absent from extraction evidence")
        marker_limit = validated_extraction.get("speaker_limit_per_split")
        if evaluation_authorized and marker_limit is not None:
            raise ManifestRunnerError("authorized evaluation requires a complete extraction split")
        extraction_evidence = dict(validated_extraction)
    transcripts, transcript_sha256 = _load_transcripts(transcript)
    rows: list[dict[str, Any]] = []
    seen_audio_hashes: set[str] = set()
    seen_audio_relpaths: set[str] = set()
    speaker_splits: dict[str, set[str]] = {}
    selected_speakers: dict[str, list[str]] = {}

    for split in splits:
        split_root = _resolve_directory(
            root / split,
            root=root,
            label="AISHELL split directory",
        )
        speaker_dirs: list[Path] = []
        for candidate in sorted(split_root.iterdir(), key=lambda path: path.name):
            if candidate.is_symlink():
                raise ManifestRunnerError(
                    f"AISHELL speaker directory must not be a symbolic link: {candidate}"
                )
            if candidate.is_dir():
                speaker_dirs.append(
                    _resolve_directory(
                        candidate,
                        root=split_root,
                        label="AISHELL speaker directory",
                    )
                )
        if speaker_limit is not None:
            speaker_dirs = speaker_dirs[:speaker_limit]
        if not speaker_dirs:
            raise ManifestRunnerError(f"AISHELL split has no speaker directories: {split}")
        selected_speakers[split] = [path.name for path in speaker_dirs]
        for speaker_dir in speaker_dirs:
            speaker_id = speaker_dir.name
            if IDENTIFIER_PATTERN.fullmatch(speaker_id) is None:
                raise ManifestRunnerError(f"invalid AISHELL speaker identifier: {speaker_id}")
            speaker_splits.setdefault(speaker_id, set()).add(split)
            wavs = [
                _resolve_wav(path, speaker_root=speaker_dir, wav_root=root)
                for path in sorted(speaker_dir.iterdir(), key=lambda path: path.name)
                if path.suffix == ".wav"
            ]
            if utterances_per_speaker is not None:
                wavs = wavs[:utterances_per_speaker]
            if not wavs:
                raise ManifestRunnerError(f"speaker has no WAV files: {speaker_dir}")
            for wav_path in wavs:
                utterance_id = wav_path.stem
                if IDENTIFIER_PATTERN.fullmatch(utterance_id) is None:
                    raise ManifestRunnerError(
                        f"invalid AISHELL utterance identifier: {utterance_id}"
                    )
                if utterance_id not in transcripts:
                    raise ManifestRunnerError(f"transcript is missing utterance: {utterance_id}")
                frames, duration_s, audio_sha256 = _wav_evidence(wav_path)
                audio_relpath = wav_path.relative_to(root).as_posix()
                if audio_sha256 in seen_audio_hashes:
                    raise ManifestRunnerError(
                        f"selected AISHELL rows reuse an audio SHA-256: {utterance_id}"
                    )
                if audio_relpath.casefold() in seen_audio_relpaths:
                    raise ManifestRunnerError(
                        f"selected AISHELL rows reuse an audio path: {audio_relpath}"
                    )
                seen_audio_hashes.add(audio_sha256)
                seen_audio_relpaths.add(audio_relpath.casefold())
                rows.append(
                    {
                        "id": utterance_id,
                        "speaker_id": speaker_id,
                        "split": split,
                        "audio_relpath": audio_relpath,
                        "audio_sha256": audio_sha256,
                        "audio_frames": frames,
                        "audio_duration_s": round(duration_s, 6),
                        "reference": transcripts[utterance_id],
                    }
                )

    overlap = sorted(speaker for speaker, split_set in speaker_splits.items() if len(split_set) > 1)
    if overlap:
        raise ManifestRunnerError("speakers occur in multiple splits: " + ", ".join(overlap))
    row_ids = [str(row["id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ManifestRunnerError("selected AISHELL rows contain duplicate utterance IDs")
    if extraction_evidence is not None:
        tree = extraction_evidence.get("tree")
        marker_speakers = tree.get("selected_speakers") if isinstance(tree, dict) else None
        if not isinstance(marker_speakers, dict) or any(
            not isinstance(marker_speakers.get(split), list)
            or not set(selected_speakers[split]).issubset(set(marker_speakers[split]))
            for split in splits
        ):
            raise ManifestRunnerError("selected speakers do not match extraction evidence")
        assert extraction_path is not None
        try:
            final_extraction_evidence = validate_aishell_extraction(
                extraction_path,
                expected_download_manifest_sha256=download_manifest_sha256,
            )
        except AishellExtractionError as exc:
            raise ManifestRunnerError(
                f"AISHELL extraction changed during manifest preparation: {exc}"
            ) from exc
        if final_extraction_evidence != extraction_evidence:
            raise ManifestRunnerError("AISHELL extraction changed during manifest preparation")

    output: dict[str, Any] = {
        "schema_version": "echoforge.eval-manifest/v1",
        "dataset": {
            "name": "AISHELL-1",
            "source": "OpenSLR 33",
            "source_url": "https://www.openslr.org/33/",
            "license": download_payload.get("license_declared"),
            "license_page_sha256": download_payload.get("license_text_sha256"),
            "download_manifest_sha256": download_manifest_sha256,
            "transcript_sha256": transcript_sha256,
            "extraction_marker_sha256": (
                extraction_evidence.get("marker_sha256")
                if extraction_evidence is not None
                else None
            ),
            "extraction_inventory_sha256": (
                extraction_evidence.get("tree", {}).get("inventory_sha256")
                if extraction_evidence is not None
                and isinstance(extraction_evidence.get("tree"), dict)
                else None
            ),
            "speaker_policy": "speaker-disjoint selected dev/test splits",
            "audio_protocol": "16 kHz mono PCM16LE",
            "raw_audio_in_repository": False,
            "selection": {
                "splits": list(splits),
                "speaker_limit_per_split": speaker_limit,
                "utterances_per_speaker": utterances_per_speaker,
                "extraction_speaker_limit_per_split": (
                    extraction_evidence.get("speaker_limit_per_split")
                    if extraction_evidence is not None
                    else None
                ),
                "selected_speakers": selected_speakers,
                "rows": len(rows),
            },
        },
        "evaluation_authorized": evaluation_authorized,
        "protocol_id": protocol_id,
        "frozen": False,
        "model": {
            "streaming_backend": "not yet executed",
            "verifier_backend": "not yet executed",
            "files": [],
        },
        "normalization": "echoforge.zh-normalizer/v1",
        "rows": rows,
        "notes": [
            "Prepared locally from hash-recorded OpenSLR archives.",
            "Do not commit row-level transcripts or audio paths to the public repository.",
        ],
    }
    try:
        write_json_new(output_path, output)
    except (EvidenceJsonError, FileExistsError, OSError) as exc:
        raise ManifestRunnerError(f"unable to publish prepared manifest: {exc}") from exc
    return output


__all__ = ["prepare_aishell_manifest"]
