"""Safe, selective extraction of the nested OpenSLR AISHELL-1 archives."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import tarfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .evidence_io import EvidenceJsonError, strict_json_loads, write_json_new

AUDIO_URL = "https://www.openslr.org/resources/33/data_aishell.tgz"
RESOURCES_URL = "https://www.openslr.org/resources/33/resource_aishell.tgz"
ARCHIVE_URLS = {"audio": AUDIO_URL, "resources": RESOURCES_URL}
MARKER_NAME = ".echoforge-aishell-extraction.json"
SCHEMA_VERSION = "echoforge.aishell-extraction/v1"
TRANSCRIPT_NAME = "aishell_transcript_v0.8.txt"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SPEAKER_PATTERN = re.compile(r"^S[0-9]{4}$")
_NESTED_ARCHIVE_PATTERN = re.compile(r"^(S[0-9]{4})\.tar\.gz$")
_WAV_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}\.wav$")
_TRANSCRIPT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_MARKER_BYTES = 4 * 1024 * 1024
_MAX_NESTED_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_TRANSCRIPT_BYTES = 128 * 1024 * 1024
_MAX_RESOURCE_BYTES = 256 * 1024 * 1024
_MAX_WAV_BYTES = 512 * 1024 * 1024


class AishellExtractionError(RuntimeError):
    """Raised when AISHELL evidence or archive layout is unsafe or inconsistent."""


@dataclass(frozen=True)
class _ArchiveRecord:
    name: str
    url: str
    byte_count: int
    sha256: str

    def marker_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "url": self.url,
            "bytes": self.byte_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class _VerifiedArchive:
    path: Path
    record: _ArchiveRecord
    signature: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class _TreeEvidence:
    inventory_sha256: str
    file_count: int
    transcript_sha256: str
    wav_files: dict[str, int]
    speakers: dict[str, int]
    selected_speakers: dict[str, list[str]]

    def marker_record(self) -> dict[str, object]:
        return {
            "inventory_sha256": self.inventory_sha256,
            "file_count": self.file_count,
            "transcript_sha256": self.transcript_sha256,
            "wav_files": self.wav_files,
            "speakers": self.speakers,
            "selected_speakers": self.selected_speakers,
        }


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _lstat(path: Path, *, label: str) -> os.stat_result | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode):
        raise AishellExtractionError(f"{label} must not be a symbolic link: {path}")
    return metadata


def _regular_file(path: Path, *, label: str) -> os.stat_result:
    metadata = _lstat(path, label=label)
    if metadata is None:
        raise AishellExtractionError(f"{label} does not exist: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise AishellExtractionError(f"{label} must be a regular file: {path}")
    return metadata


def _directory(path: Path, *, label: str) -> os.stat_result:
    metadata = _lstat(path, label=label)
    if metadata is None:
        raise AishellExtractionError(f"{label} does not exist: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise AishellExtractionError(f"{label} must be a directory: {path}")
    return metadata


def _signature(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_small_regular(path: Path, *, label: str, limit: int) -> bytes:
    metadata = _regular_file(path, label=label)
    if metadata.st_size > limit:
        raise AishellExtractionError(f"{label} exceeds the {limit}-byte safety limit: {path}")
    return path.read_bytes()


def _json_object(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        parsed: object = strict_json_loads(raw)
    except EvidenceJsonError as exc:
        raise AishellExtractionError(f"{label} is not valid strict JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise AishellExtractionError(f"{label} root must be an object")
    return cast(dict[str, object], parsed)


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _load_download_manifest(path: Path) -> tuple[str, dict[str, _ArchiveRecord]]:
    raw = _read_small_regular(
        path,
        label="download manifest",
        limit=_MAX_MANIFEST_BYTES,
    )
    payload = _json_object(raw, label="download manifest")
    if payload.get("schema_version") != "echoforge.download/v1":
        raise AishellExtractionError("unsupported download manifest schema")
    if payload.get("dataset") != "AISHELL-1":
        raise AishellExtractionError("download manifest is not for AISHELL-1")
    if payload.get("dry_run") is not False:
        raise AishellExtractionError("download manifest is not a completed download")
    if not _valid_sha256(payload.get("license_text_sha256")):
        raise AishellExtractionError("download manifest has no valid license-page SHA-256")
    license_declared = payload.get("license_declared")
    if (
        not isinstance(license_declared, str)
        or license_declared != license_declared.strip()
        or "Apache-2.0" not in license_declared
    ):
        raise AishellExtractionError("download manifest has no reviewed Apache-2.0 declaration")

    archive_values = payload.get("archives")
    if not isinstance(archive_values, list) or len(archive_values) != 2:
        raise AishellExtractionError("download manifest must contain exactly two archives")
    records: dict[str, _ArchiveRecord] = {}
    for index, value in enumerate(archive_values):
        if not isinstance(value, dict):
            raise AishellExtractionError(f"download archive record {index} must be an object")
        item = cast(dict[str, object], value)
        name = item.get("name")
        if not isinstance(name, str) or name not in ARCHIVE_URLS or name in records:
            raise AishellExtractionError(
                "download manifest must contain one audio and one resources archive"
            )
        expected_url = ARCHIVE_URLS[name]
        if item.get("url") != expected_url:
            raise AishellExtractionError(f"download archive {name} has an unexpected URL")
        byte_count = item.get("bytes")
        digest = item.get("sha256")
        if not _positive_int(byte_count):
            raise AishellExtractionError(f"download archive {name} has an invalid byte count")
        if not _valid_sha256(digest):
            raise AishellExtractionError(f"download archive {name} has an invalid SHA-256")
        assert isinstance(byte_count, int)
        assert isinstance(digest, str)
        records[name] = _ArchiveRecord(name, expected_url, byte_count, digest)
    if set(records) != set(ARCHIVE_URLS):
        raise AishellExtractionError(
            "download manifest must contain one audio and one resources archive"
        )
    return hashlib.sha256(raw).hexdigest(), records


def _verify_archive(path: Path, record: _ArchiveRecord) -> _VerifiedArchive:
    metadata = _regular_file(path, label=f"{record.name} archive")
    if metadata.st_size != record.byte_count:
        raise AishellExtractionError(
            f"{record.name} archive byte count does not match the download manifest"
        )
    signature = _signature(metadata)
    digest = _sha256(path)
    current = _regular_file(path, label=f"{record.name} archive")
    if _signature(current) != signature:
        raise AishellExtractionError(f"{record.name} archive changed while it was hashed")
    if digest != record.sha256:
        raise AishellExtractionError(
            f"{record.name} archive SHA-256 does not match the download manifest"
        )
    return _VerifiedArchive(path=path, record=record, signature=signature)


def _assert_archive_unchanged(archive: _VerifiedArchive) -> None:
    metadata = _regular_file(archive.path, label=f"{archive.record.name} archive")
    if _signature(metadata) != archive.signature:
        raise AishellExtractionError(
            f"{archive.record.name} archive changed after evidence verification"
        )


def _safe_member_parts(member: tarfile.TarInfo, *, archive_label: str) -> tuple[str, ...]:
    raw = member.name
    if not raw or "\x00" in raw or "\\" in raw:
        raise AishellExtractionError(f"unsafe {archive_label} member path: {raw!r}")
    pieces = raw.split("/")
    if member.isdir() and pieces[-1] == "":
        pieces.pop()
    if (
        not pieces
        or raw.startswith("/")
        or any(piece in {"", ".", ".."} or ":" in piece for piece in pieces)
    ):
        raise AishellExtractionError(f"{archive_label} member path traversal is not allowed: {raw}")
    return tuple(pieces)


def _validate_member_type(member: tarfile.TarInfo, *, archive_label: str) -> None:
    if member.issym() or member.islnk():
        raise AishellExtractionError(f"{archive_label} links are not allowed: {member.name}")
    if member.isdev() or member.isfifo():
        raise AishellExtractionError(f"{archive_label} devices are not allowed: {member.name}")
    if not member.isdir() and not member.isreg():
        raise AishellExtractionError(f"unsupported {archive_label} member type: {member.name}")


def _ensure_directory(root: Path, parts: tuple[str, ...]) -> Path:
    current = root
    for part in parts:
        current = current / part
        metadata = _lstat(current, label="extraction directory")
        if metadata is None:
            current.mkdir()
        elif not stat.S_ISDIR(metadata.st_mode):
            raise AishellExtractionError(f"extraction directory is not a directory: {current}")
    return current


def _copy_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
    *,
    root: Path,
    limit: int,
) -> str:
    if not member.isreg():
        raise AishellExtractionError(f"archive member is not a regular file: {member.name}")
    if member.size < 0 or member.size > limit:
        raise AishellExtractionError(
            f"archive member exceeds its safety limit: {member.name} ({member.size} bytes)"
        )
    try:
        relative = destination.relative_to(root)
    except ValueError as exc:
        raise AishellExtractionError(
            f"extraction destination escapes staging: {destination}"
        ) from exc
    _ensure_directory(root, relative.parts[:-1])
    if _lstat(destination, label="extraction output") is not None:
        raise AishellExtractionError(f"duplicate extraction output: {destination}")
    source = archive.extractfile(member)
    if source is None:
        raise AishellExtractionError(f"cannot read archive member: {member.name}")
    digest = hashlib.sha256()
    copied = 0
    try:
        with source, destination.open("xb") as target:
            while block := source.read(1024 * 1024):
                copied += len(block)
                if copied > member.size or copied > limit:
                    raise AishellExtractionError(
                        f"archive member expanded beyond its declared size: {member.name}"
                    )
                digest.update(block)
                target.write(block)
        if copied != member.size:
            raise AishellExtractionError(
                f"archive member length mismatch: {member.name} ({copied} != {member.size})"
            )
    except BaseException:
        if destination.exists():
            destination.unlink()
        raise
    return digest.hexdigest()


def _validate_wav(path: Path) -> None:
    _regular_file(path, label="AISHELL WAV")
    try:
        with wave.open(str(path), "rb") as source:
            frame_count = source.getnframes()
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != 16_000
                or source.getcomptype() != "NONE"
                or frame_count <= 0
            ):
                raise AishellExtractionError(
                    f"AISHELL WAV must be non-empty 16 kHz mono PCM16LE: {path}"
                )
            expected_audio_bytes = frame_count * 2
            actual_audio_bytes = 0
            while block := source.readframes(min(65_536, frame_count)):
                actual_audio_bytes += len(block)
                if actual_audio_bytes > expected_audio_bytes:
                    break
            if actual_audio_bytes != expected_audio_bytes:
                raise AishellExtractionError(f"AISHELL WAV has truncated PCM data: {path}")
    except (EOFError, wave.Error) as exc:
        raise AishellExtractionError(f"invalid AISHELL WAV: {path}") from exc


def _validate_transcript(path: Path) -> tuple[set[str], str]:
    raw = _read_small_regular(path, label="AISHELL transcript", limit=_MAX_TRANSCRIPT_BYTES)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AishellExtractionError("AISHELL transcript is not valid UTF-8") from exc
    identifiers: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if (
            len(fields) != 2
            or _TRANSCRIPT_ID_PATTERN.fullmatch(fields[0]) is None
            or not fields[1].strip()
        ):
            raise AishellExtractionError(f"invalid AISHELL transcript line {line_number}")
        if fields[0] in identifiers:
            raise AishellExtractionError(f"duplicate AISHELL transcript ID: {fields[0]}")
        identifiers.add(fields[0])
    if not identifiers:
        raise AishellExtractionError("AISHELL transcript contains no records")
    return identifiers, hashlib.sha256(raw).hexdigest()


def _extract_nested_speaker(
    nested_path: Path,
    *,
    speaker_id: str,
    staging: Path,
    extract_splits: set[str],
) -> tuple[str, int]:
    member_names: set[tuple[str, ...]] = set()
    observed_split: str | None = None
    wav_count = 0
    try:
        with (
            nested_path.open("rb") as source,
            tarfile.open(fileobj=source, mode="r|*") as archive,
        ):
            for member in archive:
                _validate_member_type(member, archive_label="nested speaker archive")
                parts = _safe_member_parts(member, archive_label="nested speaker archive")
                if parts in member_names:
                    raise AishellExtractionError(
                        f"duplicate nested speaker archive member: {member.name}"
                    )
                member_names.add(parts)
                if member.isdir():
                    if (len(parts) == 1 and parts[0] in {"train", "dev", "test"}) or (
                        len(parts) == 2
                        and parts[0] in {"train", "dev", "test"}
                        and parts[1] == speaker_id
                    ):
                        continue
                    raise AishellExtractionError(
                        f"unexpected nested speaker directory: {member.name}"
                    )
                if (
                    len(parts) != 3
                    or parts[0] not in {"train", "dev", "test"}
                    or parts[1] != speaker_id
                    or _WAV_NAME_PATTERN.fullmatch(parts[2]) is None
                ):
                    raise AishellExtractionError(
                        f"invalid nested split/speaker/WAV path: {member.name}"
                    )
                split = parts[0]
                if observed_split is not None and split != observed_split:
                    raise AishellExtractionError(
                        f"speaker archive contains multiple splits: {speaker_id}"
                    )
                observed_split = split
                wav_count += 1
                if split in extract_splits:
                    destination = staging / "wav" / split / speaker_id / parts[2]
                    _copy_member(
                        archive,
                        member,
                        destination,
                        root=staging,
                        limit=_MAX_WAV_BYTES,
                    )
                    _validate_wav(destination)
    except (tarfile.TarError, EOFError) as exc:
        raise AishellExtractionError(f"invalid nested speaker archive for {speaker_id}") from exc
    if observed_split is None or wav_count == 0:
        raise AishellExtractionError(f"speaker archive contains no WAV files: {speaker_id}")
    return observed_split, wav_count


def _extract_audio_archive(
    verified: _VerifiedArchive,
    staging: Path,
    splits: tuple[str, ...],
    speaker_limit_per_split: int | None,
) -> None:
    selected_splits = set(splits)
    seen_members: set[tuple[str, ...]] = set()
    seen_speakers: set[str] = set()
    selected_speakers: dict[str, list[str]] = {split: [] for split in splits}
    previous_speaker_by_split: dict[str, str] = {}
    transcript_seen = False
    nested_temporary = staging / ".current-speaker.tar.gz"
    allowed_directories = {
        ("data_aishell",),
        ("data_aishell", "wav"),
        ("data_aishell", "transcript"),
    }
    try:
        with (
            verified.path.open("rb") as source,
            tarfile.open(fileobj=source, mode="r|*") as archive,
        ):
            for member in archive:
                _validate_member_type(member, archive_label="audio archive")
                parts = _safe_member_parts(member, archive_label="audio archive")
                if parts in seen_members:
                    raise AishellExtractionError(f"duplicate audio archive member: {member.name}")
                seen_members.add(parts)
                if member.isdir():
                    if parts not in allowed_directories:
                        raise AishellExtractionError(
                            f"unexpected audio archive directory: {member.name}"
                        )
                    continue
                if parts == ("data_aishell", "transcript", TRANSCRIPT_NAME):
                    if transcript_seen:
                        raise AishellExtractionError("audio archive has duplicate transcripts")
                    _copy_member(
                        archive,
                        member,
                        staging / "transcript" / TRANSCRIPT_NAME,
                        root=staging,
                        limit=_MAX_TRANSCRIPT_BYTES,
                    )
                    transcript_seen = True
                    continue
                if len(parts) != 3 or parts[:2] != ("data_aishell", "wav"):
                    raise AishellExtractionError(f"unexpected audio archive file: {member.name}")
                match = _NESTED_ARCHIVE_PATTERN.fullmatch(parts[2])
                if match is None:
                    raise AishellExtractionError(f"unexpected audio archive file: {member.name}")
                speaker_id = match.group(1)
                if speaker_id in seen_speakers:
                    raise AishellExtractionError(f"duplicate AISHELL speaker archive: {speaker_id}")
                seen_speakers.add(speaker_id)
                _copy_member(
                    archive,
                    member,
                    nested_temporary,
                    root=staging,
                    limit=_MAX_NESTED_ARCHIVE_BYTES,
                )
                try:
                    extract_splits = {
                        split
                        for split in selected_splits
                        if speaker_limit_per_split is None
                        or len(selected_speakers[split]) < speaker_limit_per_split
                    }
                    split, _ = _extract_nested_speaker(
                        nested_temporary,
                        speaker_id=speaker_id,
                        staging=staging,
                        extract_splits=extract_splits,
                    )
                finally:
                    if nested_temporary.exists():
                        nested_temporary.unlink()
                previous_speaker = previous_speaker_by_split.get(split)
                if previous_speaker is not None and speaker_id <= previous_speaker:
                    raise AishellExtractionError(
                        "audio speaker archives must be strictly ordered within each split"
                    )
                previous_speaker_by_split[split] = speaker_id
                if split in extract_splits:
                    selected_speakers[split].append(speaker_id)
    except (tarfile.TarError, EOFError) as exc:
        raise AishellExtractionError("invalid AISHELL audio archive") from exc
    if not transcript_seen:
        raise AishellExtractionError("audio archive is missing the AISHELL transcript")
    for split, speakers in selected_speakers.items():
        if not speakers:
            raise AishellExtractionError(f"audio archive contains no selected {split} speakers")


def _extract_resources_archive(verified: _VerifiedArchive, staging: Path) -> None:
    expected_files: dict[tuple[str, ...], str] = {
        ("resource_aishell", "lexicon.txt"): "lexicon.txt",
        ("resource_aishell", "speaker.info"): "speaker.info",
    }
    seen_members: set[tuple[str, ...]] = set()
    extracted: set[tuple[str, ...]] = set()
    try:
        with (
            verified.path.open("rb") as source,
            tarfile.open(fileobj=source, mode="r|*") as archive,
        ):
            for member in archive:
                _validate_member_type(member, archive_label="resources archive")
                parts = _safe_member_parts(member, archive_label="resources archive")
                if parts in seen_members:
                    raise AishellExtractionError(
                        f"duplicate resources archive member: {member.name}"
                    )
                seen_members.add(parts)
                if member.isdir():
                    if parts != ("resource_aishell",):
                        raise AishellExtractionError(
                            f"unexpected resources archive directory: {member.name}"
                        )
                    continue
                output_name = expected_files.get(parts)
                if output_name is None:
                    raise AishellExtractionError(
                        f"unexpected resources archive file: {member.name}"
                    )
                _copy_member(
                    archive,
                    member,
                    staging / "resources" / output_name,
                    root=staging,
                    limit=_MAX_RESOURCE_BYTES,
                )
                extracted.add(parts)
    except (tarfile.TarError, EOFError) as exc:
        raise AishellExtractionError("invalid AISHELL resources archive") from exc
    if extracted != set(expected_files):
        raise AishellExtractionError("resources archive is missing lexicon.txt or speaker.info")


def _entry_names(path: Path, *, label: str) -> set[str]:
    _directory(path, label=label)
    return {entry.name for entry in os.scandir(path)}


def _inventory_digest(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        metadata = _regular_file(path, label="extracted file")
        relative = path.relative_to(root).as_posix()
        file_sha256 = _sha256(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(metadata.st_size).encode("ascii"))
        digest.update(b"\x00")
        digest.update(file_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_tree(
    root: Path,
    splits: tuple[str, ...],
    *,
    marker_expected: bool,
    speaker_limit_per_split: int | None,
) -> _TreeEvidence:
    expected_root = {"wav", "transcript", "resources"}
    if marker_expected:
        expected_root.add(MARKER_NAME)
    actual_root = _entry_names(root, label="AISHELL extraction root")
    if actual_root != expected_root:
        raise AishellExtractionError("AISHELL extraction root has an unexpected layout")

    transcript_dir = root / "transcript"
    if _entry_names(transcript_dir, label="transcript directory") != {TRANSCRIPT_NAME}:
        raise AishellExtractionError("AISHELL transcript directory has an unexpected layout")
    transcript = transcript_dir / TRANSCRIPT_NAME
    transcript_ids, transcript_sha256 = _validate_transcript(transcript)

    resources = root / "resources"
    resource_names = {"lexicon.txt", "speaker.info"}
    if _entry_names(resources, label="resources directory") != resource_names:
        raise AishellExtractionError("AISHELL resources directory has an unexpected layout")
    files = [transcript]
    for name in sorted(resource_names):
        path = resources / name
        _regular_file(path, label="AISHELL resource")
        files.append(path)

    wav_root = root / "wav"
    if _entry_names(wav_root, label="WAV root") != set(splits):
        raise AishellExtractionError("AISHELL WAV root does not match the selected splits")
    wav_counts: dict[str, int] = {}
    speaker_counts: dict[str, int] = {}
    selected_speakers: dict[str, list[str]] = {}
    observed_speakers: set[str] = set()
    observed_utterances: set[str] = set()
    for split in splits:
        split_root = wav_root / split
        speaker_names = _entry_names(split_root, label=f"{split} split directory")
        if not speaker_names:
            raise AishellExtractionError(f"AISHELL {split} split contains no speakers")
        wav_counts[split] = 0
        speaker_counts[split] = len(speaker_names)
        selected_speakers[split] = sorted(speaker_names)
        if speaker_limit_per_split is not None and len(speaker_names) > speaker_limit_per_split:
            raise AishellExtractionError(
                f"AISHELL {split} split exceeds its configured speaker limit"
            )
        for speaker_id in sorted(speaker_names):
            if _SPEAKER_PATTERN.fullmatch(speaker_id) is None:
                raise AishellExtractionError(f"invalid AISHELL speaker directory: {speaker_id}")
            if speaker_id in observed_speakers:
                raise AishellExtractionError(f"AISHELL speaker occurs in two splits: {speaker_id}")
            observed_speakers.add(speaker_id)
            speaker_root = split_root / speaker_id
            wav_names = _entry_names(speaker_root, label="AISHELL speaker directory")
            if not wav_names:
                raise AishellExtractionError(f"AISHELL speaker has no WAV files: {speaker_id}")
            for wav_name in sorted(wav_names):
                if _WAV_NAME_PATTERN.fullmatch(wav_name) is None:
                    raise AishellExtractionError(f"invalid AISHELL WAV filename: {wav_name}")
                utterance_id = wav_name[:-4]
                if utterance_id in observed_utterances:
                    raise AishellExtractionError(f"duplicate AISHELL utterance ID: {utterance_id}")
                if utterance_id not in transcript_ids:
                    raise AishellExtractionError(
                        f"AISHELL transcript is missing utterance: {utterance_id}"
                    )
                observed_utterances.add(utterance_id)
                wav_path = speaker_root / wav_name
                _validate_wav(wav_path)
                files.append(wav_path)
                wav_counts[split] += 1

    return _TreeEvidence(
        inventory_sha256=_inventory_digest(root, files),
        file_count=len(files),
        transcript_sha256=transcript_sha256,
        wav_files=wav_counts,
        speakers=speaker_counts,
        selected_speakers=selected_speakers,
    )


def _marker_base(
    download_manifest_sha256: str,
    archives: dict[str, _VerifiedArchive],
    splits: tuple[str, ...],
    speaker_limit_per_split: int | None,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "AISHELL-1",
        "download_manifest_sha256": download_manifest_sha256,
        "archives": [archives[name].record.marker_record() for name in ("audio", "resources")],
        "splits": list(splits),
        "speaker_limit_per_split": speaker_limit_per_split,
    }


def _write_marker(path: Path, payload: dict[str, object]) -> None:
    try:
        write_json_new(path, payload)
    except (EvidenceJsonError, FileExistsError, OSError) as exc:
        raise AishellExtractionError(f"unable to publish extraction marker: {exc}") from exc


def _load_and_validate_marker(
    root: Path,
    marker_base: dict[str, object],
    splits: tuple[str, ...],
    speaker_limit_per_split: int | None,
) -> _TreeEvidence:
    marker_path = root / MARKER_NAME
    raw = _read_small_regular(marker_path, label="extraction marker", limit=_MAX_MARKER_BYTES)
    marker = _json_object(raw, label="extraction marker")
    expected_keys = {*marker_base, "complete", "tree"}
    if set(marker) != expected_keys or marker.get("complete") is not True:
        raise AishellExtractionError("extraction marker is incomplete or malformed")
    for key, expected_value in marker_base.items():
        if marker.get(key) != expected_value:
            raise AishellExtractionError(f"extraction marker does not match {key}")
    evidence = _validate_tree(
        root,
        splits,
        marker_expected=True,
        speaker_limit_per_split=speaker_limit_per_split,
    )
    if marker.get("tree") != evidence.marker_record():
        raise AishellExtractionError("extraction tree no longer matches its completion marker")
    return evidence


def _remove_staging(staging: Path, *, target: Path) -> None:
    expected = target.parent / f".{target.name}.echoforge-extracting"
    if staging != expected:
        raise AishellExtractionError(f"refusing to remove unexpected staging path: {staging}")
    metadata = _lstat(staging, label="extraction staging directory")
    if metadata is None:
        return
    if not stat.S_ISDIR(metadata.st_mode):
        raise AishellExtractionError(f"extraction staging path must be a directory: {staging}")
    shutil.rmtree(staging)


def _promote_staging(staging: Path, target: Path) -> None:
    _directory(staging, label="completed extraction staging directory")
    if _lstat(target, label="extraction output") is not None:
        raise AishellExtractionError(f"refusing to overwrite existing output: {target}")
    try:
        os.rename(staging, target)
    except FileExistsError as exc:
        raise AishellExtractionError(f"refusing to overwrite existing output: {target}") from exc


def _prepare_output_path(output: Path) -> Path:
    raw = _absolute(output)
    if raw == raw.parent or raw.name in {"", ".", ".."}:
        raise AishellExtractionError(f"invalid extraction output path: {output}")
    raw.parent.mkdir(parents=True, exist_ok=True)
    parent = raw.parent.resolve(strict=True)
    _directory(parent, label="extraction output parent")
    return parent / raw.name


def _result(
    target: Path,
    splits: tuple[str, ...],
    manifest_sha256: str,
    evidence: _TreeEvidence,
    *,
    reused: bool,
) -> dict[str, object]:
    return {
        "status": "extracted",
        "output": str(target),
        "splits": list(splits),
        "reused": reused,
        "download_manifest_sha256": manifest_sha256,
        "tree": evidence.marker_record(),
    }


def validate_aishell_extraction(
    root: Path,
    *,
    expected_download_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Recompute a published extraction tree and verify its hash-bound marker."""

    target = _absolute(root)
    _directory(target, label="AISHELL extraction root")
    marker_path = target / MARKER_NAME
    raw = _read_small_regular(marker_path, label="extraction marker", limit=_MAX_MARKER_BYTES)
    marker = _json_object(raw, label="extraction marker")
    expected_keys = {
        "schema_version",
        "dataset",
        "download_manifest_sha256",
        "archives",
        "splits",
        "speaker_limit_per_split",
        "complete",
        "tree",
    }
    if (
        set(marker) != expected_keys
        or marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("dataset") != "AISHELL-1"
        or marker.get("complete") is not True
        or not _valid_sha256(marker.get("download_manifest_sha256"))
    ):
        raise AishellExtractionError("extraction marker is incomplete or malformed")
    manifest_hash = cast(str, marker["download_manifest_sha256"])
    if (
        expected_download_manifest_sha256 is not None
        and manifest_hash != expected_download_manifest_sha256
    ):
        raise AishellExtractionError("extraction marker does not match download manifest hash")

    archive_values = marker.get("archives")
    if not isinstance(archive_values, list) or len(archive_values) != 2:
        raise AishellExtractionError("extraction marker has invalid archive evidence")
    archive_names: set[str] = set()
    for value in archive_values:
        if not isinstance(value, dict):
            raise AishellExtractionError("extraction marker has invalid archive evidence")
        name = value.get("name")
        if (
            not isinstance(name, str)
            or name not in ARCHIVE_URLS
            or name in archive_names
            or value.get("url") != ARCHIVE_URLS[name]
            or not _positive_int(value.get("bytes"))
            or not _valid_sha256(value.get("sha256"))
        ):
            raise AishellExtractionError("extraction marker has invalid archive evidence")
        archive_names.add(name)
    if archive_names != set(ARCHIVE_URLS):
        raise AishellExtractionError("extraction marker has invalid archive evidence")

    split_values = marker.get("splits")
    if (
        not isinstance(split_values, list)
        or not split_values
        or any(split not in {"dev", "test"} for split in split_values)
        or split_values != [split for split in ("dev", "test") if split in split_values]
    ):
        raise AishellExtractionError("extraction marker has invalid split evidence")
    splits = tuple(cast(list[str], split_values))
    limit = marker.get("speaker_limit_per_split")
    if limit is not None and not _positive_int(limit):
        raise AishellExtractionError("extraction marker has invalid speaker limit")
    speaker_limit = cast(int | None, limit)
    evidence = _validate_tree(
        target,
        splits,
        marker_expected=True,
        speaker_limit_per_split=speaker_limit,
    )
    if marker.get("tree") != evidence.marker_record():
        raise AishellExtractionError("extraction tree no longer matches its completion marker")
    return {
        "marker_sha256": hashlib.sha256(raw).hexdigest(),
        "download_manifest_sha256": manifest_hash,
        "splits": list(splits),
        "speaker_limit_per_split": speaker_limit,
        "tree": evidence.marker_record(),
    }


def extract_aishell(
    output: Path,
    *,
    download_manifest_path: Path,
    audio_archive_path: Path,
    resources_archive_path: Path,
    splits: tuple[str, ...] = ("dev", "test"),
    speaker_limit_per_split: int | None = None,
) -> dict[str, object]:
    """Verify and selectively extract AISHELL-1 into an atomically published tree."""

    if not splits or any(split not in {"dev", "test"} for split in splits):
        raise AishellExtractionError("splits must contain only dev and/or test")
    if len(set(splits)) != len(splits):
        raise AishellExtractionError("splits must not contain duplicates")
    if speaker_limit_per_split is not None and (
        not isinstance(speaker_limit_per_split, int)
        or isinstance(speaker_limit_per_split, bool)
        or speaker_limit_per_split < 1
    ):
        raise AishellExtractionError("speaker_limit_per_split must be a positive integer")
    canonical_splits = tuple(split for split in ("dev", "test") if split in splits)

    manifest_path = _absolute(download_manifest_path)
    audio_path = _absolute(audio_archive_path)
    resources_path = _absolute(resources_archive_path)
    manifest_sha256, records = _load_download_manifest(manifest_path)
    audio = _verify_archive(audio_path, records["audio"])
    resources = _verify_archive(resources_path, records["resources"])
    try:
        if os.path.samefile(audio.path, resources.path):
            raise AishellExtractionError("audio and resources archives must be distinct files")
    except OSError as exc:
        raise AishellExtractionError("cannot compare the two archive paths") from exc
    archives = {"audio": audio, "resources": resources}

    target = _prepare_output_path(output)
    staging = target.parent / f".{target.name}.echoforge-extracting"
    marker_base = _marker_base(
        manifest_sha256,
        archives,
        canonical_splits,
        speaker_limit_per_split,
    )

    target_metadata = _lstat(target, label="extraction output")
    if target_metadata is not None:
        if not stat.S_ISDIR(target_metadata.st_mode):
            raise AishellExtractionError(f"extraction output must be a directory: {target}")
        try:
            evidence = _load_and_validate_marker(
                target,
                marker_base,
                canonical_splits,
                speaker_limit_per_split,
            )
        except (AishellExtractionError, OSError) as exc:
            raise AishellExtractionError(
                f"refusing to overwrite or trust existing output directory: {target}: {exc}"
            ) from exc
        _remove_staging(staging, target=target)
        return _result(
            target,
            canonical_splits,
            manifest_sha256,
            evidence,
            reused=True,
        )

    staging_metadata = _lstat(staging, label="extraction staging directory")
    if staging_metadata is not None:
        if not stat.S_ISDIR(staging_metadata.st_mode):
            raise AishellExtractionError(f"extraction staging path must be a directory: {staging}")
        try:
            evidence = _load_and_validate_marker(
                staging,
                marker_base,
                canonical_splits,
                speaker_limit_per_split,
            )
        except (AishellExtractionError, OSError):
            _remove_staging(staging, target=target)
        else:
            _promote_staging(staging, target)
            return _result(
                target,
                canonical_splits,
                manifest_sha256,
                evidence,
                reused=True,
            )

    staging.mkdir()
    _ensure_directory(staging, ("wav",))
    for split in canonical_splits:
        _ensure_directory(staging, ("wav", split))
    _ensure_directory(staging, ("transcript",))
    _ensure_directory(staging, ("resources",))
    # Failures intentionally leave only the controlled staging tree. A later invocation
    # validates and rebuilds it; no partial tree is published under the requested output.
    _assert_archive_unchanged(audio)
    _extract_audio_archive(
        audio,
        staging,
        canonical_splits,
        speaker_limit_per_split,
    )
    _assert_archive_unchanged(resources)
    _extract_resources_archive(resources, staging)
    _assert_archive_unchanged(audio)
    _assert_archive_unchanged(resources)
    evidence = _validate_tree(
        staging,
        canonical_splits,
        marker_expected=False,
        speaker_limit_per_split=speaker_limit_per_split,
    )
    marker = {
        **marker_base,
        "complete": True,
        "tree": evidence.marker_record(),
    }
    _write_marker(staging / MARKER_NAME, marker)
    _load_and_validate_marker(
        staging,
        marker_base,
        canonical_splits,
        speaker_limit_per_split,
    )
    _promote_staging(staging, target)
    return _result(
        target,
        canonical_splits,
        manifest_sha256,
        evidence,
        reused=False,
    )


__all__ = ["AishellExtractionError", "extract_aishell", "validate_aishell_extraction"]
