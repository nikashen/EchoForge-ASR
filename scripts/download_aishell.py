"""Opt-in AISHELL-1 downloader with hash and extraction safety checks."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import stat
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from echoforge import __version__
from echoforge.evaluation.evidence_io import (
    EvidenceJsonError,
    strict_json_dumps,
    strict_json_loads,
)

ARCHIVES = {
    "audio": "https://www.openslr.org/resources/33/data_aishell.tgz",
    "resources": "https://www.openslr.org/resources/33/resource_aishell.tgz",
}
LICENSE_URL = "https://www.openslr.org/33/"
CONTENT_RANGE_PATTERN = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
EXTRACTION_MARKER = ".echoforge-extraction.json"
DOWNLOAD_RECEIPT_SCHEMA = "echoforge.download-receipt/v1"
DOWNLOAD_MIGRATION_SCHEMA = "echoforge.download-migration/v1"
USER_AGENT = f"EchoForge-ASR/{__version__}"


@dataclass(frozen=True)
class _LegacyArchiveAttestation:
    name: str
    archive: Path
    probe: dict[str, object]
    archive_bytes: int
    archive_sha256: str
    resumed_from_bytes: int
    receipt_path: Path
    receipt_payload: dict[str, object]
    sidecar_path: Path
    partial_path: Path


def _safe_output(root: Path) -> Path:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _lstat(path: Path, *, label: str) -> os.stat_result | None:
    """Return lstat for an existing regular path without following links."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"{label} must not be a symbolic link: {path}")
    return metadata


def _regular_file(path: Path, *, label: str) -> os.stat_result | None:
    metadata = _lstat(path, label=label)
    if metadata is not None and not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be a regular file: {path}")
    return metadata


def _stat_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _sha256(path: Path) -> tuple[int, str]:
    """Return stable size/hash evidence, rejecting mutation or path replacement."""

    path_before = _regular_file(path, label="hashed file")
    if path_before is None:
        raise FileNotFoundError(f"hashed file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        descriptor_before = os.fstat(handle.fileno())
        if not stat.S_ISREG(descriptor_before.st_mode):
            raise RuntimeError(f"hashed file must remain a regular file: {path}")
        if _stat_signature(path_before) != _stat_signature(descriptor_before):
            raise RuntimeError(f"hashed file changed before hashing began: {path}")
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        descriptor_after = os.fstat(handle.fileno())
    path_after = _regular_file(path, label="hashed file")
    if path_after is None or not (
        _stat_signature(descriptor_before)
        == _stat_signature(descriptor_after)
        == _stat_signature(path_after)
    ):
        raise RuntimeError(f"hashed file changed while hashing: {path}")
    return descriptor_after.st_size, digest.hexdigest()


def _read_stable_json(path: Path, *, label: str) -> dict[str, object]:
    before_size, before_hash = _sha256(path)
    try:
        raw = path.read_bytes()
        decoded = strict_json_loads(raw)
    except (EvidenceJsonError, OSError) as exc:
        raise RuntimeError(f"{label} is not complete strict JSON: {path}: {exc}") from exc
    after_size, after_hash = _sha256(path)
    if (
        before_size != after_size
        or before_hash != after_hash
        or len(raw) != before_size
        or hashlib.sha256(raw).hexdigest() != before_hash
    ):
        raise RuntimeError(f"{label} changed while it was read: {path}")
    if not isinstance(decoded, dict):
        raise TypeError(f"{label} root must be an object: {path}")
    return dict(decoded)


def _directory(path: Path, *, label: str) -> os.stat_result | None:
    metadata = _lstat(path, label=label)
    if metadata is not None and not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"{label} must be a directory: {path}")
    return metadata


def _promote_new_file(source: Path, destination: Path) -> None:
    """Atomically publish a completed sibling file without overwriting a target."""

    if _regular_file(source, label="promotion source") is None:
        raise FileNotFoundError(f"promotion source does not exist: {source}")
    if _lstat(destination, label="promotion destination") is not None:
        raise FileExistsError(f"refusing to overwrite completed file: {destination}")
    # Hard-link publication is atomic and fails if another process won the target name.
    # Source and destination are always siblings, so cross-device links are not possible.
    os.link(source, destination, follow_symlinks=False)
    source.unlink()


def _write_json_new(path: Path, payload: dict[str, object]) -> None:
    """Durably write JSON and atomically publish it without overwriting evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if _lstat(path, label="JSON output") is not None:
        raise FileExistsError(f"refusing to overwrite JSON evidence: {path}")
    if _regular_file(temporary, label="temporary JSON output") is not None:
        recovered = _read_stable_json(temporary, label="temporary JSON output")
        if recovered != payload:
            raise RuntimeError(
                f"temporary JSON output does not match expected evidence; retained: {temporary}"
            )
        _promote_new_file(temporary, path)
        return
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(strict_json_dumps(payload))
            handle.flush()
            os.fsync(handle.fileno())
        _promote_new_file(temporary, path)
    finally:
        if _regular_file(temporary, label="temporary JSON output") is not None:
            temporary.unlink()


def _response_status(response: object) -> int:
    status_value = getattr(response, "status", None)
    if status_value is not None:
        return int(status_value)
    getcode = response.getcode  # type: ignore[attr-defined]
    return int(getcode())


def _probe_download(url: str) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
        method="HEAD",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        status = _response_status(response)
        if status != 200:
            raise RuntimeError(f"unexpected HEAD status: {status}")
        length_value = response.headers.get("Content-Length")
        if length_value is None or int(length_value) <= 0:
            raise RuntimeError("upstream did not provide a positive Content-Length")
        etag = response.headers.get("ETag")
        last_modified = response.headers.get("Last-Modified")
        if not etag and not last_modified:
            raise RuntimeError("upstream did not provide ETag or Last-Modified")
        accept_ranges = response.headers.get("Accept-Ranges")
    return {
        "url": url,
        "expected_bytes": int(length_value),
        "etag": etag,
        "last_modified": last_modified,
        "accept_ranges": accept_ranges,
    }


def _write_sidecar(path: Path, payload: dict[str, object]) -> None:
    _write_json_new(path, payload)


def _load_sidecar(path: Path) -> dict[str, object]:
    if _regular_file(path, label="download sidecar") is None:
        raise RuntimeError(f"download sidecar does not exist: {path}")
    return _read_stable_json(path, label="download sidecar")


def _validate_probe(saved: dict[str, object], current: dict[str, object]) -> None:
    for field in ("url", "expected_bytes", "etag", "last_modified"):
        if saved.get(field) != current.get(field):
            raise RuntimeError(f"upstream validator changed for resumable download: {field}")


def _positive_int(payload: dict[str, object], field: str, *, label: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{label} has no positive {field}")
    return value


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _receipt_path(destination: Path) -> Path:
    return destination.with_name(destination.name + ".download.json")


def _receipt_payload(
    probe: dict[str, object], *, archive_bytes: int, archive_sha256: str
) -> dict[str, object]:
    return {
        "schema_version": DOWNLOAD_RECEIPT_SCHEMA,
        **probe,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "dry_run": False,
        "complete": True,
    }


def _validate_receipt(
    receipt: dict[str, object],
    probe: dict[str, object],
    *,
    archive_bytes: int,
    archive_sha256: str,
) -> None:
    if (
        receipt.get("schema_version") != DOWNLOAD_RECEIPT_SCHEMA
        or receipt.get("dry_run") is not False
        or receipt.get("complete") is not True
    ):
        raise RuntimeError("completed archive has an invalid validator receipt")
    _validate_probe(receipt, probe)
    if receipt.get("archive_bytes") != archive_bytes:
        raise RuntimeError("completed archive byte count differs from validator receipt")
    saved_hash = receipt.get("archive_sha256")
    if not _valid_sha256(saved_hash) or str(saved_hash).lower() != archive_sha256:
        raise RuntimeError("completed archive SHA-256 differs from validator receipt")


def _persist_receipt_from_sidecar(
    destination: Path,
    sidecar: Path,
    probe: dict[str, object],
    *,
    archive_bytes: int,
    archive_sha256: str,
) -> Path:
    """Persist the temporary validator before removing its only durable copy."""

    if _regular_file(sidecar, label="download sidecar") is None:
        raise RuntimeError("completed archive is missing its temporary validator sidecar")
    _validate_probe(_load_sidecar(sidecar), probe)
    receipt_path = _receipt_path(destination)
    payload = _receipt_payload(probe, archive_bytes=archive_bytes, archive_sha256=archive_sha256)
    if _regular_file(receipt_path, label="download receipt") is None:
        _write_json_new(receipt_path, payload)
    receipt = _load_sidecar(receipt_path)
    _validate_receipt(
        receipt,
        probe,
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
    )
    sidecar.unlink()
    return receipt_path


def _download(url: str, destination: Path) -> dict[str, object]:
    if _regular_file(destination, label="completed archive") is not None:
        raise FileExistsError(f"refusing to overwrite completed archive: {destination}")
    receipt = _receipt_path(destination)
    if _regular_file(receipt, label="download receipt") is not None:
        raise RuntimeError(f"download receipt exists without its completed archive: {receipt}")
    partial = destination.with_name(destination.name + ".part")
    sidecar = destination.with_name(destination.name + ".part.json")
    partial_metadata = _regular_file(partial, label="partial download")
    sidecar_metadata = _regular_file(sidecar, label="download sidecar")
    resumed_from = partial_metadata.st_size if partial_metadata is not None else 0
    probe = _probe_download(url)
    expected_bytes = _positive_int(probe, "expected_bytes", label="download probe")
    if resumed_from > expected_bytes:
        raise RuntimeError(
            f"partial download exceeds upstream length: {resumed_from} > {expected_bytes}"
        )
    if sidecar_metadata is not None:
        _validate_probe(_load_sidecar(sidecar), probe)
    elif partial_metadata is not None:
        raise RuntimeError("partial download is missing its validator sidecar")
    else:
        _write_sidecar(sidecar, probe)

    if resumed_from == expected_bytes and partial_metadata is not None:
        hashed_bytes, digest = _sha256(partial)
        if hashed_bytes != expected_bytes:
            raise RuntimeError("completed partial changed before promotion")
        _promote_new_file(partial, destination)
        receipt = _persist_receipt_from_sidecar(
            destination,
            sidecar,
            probe,
            archive_bytes=expected_bytes,
            archive_sha256=digest,
        )
        return {
            "bytes": expected_bytes,
            "sha256": digest,
            "resumed_from_bytes": resumed_from,
            "etag": probe.get("etag"),
            "last_modified": probe.get("last_modified"),
            "promoted_complete_partial": True,
            "validator_receipt": receipt.name,
        }

    if resumed_from and str(probe.get("accept_ranges") or "").lower() != "bytes":
        raise RuntimeError("upstream does not advertise byte-range support")
    headers = {"User-Agent": USER_AGENT}
    if resumed_from:
        headers["Range"] = f"bytes={resumed_from}-"
        headers["If-Range"] = str(probe.get("etag") or probe["last_modified"])
    elif probe.get("etag"):
        headers["If-Match"] = str(probe["etag"])
    elif probe.get("last_modified"):
        headers["If-Unmodified-Since"] = str(probe["last_modified"])
    request = urllib.request.Request(url, headers=headers)
    try:
        response_context = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            raise RuntimeError(
                "upstream rejected the resume boundary; retained validated partial "
                f"at {resumed_from}/{expected_bytes} bytes"
            ) from exc
        raise
    with response_context as response:
        status = _response_status(response)
        if resumed_from and status != 206:
            raise RuntimeError("server did not honor the resume Range request")
        if not resumed_from and status != 200:
            raise RuntimeError(f"unexpected download status: {status}")
        content_range = response.headers.get("Content-Range")
        response_etag = response.headers.get("ETag")
        response_last_modified = response.headers.get("Last-Modified")
        if probe.get("etag") and response_etag != probe["etag"]:
            raise RuntimeError("GET response ETag differs from the validated upstream object")
        if probe.get("last_modified") and response_last_modified != probe["last_modified"]:
            raise RuntimeError(
                "GET response Last-Modified differs from the validated upstream object"
            )
        response_length = response.headers.get("Content-Length")
        if response_length is None:
            raise RuntimeError("download response is missing Content-Length")
        response_bytes = int(response_length)
        if resumed_from:
            match = CONTENT_RANGE_PATTERN.fullmatch(str(content_range or ""))
            if match is None:
                raise RuntimeError("server returned an invalid Content-Range")
            range_start, range_end, range_total = (int(value) for value in match.groups())
            if (
                range_start != resumed_from
                or range_end < range_start
                or range_end >= range_total
                or range_total != expected_bytes
                or response_bytes != range_end - range_start + 1
            ):
                raise RuntimeError("server returned an inconsistent Content-Range")
        elif response_bytes != expected_bytes:
            raise RuntimeError(
                f"download response length differs from HEAD: {response_bytes} != {expected_bytes}"
            )
        total = resumed_from
        mode = "ab" if partial_metadata is not None else "xb"
        with partial.open(mode) as handle:
            while block := response.read(1024 * 1024):
                total += len(block)
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        if total != expected_bytes:
            raise RuntimeError(
                f"download length mismatch: expected {expected_bytes} bytes, received {total}"
            )
    hashed_bytes, digest = _sha256(partial)
    if hashed_bytes != total:
        raise RuntimeError("download changed before promotion")
    _promote_new_file(partial, destination)
    receipt = _persist_receipt_from_sidecar(
        destination,
        sidecar,
        probe,
        archive_bytes=total,
        archive_sha256=digest,
    )
    return {
        "bytes": total,
        "sha256": digest,
        "resumed_from_bytes": resumed_from,
        "etag": probe.get("etag"),
        "last_modified": probe.get("last_modified"),
        "validator_receipt": receipt.name,
    }


def _verify_completed_download(url: str, destination: Path) -> dict[str, object]:
    destination_metadata = _regular_file(destination, label="completed archive")
    if destination_metadata is None:
        raise FileNotFoundError(f"completed archive does not exist: {destination}")
    probe = _probe_download(url)
    expected_bytes = _positive_int(probe, "expected_bytes", label="download probe")
    actual_bytes = destination_metadata.st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"existing archive length mismatch: expected {expected_bytes}, got {actual_bytes}"
        )
    hashed_bytes, digest = _sha256(destination)
    if hashed_bytes != actual_bytes:
        raise RuntimeError("completed archive changed while it was verified")
    partial = destination.with_name(destination.name + ".part")
    sidecar = destination.with_name(destination.name + ".part.json")
    receipt_path = _receipt_path(destination)
    partial_metadata = _regular_file(partial, label="stale partial download")
    if partial_metadata is not None:
        partial_bytes, partial_hash = _sha256(partial)
        if partial_bytes != actual_bytes or partial_hash != digest:
            raise RuntimeError("completed archive conflicts with stale partial download state")
    sidecar_metadata = _regular_file(sidecar, label="stale download sidecar")
    receipt_metadata = _regular_file(receipt_path, label="download receipt")
    migrated_legacy_validator = False
    if receipt_metadata is not None:
        _validate_receipt(
            _load_sidecar(receipt_path),
            probe,
            archive_bytes=actual_bytes,
            archive_sha256=digest,
        )
        if sidecar_metadata is not None:
            _validate_probe(_load_sidecar(sidecar), probe)
            sidecar.unlink()
    elif sidecar_metadata is not None:
        _validate_probe(_load_sidecar(sidecar), probe)
        _write_json_new(
            receipt_path,
            _receipt_payload(probe, archive_bytes=actual_bytes, archive_sha256=digest),
        )
        _validate_receipt(
            _load_sidecar(receipt_path),
            probe,
            archive_bytes=actual_bytes,
            archive_sha256=digest,
        )
        sidecar.unlink()
        migrated_legacy_validator = True
    else:
        raise RuntimeError(
            "completed archive has no persistent validator receipt or legacy sidecar"
        )
    if partial_metadata is not None:
        partial.unlink()
    return {
        "bytes": actual_bytes,
        "sha256": digest,
        "resumed_from_bytes": actual_bytes,
        "etag": probe.get("etag"),
        "last_modified": probe.get("last_modified"),
        "reused_completed_archive": True,
        "validator_receipt": receipt_path.name,
        "migrated_legacy_validator": migrated_legacy_validator,
    }


def _load_legacy_download_manifest(
    path: Path,
) -> tuple[dict[str, object], dict[str, dict[str, object]], int, str]:
    before_bytes, before_hash = _sha256(path)
    payload = _read_stable_json(path, label="legacy download manifest")
    after_bytes, after_hash = _sha256(path)
    if (before_bytes, before_hash) != (after_bytes, after_hash):
        raise RuntimeError("legacy download manifest changed during attestation")
    if payload.get("schema_version") != "echoforge.download/v1":
        raise RuntimeError("legacy download manifest has an unsupported schema")
    if "dry_run" in payload:
        raise RuntimeError(
            "legacy migration requires the historical manifest with no dry_run field"
        )
    if payload.get("dataset") != "AISHELL-1" or payload.get("source") != "OpenSLR 33":
        raise RuntimeError("legacy download manifest does not identify OpenSLR AISHELL-1")
    license_declared = payload.get("license_declared")
    license_digest = payload.get("license_text_sha256")
    if (
        payload.get("license_url") != LICENSE_URL
        or not isinstance(license_declared, str)
        or license_declared != license_declared.strip()
        or "Apache-2.0" not in license_declared
        or not _valid_sha256(license_digest)
        or license_digest != str(license_digest).lower()
    ):
        raise RuntimeError("legacy download manifest has incomplete license evidence")
    raw_audio_retained = payload.get("raw_audio_retained")
    if not isinstance(raw_audio_retained, bool):
        raise TypeError("legacy download manifest raw_audio_retained must be a boolean")
    if not isinstance(payload.get("extracted"), bool):
        raise TypeError("legacy download manifest extracted must be a boolean")
    archives = payload.get("archives")
    if not isinstance(archives, list) or len(archives) != len(ARCHIVES):
        raise RuntimeError("legacy download manifest must contain exactly two archive records")
    records: dict[str, dict[str, object]] = {}
    for index, value in enumerate(archives):
        if not isinstance(value, dict):
            raise TypeError(f"legacy archive record {index} is not an object")
        record = dict(value)
        name = record.get("name")
        if not isinstance(name, str) or name not in ARCHIVES or name in records:
            raise RuntimeError("legacy manifest must contain one audio and one resources archive")
        if record.get("url") != ARCHIVES[name]:
            raise RuntimeError(f"legacy archive {name} has an unexpected URL")
        expected_bytes = _positive_int(record, "bytes", label=f"legacy archive {name}")
        digest = record.get("sha256")
        if not _valid_sha256(digest) or digest != str(digest).lower():
            raise RuntimeError(f"legacy archive {name} has a non-canonical SHA-256")
        recorded_path = record.get("path")
        if not isinstance(recorded_path, str) or not recorded_path.strip():
            raise RuntimeError(f"legacy archive {name} is missing its recorded path")
        resumed_from = record.get("resumed_from_bytes", 0)
        if (
            isinstance(resumed_from, bool)
            or not isinstance(resumed_from, int)
            or resumed_from < 0
            or resumed_from > expected_bytes
        ):
            raise RuntimeError(f"legacy archive {name} has invalid resumed_from_bytes")
        for validator in ("etag", "last_modified"):
            field = record.get(validator)
            if not isinstance(field, str) or not field.strip():
                raise RuntimeError(f"legacy archive {name} is missing {validator}")
        for boolean_field in (
            "promoted_complete_partial",
            "reused_completed_archive",
            "migrated_legacy_validator",
        ):
            if boolean_field in record and not isinstance(record[boolean_field], bool):
                raise TypeError(f"legacy archive {name} {boolean_field} must be a boolean")
        records[name] = record
    return payload, records, before_bytes, before_hash


def _attest_legacy_archive(
    name: str,
    archive: Path,
    record: dict[str, object],
    *,
    source_manifest_sha256: str,
) -> _LegacyArchiveAttestation:
    """Validate one legacy archive and its state without changing the filesystem."""

    url = ARCHIVES[name]
    probe = _probe_download(url)
    expected_bytes = _positive_int(record, "bytes", label=f"legacy archive {name}")
    saved_probe: dict[str, object] = {
        "url": url,
        "expected_bytes": expected_bytes,
        "etag": record["etag"],
        "last_modified": record["last_modified"],
    }
    _validate_probe(saved_probe, probe)
    actual_bytes, actual_hash = _sha256(archive)
    if actual_bytes != expected_bytes or actual_hash != record["sha256"]:
        raise RuntimeError(f"legacy archive {name} does not match its recorded size/SHA-256")

    receipt_path = _receipt_path(archive)
    receipt_payload = _receipt_payload(
        saved_probe, archive_bytes=actual_bytes, archive_sha256=actual_hash
    )
    receipt_payload.update(
        {
            "migration_schema_version": DOWNLOAD_MIGRATION_SCHEMA,
            "migration_source_manifest_sha256": source_manifest_sha256,
        }
    )
    if _regular_file(receipt_path, label="download receipt") is not None:
        existing_receipt = _read_stable_json(receipt_path, label="download receipt")
        if existing_receipt != receipt_payload:
            raise RuntimeError(
                f"legacy archive {name} receipt is not bound to this migration source"
            )
    receipt_temporary = receipt_path.with_name(receipt_path.name + ".tmp")
    if _regular_file(receipt_temporary, label="temporary download receipt") is not None:
        temporary_receipt = _read_stable_json(receipt_temporary, label="temporary download receipt")
        if temporary_receipt != receipt_payload:
            raise RuntimeError(
                f"legacy archive {name} temporary receipt conflicts with this migration"
            )

    sidecar = archive.with_name(archive.name + ".part.json")
    if _regular_file(sidecar, label="legacy download sidecar") is not None:
        _validate_probe(_load_sidecar(sidecar), probe)
    partial = archive.with_name(archive.name + ".part")
    partial_metadata = _regular_file(partial, label="legacy partial download")
    if partial_metadata is not None:
        partial_bytes, partial_hash = _sha256(partial)
        if partial_bytes != actual_bytes or partial_hash != actual_hash:
            raise RuntimeError(f"legacy archive {name} conflicts with its partial state")
    resumed_from = record.get("resumed_from_bytes", 0)
    assert isinstance(resumed_from, int) and not isinstance(resumed_from, bool)
    return _LegacyArchiveAttestation(
        name=name,
        archive=archive,
        probe=probe,
        archive_bytes=actual_bytes,
        archive_sha256=actual_hash,
        resumed_from_bytes=resumed_from,
        receipt_path=receipt_path,
        receipt_payload=receipt_payload,
        sidecar_path=sidecar,
        partial_path=partial,
    )


def _publish_legacy_archive_attestation(
    attestation: _LegacyArchiveAttestation,
) -> dict[str, object]:
    """Publish and revalidate one receipt, then retire matching temporary state."""

    if _regular_file(attestation.receipt_path, label="download receipt") is None:
        _write_json_new(attestation.receipt_path, attestation.receipt_payload)
    persisted = _read_stable_json(attestation.receipt_path, label="download receipt")
    if persisted != attestation.receipt_payload:
        raise RuntimeError(f"legacy archive {attestation.name} receipt changed during publication")
    actual_bytes, actual_hash = _sha256(attestation.archive)
    if actual_bytes != attestation.archive_bytes or actual_hash != attestation.archive_sha256:
        raise RuntimeError(f"legacy archive {attestation.name} changed during receipt publication")
    if _regular_file(attestation.sidecar_path, label="legacy download sidecar") is not None:
        _validate_probe(_load_sidecar(attestation.sidecar_path), attestation.probe)
        attestation.sidecar_path.unlink()
    if _regular_file(attestation.partial_path, label="legacy partial download") is not None:
        partial_bytes, partial_hash = _sha256(attestation.partial_path)
        if partial_bytes != actual_bytes or partial_hash != actual_hash:
            raise RuntimeError(
                f"legacy archive {attestation.name} conflicts with its partial state"
            )
        attestation.partial_path.unlink()
    return {
        "name": attestation.name,
        "url": ARCHIVES[attestation.name],
        "path": str(attestation.archive),
        "bytes": actual_bytes,
        "sha256": actual_hash,
        "resumed_from_bytes": attestation.resumed_from_bytes,
        "etag": attestation.probe.get("etag"),
        "last_modified": attestation.probe.get("last_modified"),
        "reused_completed_archive": True,
        "validator_receipt": attestation.receipt_path.name,
        "legacy_attested": True,
    }


def migrate_legacy_download_manifest(
    legacy_manifest_path: Path,
    output_manifest_path: Path,
    *,
    archive_root: Path,
    accept_license: bool,
) -> Path:
    """Attest a pre-receipt completed download without modifying its source manifest."""

    if not accept_license:
        raise ValueError("pass --accept-license after reviewing current OpenSLR terms")
    source_input = legacy_manifest_path.expanduser().absolute()
    output_input = output_manifest_path.expanduser().absolute()
    root_input = archive_root.expanduser().absolute()
    if _regular_file(source_input, label="legacy download manifest") is None:
        raise FileNotFoundError(f"legacy download manifest does not exist: {source_input}")
    if _directory(root_input, label="legacy archive root") is None:
        raise FileNotFoundError(f"legacy archive root does not exist: {root_input}")
    source = source_input.resolve(strict=True)
    output = output_input.resolve(strict=False)
    root = root_input.resolve(strict=True)
    if _lstat(output, label="migrated download manifest") is not None:
        raise FileExistsError(f"refusing to overwrite migrated download manifest: {output}")

    legacy, records, source_bytes, source_hash = _load_legacy_download_manifest(source)
    attestations: list[_LegacyArchiveAttestation] = []
    for name in ARCHIVES:
        archive = root / f"{name}.tgz"
        if archive.parent != root:
            raise RuntimeError(f"legacy archive path escapes archive root: {archive}")
        if _regular_file(archive, label=f"legacy {name} archive") is None:
            raise FileNotFoundError(f"legacy archive does not exist: {archive}")
        attestations.append(
            _attest_legacy_archive(
                name,
                archive,
                records[name],
                source_manifest_sha256=source_hash,
            )
        )

    source_extracted = legacy["extracted"]
    raw_audio_retained = legacy["raw_audio_retained"]
    assert isinstance(source_extracted, bool)
    assert isinstance(raw_audio_retained, bool)
    migrated_records = [
        {
            "name": attestation.name,
            "url": ARCHIVES[attestation.name],
            "path": str(attestation.archive),
            "bytes": attestation.archive_bytes,
            "sha256": attestation.archive_sha256,
            "resumed_from_bytes": attestation.resumed_from_bytes,
            "etag": attestation.probe.get("etag"),
            "last_modified": attestation.probe.get("last_modified"),
            "reused_completed_archive": True,
            "validator_receipt": attestation.receipt_path.name,
            "legacy_attested": True,
        }
        for attestation in attestations
    ]
    migrated: dict[str, object] = {
        "schema_version": "echoforge.download/v1",
        "dataset": "AISHELL-1",
        "source": "OpenSLR 33",
        "license_url": LICENSE_URL,
        "license_declared": legacy.get("license_declared"),
        "license_text_sha256": legacy["license_text_sha256"],
        "archives": migrated_records,
        "dry_run": False,
        "raw_audio_retained": raw_audio_retained,
        "extracted": False,
        "migration": {
            "schema_version": DOWNLOAD_MIGRATION_SCHEMA,
            "source_manifest_sha256": source_hash,
            "source_manifest_bytes": source_bytes,
            "source_dry_run_field": "absent",
            "source_extracted": source_extracted,
            "policy": "legacy-manifest + current HEAD validators + stable local size/SHA-256",
        },
    }
    output_temporary = output.with_name(output.name + ".tmp")
    if _regular_file(output_temporary, label="temporary migrated manifest") is not None:
        temporary_manifest = _read_stable_json(
            output_temporary, label="temporary migrated manifest"
        )
        if temporary_manifest != migrated:
            raise RuntimeError("temporary migrated manifest conflicts with the requested migration")
    if _sha256(source) != (source_bytes, source_hash):
        raise RuntimeError("legacy download manifest changed before receipt publication")
    published_records = [
        _publish_legacy_archive_attestation(attestation) for attestation in attestations
    ]
    if published_records != migrated_records:
        raise RuntimeError("published legacy archive evidence changed unexpectedly")
    if _sha256(source) != (source_bytes, source_hash):
        raise RuntimeError("legacy download manifest changed before manifest publication")
    _write_json_new(output, migrated)
    return output


def _remote_sha256(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=60) as response:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: Path, target: Path) -> None:
    if _regular_file(archive, label="archive") is None:
        raise FileNotFoundError(f"archive does not exist: {archive}")
    target_metadata = _directory(target, label="extraction target")
    target = target.resolve()
    if target_metadata is not None and any(target.iterdir()):
        raise FileExistsError(f"refusing to extract into a non-empty directory: {target}")
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        target.mkdir(parents=True, exist_ok=True)
        for member in members:
            candidate = (target / member.name).resolve()
            if target not in candidate.parents and candidate != target:
                raise ValueError(f"archive member escapes extraction directory: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"archive links are not allowed: {member.name}")
        for member in members:
            tar.extract(member, target)


def _extraction_payload(
    name: str,
    url: str,
    archive: Path,
    download: dict[str, object],
) -> dict[str, object]:
    digest = download.get("sha256")
    byte_count = download.get("bytes")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"download record for {name} has no archive hash")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
        raise RuntimeError(f"download record for {name} has no positive byte count")
    return {
        "schema_version": "echoforge.extraction/v1",
        "archive_name": name,
        "archive_url": url,
        "archive_file": archive.name,
        "archive_bytes": byte_count,
        "archive_sha256": digest,
        "complete": True,
    }


def _verify_archive_record(
    name: str, archive: Path, download: dict[str, object]
) -> tuple[int, str]:
    expected_bytes = _positive_int(download, "bytes", label=f"download record for {name}")
    expected_hash = download.get("sha256")
    if not _valid_sha256(expected_hash):
        raise RuntimeError(f"download record for {name} has no canonical archive SHA-256")
    actual_bytes, actual_hash = _sha256(archive)
    if actual_bytes != expected_bytes or actual_hash != str(expected_hash).lower():
        raise RuntimeError(f"archive evidence mismatch before/after extraction: {archive}")
    return actual_bytes, actual_hash


def _validate_extraction_marker(path: Path, expected: dict[str, object]) -> None:
    marker = _load_sidecar(path)
    for field, value in expected.items():
        if marker.get(field) != value:
            raise RuntimeError(f"extraction marker does not match {field}: {path}")


def _remove_incomplete_staging(staging: Path, *, output: Path) -> None:
    metadata = _directory(staging, label="extraction staging directory")
    if metadata is None:
        return
    resolved_output = output.resolve()
    resolved_staging = staging.resolve()
    if resolved_staging.parent != resolved_output or not staging.name.startswith("."):
        raise RuntimeError(f"refusing to remove unexpected staging directory: {staging}")
    shutil.rmtree(staging)


def _extract_or_reuse(
    name: str,
    url: str,
    archive: Path,
    target: Path,
    download: dict[str, object],
) -> dict[str, object]:
    """Recover or create an extraction with an in-directory completion marker."""

    raw_target = target
    target_metadata = _directory(raw_target, label="extraction target")
    output = raw_target.parent.resolve()
    target = raw_target.resolve()
    if target.parent != output:
        raise RuntimeError(f"extraction target escapes output directory: {target}")
    staging = output / f".{name}.extracting"
    _verify_archive_record(name, archive, download)
    expected = _extraction_payload(name, url, archive, download)
    staging_metadata = _directory(staging, label="extraction staging directory")

    if target_metadata is not None:
        _validate_extraction_marker(target / EXTRACTION_MARKER, expected)
        if staging_metadata is not None:
            _remove_incomplete_staging(staging, output=output)
        return {"extracted_to": str(target), "extraction_reused": True}

    if staging_metadata is not None:
        marker = staging / EXTRACTION_MARKER
        if _regular_file(marker, label="extraction marker") is not None:
            _validate_extraction_marker(marker, expected)
            os.replace(staging, target)
            return {"extracted_to": str(target), "extraction_reused": True}
        _remove_incomplete_staging(staging, output=output)

    _safe_extract(archive, staging)
    _verify_archive_record(name, archive, download)
    _write_json_new(staging / EXTRACTION_MARKER, expected)
    os.replace(staging, target)
    return {"extracted_to": str(target), "extraction_reused": False}


def run(output: Path, *, accept_license: bool, dry_run: bool, extract: bool) -> Path:
    if not accept_license:
        raise ValueError("pass --accept-license after reviewing current OpenSLR terms")
    if extract:
        raise ValueError(
            "download-time extraction is disabled; use scripts/extract_aishell.py "
            "for the nested AISHELL layout"
        )
    output = _safe_output(output)
    manifest_path = output / ("download_plan.json" if dry_run else "download_manifest.json")
    if _lstat(manifest_path, label="download manifest") is not None:
        raise FileExistsError(f"refusing to overwrite download manifest: {manifest_path}")
    manifest: dict[str, object] = {
        "schema_version": "echoforge.download/v1",
        "dataset": "AISHELL-1",
        "source": "OpenSLR 33",
        "license_url": LICENSE_URL,
        "license_declared": "Apache-2.0; verify upstream before redistribution",
        "license_text_sha256": None,
        "archives": [],
        "dry_run": dry_run,
        "raw_audio_retained": True,
        "extracted": False,
    }
    if not dry_run:
        manifest["license_text_sha256"] = _remote_sha256(LICENSE_URL)
        records: list[dict[str, object]] = []
        for name, url in ARCHIVES.items():
            destination = output / f"{name}.tgz"
            destination_metadata = _regular_file(destination, label=f"{name} archive")
            download = (
                _verify_completed_download(url, destination)
                if destination_metadata is not None
                else _download(url, destination)
            )
            record = {"name": name, "url": url, "path": str(destination), **download}
            records.append(record)
        manifest["archives"] = records
    _write_json_new(manifest_path, manifest)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".cache/aishell1"))
    parser.add_argument("--accept-license", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument(
        "--migrate-legacy-manifest",
        type=Path,
        metavar="PATH",
        help="attest a completed pre-receipt download manifest instead of downloading",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        metavar="DIR",
        help="directory containing fixed audio.tgz and resources.tgz migration inputs",
    )
    parser.add_argument(
        "--migration-output",
        type=Path,
        metavar="PATH",
        help="new no-overwrite completed manifest created by legacy migration",
    )
    args = parser.parse_args()
    if args.extract:
        parser.error("--extract is disabled; use scripts/extract_aishell.py after download")
    migration_arguments = (
        args.migrate_legacy_manifest,
        args.archive_root,
        args.migration_output,
    )
    if any(value is not None for value in migration_arguments):
        if not all(value is not None for value in migration_arguments):
            parser.error(
                "legacy migration requires --migrate-legacy-manifest, "
                "--archive-root, and --migration-output"
            )
        if args.dry_run or args.extract:
            parser.error("legacy migration cannot be combined with --dry-run or --extract")
        assert args.migrate_legacy_manifest is not None
        assert args.archive_root is not None
        assert args.migration_output is not None
        try:
            migrated = migrate_legacy_download_manifest(
                args.migrate_legacy_manifest,
                args.migration_output,
                archive_root=args.archive_root,
                accept_license=args.accept_license,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"AISHELL download migration failed: {exc}", file=sys.stderr)
            return 1
        print(migrated)
        return 0
    try:
        manifest = run(
            args.output,
            accept_license=args.accept_license,
            dry_run=args.dry_run,
            extract=args.extract,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"AISHELL download failed: {exc}", file=sys.stderr)
        return 1
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
