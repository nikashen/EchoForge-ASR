from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts import download_aishell


def _legacy_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, dict[str, object]], dict[str, object]]:
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    payloads = {"audio": b"legacy-audio", "resources": b"legacy-resources"}
    validators = {
        "audio": ('"audio-v1"', "Sat, 15 Aug 2026 01:00:00 GMT"),
        "resources": ('"resources-v1"', "Sat, 15 Aug 2026 02:00:00 GMT"),
    }
    probes: dict[str, dict[str, object]] = {}
    records: list[dict[str, object]] = []
    for name, url in download_aishell.ARCHIVES.items():
        archive = archive_root / f"{name}.tgz"
        archive.write_bytes(payloads[name])
        etag, last_modified = validators[name]
        probe: dict[str, object] = {
            "url": url,
            "expected_bytes": len(payloads[name]),
            "etag": etag,
            "last_modified": last_modified,
            "accept_ranges": "bytes",
        }
        probes[url] = probe
        records.append(
            {
                "name": name,
                "url": url,
                # Migration intentionally validates but does not trust this old path.
                "path": str(tmp_path / "untrusted-old-location" / f"{name}.tgz"),
                "bytes": len(payloads[name]),
                "sha256": hashlib.sha256(payloads[name]).hexdigest(),
                "resumed_from_bytes": 3 if name == "audio" else 0,
                "etag": etag,
                "last_modified": last_modified,
            }
        )
    legacy: dict[str, object] = {
        "schema_version": "echoforge.download/v1",
        "dataset": "AISHELL-1",
        "source": "OpenSLR 33",
        "license_url": download_aishell.LICENSE_URL,
        "license_declared": "Apache-2.0; verify upstream before redistribution",
        "license_text_sha256": "a" * 64,
        "archives": records,
        "raw_audio_retained": True,
        "extracted": False,
    }
    manifest = tmp_path / "legacy-download-manifest.json"
    manifest.write_text(json.dumps(legacy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest, archive_root, probes, legacy


def _record(legacy: dict[str, object], name: str) -> dict[str, object]:
    archives = legacy["archives"]
    assert isinstance(archives, list)
    for value in archives:
        assert isinstance(value, dict)
        if value.get("name") == name:
            return value
    raise AssertionError(f"missing fixture record: {name}")


def _rewrite_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_audio_sidecar(archive_root: Path, probes: dict[str, dict[str, object]]) -> Path:
    sidecar = archive_root / "audio.tgz.part.json"
    sidecar.write_text(json.dumps(probes[download_aishell.ARCHIVES["audio"]]), encoding="utf-8")
    return sidecar


def test_legacy_migration_attests_both_archives_without_mutating_source(
    tmp_path: Path, monkeypatch
) -> None:
    source, archive_root, probes, _legacy = _legacy_fixture(tmp_path)
    source_before = source.read_bytes()
    source_hash = hashlib.sha256(source_before).hexdigest()
    audio_sidecar = _write_audio_sidecar(archive_root, probes)
    monkeypatch.setattr(
        download_aishell,
        "_probe_download",
        lambda url: dict(probes[url]),
    )

    # A completed archive without sidecar/receipt remains rejected by the normal path.
    with pytest.raises(RuntimeError, match="no persistent validator"):
        download_aishell._verify_completed_download(
            download_aishell.ARCHIVES["resources"], archive_root / "resources.tgz"
        )

    output = tmp_path / "migrated-download-manifest.json"
    result = download_aishell.migrate_legacy_download_manifest(
        source,
        output,
        archive_root=archive_root,
        accept_license=True,
    )

    assert result == output.resolve()
    assert source.read_bytes() == source_before
    assert not audio_sidecar.exists()
    migrated = json.loads(output.read_text(encoding="utf-8"))
    assert migrated["dry_run"] is False
    assert migrated["extracted"] is False
    assert migrated["migration"]["source_manifest_sha256"] == source_hash
    assert migrated["migration"]["source_manifest_bytes"] == len(source_before)
    assert [record["name"] for record in migrated["archives"]] == ["audio", "resources"]
    for record in migrated["archives"]:
        name = record["name"]
        assert Path(record["path"]) == archive_root / f"{name}.tgz"
        receipt_path = archive_root / record["validator_receipt"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["dry_run"] is False
        assert receipt["migration_schema_version"] == download_aishell.DOWNLOAD_MIGRATION_SCHEMA
        assert receipt["migration_source_manifest_sha256"] == source_hash


def test_legacy_migration_preflights_all_archives_before_writing_receipts(
    tmp_path: Path, monkeypatch
) -> None:
    source, archive_root, probes, _legacy = _legacy_fixture(tmp_path)
    _write_audio_sidecar(archive_root, probes)
    changed = {url: dict(probe) for url, probe in probes.items()}
    changed[download_aishell.ARCHIVES["resources"]]["etag"] = '"same-size-v2"'
    monkeypatch.setattr(download_aishell, "_probe_download", lambda url: changed[url])
    output = tmp_path / "migrated.json"

    with pytest.raises(RuntimeError, match="validator changed"):
        download_aishell.migrate_legacy_download_manifest(
            source,
            output,
            archive_root=archive_root,
            accept_license=True,
        )

    assert not output.exists()
    assert (archive_root / "audio.tgz.part.json").is_file()
    assert not list(archive_root.glob("*.download.json"))


def test_legacy_migration_rejects_local_hash_mismatch_before_writing(
    tmp_path: Path, monkeypatch
) -> None:
    source, archive_root, probes, _legacy = _legacy_fixture(tmp_path)
    resources = archive_root / "resources.tgz"
    resources.write_bytes(b"X" + resources.read_bytes()[1:])
    monkeypatch.setattr(download_aishell, "_probe_download", lambda url: probes[url])
    output = tmp_path / "migrated.json"

    with pytest.raises(RuntimeError, match="recorded size/SHA-256"):
        download_aishell.migrate_legacy_download_manifest(
            source,
            output,
            archive_root=archive_root,
            accept_license=True,
        )

    assert not output.exists()
    assert not list(archive_root.glob("*.download.json"))


@pytest.mark.parametrize(
    "missing_field",
    ["url", "path", "bytes", "sha256", "etag", "last_modified"],
)
def test_legacy_migration_requires_complete_archive_records(
    tmp_path: Path, monkeypatch, missing_field: str
) -> None:
    source, archive_root, _probes, legacy = _legacy_fixture(tmp_path)
    _record(legacy, "audio").pop(missing_field)
    _rewrite_manifest(source, legacy)
    monkeypatch.setattr(
        download_aishell,
        "_probe_download",
        lambda _url: pytest.fail("invalid manifest must fail before network validation"),
    )

    with pytest.raises(RuntimeError, match="legacy archive audio"):
        download_aishell.migrate_legacy_download_manifest(
            source,
            tmp_path / "migrated.json",
            archive_root=archive_root,
            accept_license=True,
        )


@pytest.mark.parametrize("invalid_value", [True, -1, "3", 10_000])
def test_legacy_migration_rejects_invalid_resume_offsets(
    tmp_path: Path, monkeypatch, invalid_value: object
) -> None:
    source, archive_root, _probes, legacy = _legacy_fixture(tmp_path)
    _record(legacy, "audio")["resumed_from_bytes"] = invalid_value
    _rewrite_manifest(source, legacy)
    monkeypatch.setattr(
        download_aishell,
        "_probe_download",
        lambda _url: pytest.fail("invalid manifest must fail before network validation"),
    )

    with pytest.raises(RuntimeError, match="invalid resumed_from_bytes"):
        download_aishell.migrate_legacy_download_manifest(
            source,
            tmp_path / "migrated.json",
            archive_root=archive_root,
            accept_license=True,
        )


def test_legacy_migration_rejects_non_historical_or_ill_typed_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    source, archive_root, _probes, legacy = _legacy_fixture(tmp_path)
    legacy["dry_run"] = False
    _rewrite_manifest(source, legacy)
    monkeypatch.setattr(
        download_aishell,
        "_probe_download",
        lambda _url: pytest.fail("invalid manifest must fail before network validation"),
    )
    with pytest.raises(RuntimeError, match="historical manifest with no dry_run"):
        download_aishell.migrate_legacy_download_manifest(
            source,
            tmp_path / "migrated.json",
            archive_root=archive_root,
            accept_license=True,
        )

    legacy.pop("dry_run")
    legacy["raw_audio_retained"] = "yes"
    _rewrite_manifest(source, legacy)
    with pytest.raises(TypeError, match="raw_audio_retained"):
        download_aishell.migrate_legacy_download_manifest(
            source,
            tmp_path / "migrated.json",
            archive_root=archive_root,
            accept_license=True,
        )


def test_legacy_migration_rejects_receipt_from_another_source_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    source, archive_root, probes, _legacy = _legacy_fixture(tmp_path)
    audio = archive_root / "audio.tgz"
    audio_bytes = audio.read_bytes()
    probe = probes[download_aishell.ARCHIVES["audio"]]
    receipt = download_aishell._receipt_payload(
        probe,
        archive_bytes=len(audio_bytes),
        archive_sha256=hashlib.sha256(audio_bytes).hexdigest(),
    )
    receipt.update(
        {
            "migration_schema_version": download_aishell.DOWNLOAD_MIGRATION_SCHEMA,
            "migration_source_manifest_sha256": "0" * 64,
        }
    )
    (archive_root / "audio.tgz.download.json").write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(download_aishell, "_probe_download", lambda url: probes[url])

    with pytest.raises(RuntimeError, match="not bound to this migration source"):
        download_aishell.migrate_legacy_download_manifest(
            source,
            tmp_path / "migrated.json",
            archive_root=archive_root,
            accept_license=True,
        )

    assert not (archive_root / "resources.tgz.download.json").exists()


def test_legacy_migration_never_overwrites_output(tmp_path: Path, monkeypatch) -> None:
    source, archive_root, _probes, _legacy = _legacy_fixture(tmp_path)
    output = tmp_path / "migrated.json"
    output.write_text("sentinel", encoding="utf-8")
    monkeypatch.setattr(
        download_aishell,
        "_probe_download",
        lambda _url: pytest.fail("existing output must fail before network validation"),
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        download_aishell.migrate_legacy_download_manifest(
            source,
            output,
            archive_root=archive_root,
            accept_license=True,
        )
    assert output.read_text(encoding="utf-8") == "sentinel"


def test_legacy_migration_cli_routes_explicit_paths(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "legacy.json"
    archive_root = tmp_path / "archives"
    output = tmp_path / "migrated.json"
    received: dict[str, object] = {}

    def migrate_fixture(
        source_path: Path,
        output_path: Path,
        *,
        archive_root: Path,
        accept_license: bool,
    ) -> Path:
        received.update(
            {
                "source": source_path,
                "output": output_path,
                "archive_root": archive_root,
                "accept_license": accept_license,
            }
        )
        return output_path

    monkeypatch.setattr(download_aishell, "migrate_legacy_download_manifest", migrate_fixture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_aishell.py",
            "--accept-license",
            "--migrate-legacy-manifest",
            str(source),
            "--archive-root",
            str(archive_root),
            "--migration-output",
            str(output),
        ],
    )

    assert download_aishell.main() == 0
    assert received == {
        "source": source,
        "output": output,
        "archive_root": archive_root,
        "accept_license": True,
    }


@pytest.mark.parametrize(
    "arguments",
    [
        ["--migrate-legacy-manifest", "legacy.json"],
        [
            "--migrate-legacy-manifest",
            "legacy.json",
            "--archive-root",
            "archives",
            "--migration-output",
            "migrated.json",
            "--dry-run",
        ],
    ],
)
def test_legacy_migration_cli_rejects_incomplete_or_conflicting_modes(
    monkeypatch, arguments: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["download_aishell.py", *arguments])
    with pytest.raises(SystemExit) as exc_info:
        download_aishell.main()
    assert exc_info.value.code == 2
