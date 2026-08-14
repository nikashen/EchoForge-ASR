"""Opt-in AISHELL-1 downloader with hash and extraction safety checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ARCHIVES = {
    "audio": "https://www.openslr.org/resources/33/data_aishell.tgz",
    "resources": "https://www.openslr.org/resources/33/resource_aishell.tgz",
}
LICENSE_URL = "https://www.openslr.org/33/"


def _safe_output(root: Path) -> Path:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "EchoForge-ASR/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as handle:
        total = 0
        while block := response.read(1024 * 1024):
            total += len(block)
            handle.write(block)
    return total, _sha256(destination)


def _remote_sha256(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "EchoForge-ASR/0.1"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=60) as response:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: Path, target: Path) -> None:
    target = target.resolve()
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


def run(output: Path, *, accept_license: bool, dry_run: bool, extract: bool) -> Path:
    if not accept_license:
        raise ValueError("pass --accept-license after reviewing current OpenSLR terms")
    output = _safe_output(output)
    manifest: dict[str, object] = {
        "schema_version": "echoforge.download/v1",
        "dataset": "AISHELL-1",
        "source": "OpenSLR 33",
        "license_url": LICENSE_URL,
        "license_declared": "Apache-2.0; verify upstream before redistribution",
        "license_text_sha256": None,
        "archives": [],
        "raw_audio_retained": True,
        "extracted": False,
    }
    if dry_run:
        manifest["dry_run"] = True
    else:
        manifest["license_text_sha256"] = _remote_sha256(LICENSE_URL)
        records: list[dict[str, object]] = []
        for name, url in ARCHIVES.items():
            destination = output / f"{name}.tgz"
            bytes_written, digest = _download(url, destination)
            records.append(
                {
                    "name": name,
                    "url": url,
                    "path": str(destination),
                    "bytes": bytes_written,
                    "sha256": digest,
                }
            )
            if extract:
                _safe_extract(destination, output / name)
        manifest["archives"] = records
        manifest["extracted"] = extract
    manifest_path = output / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".cache/aishell1"))
    parser.add_argument("--accept-license", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    print(
        run(
            args.output,
            accept_license=args.accept_license,
            dry_run=args.dry_run,
            extract=args.extract,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
