from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import pytest

from echoforge.evaluation.aishell import prepare_aishell_manifest
from echoforge.evaluation.runner import ManifestRunnerError

AUDIO_URL = "https://www.openslr.org/resources/33/data_aishell.tgz"
RESOURCES_URL = "https://www.openslr.org/resources/33/resource_aishell.tgz"


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(b"\x00\x00" * 1600)


def _download_payload() -> dict[str, Any]:
    return {
        "schema_version": "echoforge.download/v1",
        "dataset": "AISHELL-1",
        "license_declared": "Apache-2.0",
        "license_text_sha256": "c" * 64,
        "dry_run": False,
        "archives": [
            {
                "name": "audio",
                "url": AUDIO_URL,
                "bytes": 15_582_913_665,
                "sha256": "a" * 64,
            },
            {
                "name": "resources",
                "url": RESOURCES_URL,
                "bytes": 1_043_765,
                "sha256": "b" * 64,
            },
        ],
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    wav_root = tmp_path / "wav"
    _write_wav(wav_root / "dev" / "S0001" / "U0001.wav")
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("U0001 \u4f60 \u597d\n", encoding="utf-8")
    download_manifest = tmp_path / "download_manifest.json"
    download_manifest.write_text(json.dumps(_download_payload()), encoding="utf-8")
    return wav_root, transcript, download_manifest


def _prepare(
    tmp_path: Path,
    *,
    wav_root: Path,
    transcript: Path,
    download_manifest: Path,
    **kwargs: object,
) -> dict[str, Any]:
    return prepare_aishell_manifest(
        tmp_path / "prepared.json",
        wav_root=wav_root,
        transcript_path=transcript,
        download_manifest_path=download_manifest,
        **kwargs,  # type: ignore[arg-type]
    )


def test_prepare_rejects_authorization_without_verified_extraction(tmp_path: Path) -> None:
    wav_root, transcript, download_manifest = _fixture(tmp_path)

    with pytest.raises(ManifestRunnerError, match="verified extraction evidence"):
        _prepare(
            tmp_path,
            wav_root=wav_root,
            transcript=transcript,
            download_manifest=download_manifest,
            evaluation_authorized=True,
            protocol_id="aishell1-dev-v1",
        )


def test_prepare_rejects_authorized_multi_split_before_reading_data(tmp_path: Path) -> None:
    with pytest.raises(ManifestRunnerError, match="one dev or test split"):
        prepare_aishell_manifest(
            tmp_path / "prepared.json",
            wav_root=tmp_path / "missing-wav",
            transcript_path=tmp_path / "missing-transcript",
            download_manifest_path=tmp_path / "missing-download-manifest",
            splits=("dev", "test"),
            evaluation_authorized=True,
            protocol_id="aishell1-dev-test-v1",
        )


@pytest.mark.parametrize(
    "limits",
    [
        {"speaker_limit": 1},
        {"utterances_per_speaker": 1},
    ],
)
def test_prepare_rejects_authorized_subset_selection(
    tmp_path: Path, limits: dict[str, int]
) -> None:
    with pytest.raises(ManifestRunnerError, match="complete selected split"):
        prepare_aishell_manifest(
            tmp_path / "prepared.json",
            wav_root=tmp_path / "missing-wav",
            transcript_path=tmp_path / "missing-transcript",
            download_manifest_path=tmp_path / "missing-download-manifest",
            evaluation_authorized=True,
            protocol_id="aishell1-dev-v1",
            **limits,
        )


@pytest.mark.parametrize("protocol_id", ["", " leading", "two words", "x" * 129])
def test_prepare_rejects_noncanonical_protocol_id(tmp_path: Path, protocol_id: str) -> None:
    with pytest.raises(ManifestRunnerError, match="canonical identifier"):
        prepare_aishell_manifest(
            tmp_path / "prepared.json",
            wav_root=tmp_path / "missing-wav",
            transcript_path=tmp_path / "missing-transcript",
            download_manifest_path=tmp_path / "missing-download-manifest",
            protocol_id=protocol_id,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing-dry-run", "completed AISHELL-1 download"),
        ("dry-run", "completed AISHELL-1 download"),
        ("non-boolean-dry-run", "completed AISHELL-1 download"),
        ("missing", "exactly audio and resources"),
        ("extra", "exactly audio and resources"),
        ("duplicate", "one audio and one resources"),
        ("wrong-url", "unexpected OpenSLR URL"),
        ("short-sha", "invalid SHA-256"),
        ("uppercase-sha", "invalid SHA-256"),
        ("non-string-sha", "invalid SHA-256"),
        ("zero-bytes", "invalid byte count"),
        ("boolean-bytes", "invalid byte count"),
    ],
)
def test_prepare_rejects_incomplete_download_evidence(
    tmp_path: Path, case: str, message: str
) -> None:
    wav_root, transcript, download_manifest = _fixture(tmp_path)
    payload = _download_payload()
    archives = payload["archives"]
    assert isinstance(archives, list)
    if case == "missing-dry-run":
        payload.pop("dry_run")
    elif case == "dry-run":
        payload["dry_run"] = True
    elif case == "non-boolean-dry-run":
        payload["dry_run"] = "false"
    elif case == "missing":
        payload["archives"] = archives[:1]
    elif case == "extra":
        payload["archives"] = [*archives, dict(archives[0])]
    elif case == "duplicate":
        payload["archives"] = [archives[0], dict(archives[0])]
    elif case == "wrong-url":
        archives[0]["url"] = "https://example.invalid/data_aishell.tgz"
    elif case == "short-sha":
        archives[0]["sha256"] = "a" * 63
    elif case == "uppercase-sha":
        archives[0]["sha256"] = "A" * 64
    elif case == "non-string-sha":
        archives[0]["sha256"] = int("1" * 64)
    elif case == "zero-bytes":
        archives[0]["bytes"] = 0
    elif case == "boolean-bytes":
        archives[0]["bytes"] = True
    download_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestRunnerError, match=message):
        _prepare(
            tmp_path,
            wav_root=wav_root,
            transcript=transcript,
            download_manifest=download_manifest,
        )
    assert not (tmp_path / "prepared.json").exists()


def _symlink_or_skip(link: Path, target: Path, *, directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable on this platform: {exc}")


def test_prepare_rejects_linked_speaker_directory(tmp_path: Path) -> None:
    wav_root = tmp_path / "wav"
    split_root = wav_root / "dev"
    split_root.mkdir(parents=True)
    outside_speaker = tmp_path / "outside" / "S0001"
    _write_wav(outside_speaker / "U0001.wav")
    _symlink_or_skip(split_root / "S0001", outside_speaker, directory=True)
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("U0001 \u4f60\u597d\n", encoding="utf-8")
    download_manifest = tmp_path / "download_manifest.json"
    download_manifest.write_text(json.dumps(_download_payload()), encoding="utf-8")

    with pytest.raises(ManifestRunnerError, match="speaker directory.*symbolic link"):
        _prepare(
            tmp_path,
            wav_root=wav_root,
            transcript=transcript,
            download_manifest=download_manifest,
        )


def test_prepare_rejects_linked_wav_path(tmp_path: Path) -> None:
    wav_root = tmp_path / "wav"
    speaker_root = wav_root / "dev" / "S0001"
    speaker_root.mkdir(parents=True)
    outside_wav = tmp_path / "outside" / "U0001.wav"
    _write_wav(outside_wav)
    _symlink_or_skip(speaker_root / "U0001.wav", outside_wav, directory=False)
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("U0001 \u4f60\u597d\n", encoding="utf-8")
    download_manifest = tmp_path / "download_manifest.json"
    download_manifest.write_text(json.dumps(_download_payload()), encoding="utf-8")

    with pytest.raises(ManifestRunnerError, match="WAV.*symbolic link"):
        _prepare(
            tmp_path,
            wav_root=wav_root,
            transcript=transcript,
            download_manifest=download_manifest,
        )


@pytest.mark.parametrize("truncated_bytes", [1, 2, 3])
def test_prepare_rejects_truncated_wav_data(tmp_path: Path, truncated_bytes: int) -> None:
    wav_root, transcript, download_manifest = _fixture(tmp_path)
    wav_path = wav_root / "dev" / "S0001" / "U0001.wav"
    wav_path.write_bytes(wav_path.read_bytes()[:-truncated_bytes])

    with pytest.raises(ManifestRunnerError, match="truncated"):
        _prepare(
            tmp_path,
            wav_root=wav_root,
            transcript=transcript,
            download_manifest=download_manifest,
        )
