from __future__ import annotations

import hashlib
import io
import json
import tarfile
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import echoforge.evaluation.aishell_extract as extraction
from echoforge.evaluation.aishell import prepare_aishell_manifest
from echoforge.evaluation.aishell_extract import AishellExtractionError, extract_aishell
from scripts import extract_aishell as extract_cli


def _wav_bytes(*, sample: int = 0) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(int(sample).to_bytes(2, "little", signed=True) * 1600)
    return output.getvalue()


def _add_directory(archive: tarfile.TarFile, name: str) -> None:
    member = tarfile.TarInfo(name)
    member.type = tarfile.DIRTYPE
    member.mode = 0o755
    member.mtime = 0
    archive.addfile(member)


def _add_file(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(data)
    member.mode = 0o644
    member.mtime = 0
    archive.addfile(member, io.BytesIO(data))


def _add_unsafe_member(archive: tarfile.TarFile, kind: str, *, prefix: str = "") -> None:
    if kind == "traversal":
        _add_file(archive, f"{prefix}../../escaped.wav", _wav_bytes())
        return
    member = tarfile.TarInfo(f"{prefix}unsafe")
    member.mtime = 0
    if kind == "link":
        member.type = tarfile.SYMTYPE
        member.linkname = "../../escaped"
    elif kind == "device":
        member.type = tarfile.CHRTYPE
        member.devmajor = 1
        member.devminor = 3
    else:
        raise AssertionError(kind)
    archive.addfile(member)


def _tar_bytes(write: Callable[[tarfile.TarFile], None]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        write(archive)
    return output.getvalue()


def _speaker_archive(
    speaker_id: str,
    split: str,
    utterance_id: str,
    *,
    wav: bytes | None = None,
    unsafe_kind: str | None = None,
    path_override: str | None = None,
) -> bytes:
    def write(archive: tarfile.TarFile) -> None:
        _add_directory(archive, split)
        _add_directory(archive, f"{split}/{speaker_id}")
        if unsafe_kind is not None:
            _add_unsafe_member(archive, unsafe_kind, prefix=f"{split}/{speaker_id}/")
        else:
            name = path_override or f"{split}/{speaker_id}/{utterance_id}.wav"
            sample = sum(utterance_id.encode("ascii")) % 1000
            _add_file(archive, name, _wav_bytes(sample=sample) if wav is None else wav)

    return _tar_bytes(write)


def _audio_archive(
    speakers: list[tuple[str, bytes]] | None = None,
    *,
    unsafe_kind: str | None = None,
) -> bytes:
    speaker_values = speakers or [
        ("S0002", _speaker_archive("S0002", "dev", "D0002")),
        ("S0003", _speaker_archive("S0003", "dev", "D0003")),
        ("S0004", _speaker_archive("S0004", "test", "E0004")),
        ("S0005", _speaker_archive("S0005", "test", "E0005")),
        # The official outer archive groups splits rather than sorting globally:
        # dev/test speaker archives precede lower-numbered train speakers.
        ("S0001", _speaker_archive("S0001", "train", "T0001")),
    ]

    def write(archive: tarfile.TarFile) -> None:
        _add_directory(archive, "data_aishell")
        _add_directory(archive, "data_aishell/wav")
        for speaker_id, nested in speaker_values:
            _add_file(archive, f"data_aishell/wav/{speaker_id}.tar.gz", nested)
        _add_directory(archive, "data_aishell/transcript")
        transcript = (
            "T0001 训 练\nD0002 开 发 二\nD0003 开 发 三\nE0004 测 试 四\nE0005 测 试 五\n"
        ).encode()
        _add_file(
            archive,
            "data_aishell/transcript/aishell_transcript_v0.8.txt",
            transcript,
        )
        if unsafe_kind is not None:
            _add_unsafe_member(archive, unsafe_kind)

    return _tar_bytes(write)


def _resources_archive(*, unsafe_kind: str | None = None) -> bytes:
    def write(archive: tarfile.TarFile) -> None:
        _add_directory(archive, "resource_aishell")
        _add_file(archive, "resource_aishell/lexicon.txt", "你 ni3\n".encode())
        _add_file(archive, "resource_aishell/speaker.info", b"S0002 F\n")
        if unsafe_kind is not None:
            _add_unsafe_member(archive, unsafe_kind)

    return _tar_bytes(write)


def _fixture(
    tmp_path: Path,
    *,
    audio_bytes: bytes | None = None,
    resources_bytes: bytes | None = None,
    mutate_manifest: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    audio = tmp_path / "audio.tgz"
    resources = tmp_path / "resources.tgz"
    audio.write_bytes(_audio_archive() if audio_bytes is None else audio_bytes)
    resources.write_bytes(_resources_archive() if resources_bytes is None else resources_bytes)
    payload: dict[str, Any] = {
        "schema_version": "echoforge.download/v1",
        "dataset": "AISHELL-1",
        "license_declared": "Apache-2.0",
        "license_text_sha256": "c" * 64,
        "dry_run": False,
        "archives": [
            {
                "name": "audio",
                "url": extraction.AUDIO_URL,
                "bytes": audio.stat().st_size,
                "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
            },
            {
                "name": "resources",
                "url": extraction.RESOURCES_URL,
                "bytes": resources.stat().st_size,
                "sha256": hashlib.sha256(resources.read_bytes()).hexdigest(),
            },
        ],
    }
    if mutate_manifest is not None:
        mutate_manifest(payload)
    manifest = tmp_path / "download_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, audio, resources


def _extract(
    tmp_path: Path,
    *,
    output: Path | None = None,
    splits: tuple[str, ...] = ("dev", "test"),
    speaker_limit_per_split: int | None = None,
    audio_bytes: bytes | None = None,
    resources_bytes: bytes | None = None,
    mutate_manifest: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, object]:
    manifest, audio, resources = _fixture(
        tmp_path,
        audio_bytes=audio_bytes,
        resources_bytes=resources_bytes,
        mutate_manifest=mutate_manifest,
    )
    return extract_aishell(
        output or tmp_path / "extracted",
        download_manifest_path=manifest,
        audio_archive_path=audio,
        resources_archive_path=resources,
        splits=splits,
        speaker_limit_per_split=speaker_limit_per_split,
    )


def test_extracts_default_dev_test_with_deterministic_speaker_limit_and_reuses(
    tmp_path: Path,
) -> None:
    manifest, audio, resources = _fixture(tmp_path)
    output = tmp_path / "extracted"

    first = extract_aishell(
        output,
        download_manifest_path=manifest,
        audio_archive_path=audio,
        resources_archive_path=resources,
        speaker_limit_per_split=1,
    )

    assert first["reused"] is False
    tree = first["tree"]
    assert isinstance(tree, dict)
    assert tree["selected_speakers"] == {"dev": ["S0002"], "test": ["S0004"]}
    assert (output / "wav" / "dev" / "S0002" / "D0002.wav").is_file()
    assert (output / "wav" / "test" / "S0004" / "E0004.wav").is_file()
    assert not (output / "wav" / "dev" / "S0003").exists()
    assert not (output / "wav" / "train").exists()
    assert (output / "transcript" / extraction.TRANSCRIPT_NAME).is_file()
    assert (output / "resources" / "lexicon.txt").is_file()
    marker = json.loads((output / extraction.MARKER_NAME).read_text(encoding="utf-8"))
    assert marker["splits"] == ["dev", "test"]
    assert marker["speaker_limit_per_split"] == 1
    assert marker["tree"]["selected_speakers"] == tree["selected_speakers"]

    second = extract_aishell(
        output,
        download_manifest_path=manifest,
        audio_archive_path=audio,
        resources_archive_path=resources,
        speaker_limit_per_split=1,
    )
    assert second["reused"] is True
    assert second["tree"] == first["tree"]


def test_extracts_only_requested_split(tmp_path: Path) -> None:
    result = _extract(tmp_path, splits=("test",), speaker_limit_per_split=1)

    assert result["splits"] == ["test"]
    assert (tmp_path / "extracted" / "wav" / "test" / "S0004").is_dir()
    assert not (tmp_path / "extracted" / "wav" / "dev").exists()


def test_verified_full_split_can_feed_an_authorized_prepared_manifest(tmp_path: Path) -> None:
    manifest, audio, resources = _fixture(tmp_path)
    extracted = tmp_path / "extracted"
    extract_aishell(
        extracted,
        download_manifest_path=manifest,
        audio_archive_path=audio,
        resources_archive_path=resources,
        splits=("dev",),
    )

    prepared = prepare_aishell_manifest(
        tmp_path / "prepared.json",
        wav_root=extracted / "wav",
        transcript_path=extracted / "transcript" / extraction.TRANSCRIPT_NAME,
        download_manifest_path=manifest,
        splits=("dev",),
        evaluation_authorized=True,
        protocol_id="aishell1-dev-full-v1",
        extraction_root=extracted,
    )

    assert prepared["evaluation_authorized"] is True
    assert prepared["protocol_id"] == "aishell1-dev-full-v1"
    assert prepared["dataset"]["selection"]["selected_speakers"] == {"dev": ["S0002", "S0003"]}
    assert len(prepared["rows"]) == 2
    assert len(prepared["dataset"]["extraction_marker_sha256"]) == 64
    assert len(prepared["dataset"]["extraction_inventory_sha256"]) == 64


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_rejects_invalid_speaker_limit_before_reading_inputs(tmp_path: Path, limit: object) -> None:
    with pytest.raises(AishellExtractionError, match="positive integer"):
        extract_aishell(
            tmp_path / "output",
            download_manifest_path=tmp_path / "missing-manifest",
            audio_archive_path=tmp_path / "missing-audio",
            resources_archive_path=tmp_path / "missing-resources",
            speaker_limit_per_split=limit,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("dry_run"), "completed download"),
        (
            lambda payload: payload.__setitem__("license_declared", "unknown"),
            "Apache-2.0",
        ),
        (
            lambda payload: payload["archives"][0].__setitem__(
                "url", "https://example.invalid/audio.tgz"
            ),
            "unexpected URL",
        ),
        (lambda payload: payload["archives"][0].__setitem__("bytes", 1), "byte count"),
        (
            lambda payload: payload["archives"][0].__setitem__("sha256", "0" * 64),
            "SHA-256",
        ),
    ],
)
def test_verifies_manifest_url_bytes_and_hash_before_creating_output(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    with pytest.raises(AishellExtractionError, match=message):
        _extract(tmp_path, mutate_manifest=mutation)
    assert not (tmp_path / "extracted").exists()


@pytest.mark.parametrize("kind", ["traversal", "link", "device"])
def test_rejects_unsafe_outer_audio_members(tmp_path: Path, kind: str) -> None:
    with pytest.raises(AishellExtractionError, match="traversal|links|devices"):
        _extract(tmp_path, audio_bytes=_audio_archive(unsafe_kind=kind))
    assert not (tmp_path / "extracted").exists()


@pytest.mark.parametrize("kind", ["traversal", "link", "device"])
def test_rejects_unsafe_nested_speaker_members(tmp_path: Path, kind: str) -> None:
    nested = _speaker_archive("S0002", "dev", "D0002", unsafe_kind=kind)
    audio = _audio_archive(speakers=[("S0002", nested)])

    with pytest.raises(AishellExtractionError, match="traversal|links|devices"):
        _extract(tmp_path, audio_bytes=audio, splits=("dev",))


@pytest.mark.parametrize("kind", ["traversal", "link", "device"])
def test_rejects_unsafe_resources_members(tmp_path: Path, kind: str) -> None:
    with pytest.raises(AishellExtractionError, match="traversal|links|devices"):
        _extract(tmp_path, resources_bytes=_resources_archive(unsafe_kind=kind))


def test_rejects_inner_speaker_mismatch_and_invalid_wav(tmp_path: Path) -> None:
    mismatched = _speaker_archive(
        "S0002",
        "dev",
        "D0002",
        path_override="dev/S9999/D0002.wav",
    )
    with pytest.raises(AishellExtractionError, match="split/speaker/WAV path"):
        _extract(
            tmp_path / "mismatch",
            audio_bytes=_audio_archive(speakers=[("S0002", mismatched)]),
            splits=("dev",),
        )

    corrupt = _speaker_archive("S0002", "dev", "D0002", wav=b"not a wav")
    with pytest.raises(AishellExtractionError, match="invalid AISHELL WAV"):
        _extract(
            tmp_path / "corrupt",
            audio_bytes=_audio_archive(speakers=[("S0002", corrupt)]),
            splits=("dev",),
        )

    truncated = _speaker_archive("S0002", "dev", "D0002", wav=_wav_bytes()[:-10])
    with pytest.raises(AishellExtractionError, match="truncated PCM data"):
        _extract(
            tmp_path / "truncated",
            audio_bytes=_audio_archive(speakers=[("S0002", truncated)]),
            splits=("dev",),
        )


def test_rejects_unsorted_outer_speaker_archives(tmp_path: Path) -> None:
    audio = _audio_archive(
        speakers=[
            ("S0003", _speaker_archive("S0003", "dev", "D0003")),
            ("S0002", _speaker_archive("S0002", "dev", "D0002")),
        ]
    )
    with pytest.raises(AishellExtractionError, match="strictly ordered within each split"):
        _extract(tmp_path, audio_bytes=audio, splits=("dev",))


def test_never_overwrites_an_existing_untrusted_output(tmp_path: Path) -> None:
    output = tmp_path / "extracted"
    output.mkdir()
    sentinel = output / "owned-by-user.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(AishellExtractionError, match="refusing to overwrite or trust"):
        _extract(tmp_path, output=output)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_rebuilds_incomplete_controlled_staging(tmp_path: Path) -> None:
    output = tmp_path / "extracted"
    staging = tmp_path / ".extracted.echoforge-extracting"
    staging.mkdir()
    (staging / "partial.bin").write_bytes(b"partial")

    result = _extract(tmp_path, output=output, speaker_limit_per_split=1)

    assert result["reused"] is False
    assert output.is_dir()
    assert not staging.exists()
    assert not (output / "partial.bin").exists()


def test_promotes_a_complete_interrupted_staging_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, audio, resources = _fixture(tmp_path)
    output = tmp_path / "extracted"
    real_promote = extraction._promote_staging

    def interrupt(_staging: Path, _target: Path) -> None:
        raise AishellExtractionError("simulated interruption")

    monkeypatch.setattr(extraction, "_promote_staging", interrupt)
    with pytest.raises(AishellExtractionError, match="simulated interruption"):
        extract_aishell(
            output,
            download_manifest_path=manifest,
            audio_archive_path=audio,
            resources_archive_path=resources,
            speaker_limit_per_split=1,
        )
    staging = tmp_path / ".extracted.echoforge-extracting"
    assert (staging / extraction.MARKER_NAME).is_file()

    monkeypatch.setattr(extraction, "_promote_staging", real_promote)
    result = extract_aishell(
        output,
        download_manifest_path=manifest,
        audio_archive_path=audio,
        resources_archive_path=resources,
        speaker_limit_per_split=1,
    )
    assert result["reused"] is True
    assert output.is_dir()
    assert not staging.exists()


def test_refuses_a_tampered_completed_output(tmp_path: Path) -> None:
    manifest, audio, resources = _fixture(tmp_path)
    output = tmp_path / "extracted"
    extract_aishell(
        output,
        download_manifest_path=manifest,
        audio_archive_path=audio,
        resources_archive_path=resources,
        speaker_limit_per_split=1,
    )
    wav = output / "wav" / "dev" / "S0002" / "D0002.wav"
    wav.write_bytes(wav.read_bytes() + b"tampered")

    with pytest.raises(AishellExtractionError, match="refusing to overwrite or trust"):
        extract_aishell(
            output,
            download_manifest_path=manifest,
            audio_archive_path=audio,
            resources_archive_path=resources,
            speaker_limit_per_split=1,
        )


def test_cli_reports_machine_readable_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, audio, resources = _fixture(tmp_path)
    output = tmp_path / "cli-output"

    status = extract_cli.main(
        [
            "--output",
            str(output),
            "--download-manifest",
            str(manifest),
            "--audio-archive",
            str(audio),
            "--resources-archive",
            str(resources),
            "--split",
            "dev",
            "--speaker-limit-per-split",
            "1",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "extracted"
    assert payload["splits"] == ["dev"]
    assert payload["tree"]["selected_speakers"] == {"dev": ["S0002"]}
