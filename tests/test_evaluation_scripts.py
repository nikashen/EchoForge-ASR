from __future__ import annotations

import hashlib
import io
import json
import tarfile
import wave
from pathlib import Path

import pytest

from echoforge.evaluation.aishell import prepare_aishell_manifest
from echoforge.evaluation.runner import ManifestRunnerError, run_manifest
from scripts import download_aishell
from scripts.download_aishell import _download, _safe_extract
from scripts.evaluate_manifest import evaluate


def _write_pcm16_wav(path: Path, *, frames: int = 3200) -> str:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(b"\x00\x00" * frames)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepared_manifest(audio_hash: str) -> dict[str, object]:
    return {
        "schema_version": "echoforge.eval-manifest/v1",
        "dataset": {
            "name": "unit-fixture",
            "source": "generated unit-test fixture",
            "speaker_policy": "speaker-disjoint",
        },
        "protocol_id": "unit-fixture-v1",
        "evaluation_authorized": False,
        "frozen": False,
        "normalization": "echoforge.zh-normalizer/v1",
        "rows": [
            {
                "id": "utt-1",
                "speaker_id": "speaker-1",
                "split": "dev",
                "audio_relpath": "utt-1.wav",
                "audio_sha256": audio_hash,
                "reference": "你好世界",
            }
        ],
    }


def _write_download_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "echoforge.download/v1",
                "dataset": "AISHELL-1",
                "license_declared": "Apache-2.0",
                "license_text_sha256": "c" * 64,
                "dry_run": False,
                "archives": [
                    {
                        "name": "audio",
                        "url": "https://www.openslr.org/resources/33/data_aishell.tgz",
                        "bytes": 15_582_913_665,
                        "sha256": "a" * 64,
                    },
                    {
                        "name": "resources",
                        "url": "https://www.openslr.org/resources/33/resource_aishell.tgz",
                        "bytes": 1_246_920,
                        "sha256": "b" * 64,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_evaluate_manifest_fails_closed_until_frozen(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "echoforge.eval-manifest/v1",
        "frozen": False,
        "rows": [],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = evaluate(path)
    assert report["status"] == "not_yet_evaluated"
    assert "manifest has no evaluation rows" in report["reasons"]
    assert "manifest is not authorized for quality evaluation" in report["reasons"]


def test_evaluate_manifest_fails_closed_for_empty_normalized_references(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "echoforge.eval-manifest/v1",
        "evaluation_authorized": True,
        "frozen": True,
        "rows": [
            {
                "id": "utt-empty",
                "audio_sha256": "b" * 64,
                "reference": "!?  ",
                "hypothesis": "任意文本",
            }
        ],
    }
    path = tmp_path / "empty-reference.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = evaluate(path)
    assert report["status"] == "not_yet_evaluated"
    assert "normalized references contain no evaluation units" in report["reasons"]


def test_safe_extract_rejects_path_traversal_and_links(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 0
        tar.addfile(info)
    with pytest.raises(ValueError, match="escapes"):
        _safe_extract(archive, tmp_path / "out")


@pytest.mark.parametrize("resumed", [False, True])
def test_download_is_atomic_and_resumable(tmp_path: Path, monkeypatch, resumed: bool) -> None:
    destination = tmp_path / "archive.tgz"
    partial = tmp_path / "archive.tgz.part"
    if resumed:
        partial.write_bytes(b"abc")

    probe_headers = {
        "Content-Length": "6",
        "Accept-Ranges": "bytes",
        "ETag": '"fixture"',
        "Last-Modified": "Sat, 15 Aug 2026 00:00:00 GMT",
    }
    if resumed:
        (tmp_path / "archive.tgz.part.json").write_text(
            json.dumps(
                {
                    "url": "https://example.invalid/archive.tgz",
                    "expected_bytes": 6,
                    "etag": '"fixture"',
                    "last_modified": "Sat, 15 Aug 2026 00:00:00 GMT",
                    "accept_ranges": "bytes",
                }
            ),
            encoding="utf-8",
        )

    class Response(io.BytesIO):
        def __init__(self, data: bytes, status: int, headers: dict[str, str | None]) -> None:
            super().__init__(data)
            self.status = status
            self.headers = headers

        def getcode(self) -> int:
            return self.status

    def open_fixture(request, timeout):
        assert timeout == 60
        if request.get_method() == "HEAD":
            return Response(b"", 200, probe_headers)
        assert request.get_header("Range") == ("bytes=3-" if resumed else None)
        assert request.get_header("If-range") == ('"fixture"' if resumed else None)
        assert request.get_header("If-match") == (None if resumed else '"fixture"')
        headers = dict(probe_headers)
        headers["Content-Length"] = "3" if resumed else "6"
        headers["Content-Range"] = "bytes 3-5/6" if resumed else None
        return Response(b"def" if resumed else b"abcdef", 206 if resumed else 200, headers)

    monkeypatch.setattr(download_aishell.urllib.request, "urlopen", open_fixture)
    result = _download("https://example.invalid/archive.tgz", destination)

    assert destination.read_bytes() == b"abcdef"
    assert not partial.exists()
    assert result["resumed_from_bytes"] == (3 if resumed else 0)
    assert result["sha256"] == hashlib.sha256(b"abcdef").hexdigest()
    receipt = tmp_path / "archive.tgz.download.json"
    assert receipt.is_file()
    assert json.loads(receipt.read_text(encoding="utf-8"))["dry_run"] is False


def _download_probe() -> dict[str, object]:
    return {
        "url": "https://example.invalid/archive.tgz",
        "expected_bytes": 6,
        "etag": '"fixture"',
        "last_modified": "Sat, 15 Aug 2026 00:00:00 GMT",
        "accept_ranges": "bytes",
    }


def _write_download_sidecar(path: Path) -> None:
    path.write_text(json.dumps(_download_probe()), encoding="utf-8")


def test_download_recovers_sidecar_without_partial_bytes(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "archive.tgz"
    sidecar = tmp_path / "archive.tgz.part.json"
    _write_download_sidecar(sidecar)

    class Response(io.BytesIO):
        def __init__(self, data: bytes, status: int, headers: dict[str, str]) -> None:
            super().__init__(data)
            self.status = status
            self.headers = headers

    def open_fixture(request, timeout):
        assert timeout == 60
        headers = {
            "Content-Length": "6",
            "Accept-Ranges": "bytes",
            "ETag": '"fixture"',
            "Last-Modified": "Sat, 15 Aug 2026 00:00:00 GMT",
        }
        return Response(b"" if request.get_method() == "HEAD" else b"abcdef", 200, headers)

    monkeypatch.setattr(download_aishell.urllib.request, "urlopen", open_fixture)
    result = _download("https://example.invalid/archive.tgz", destination)

    assert destination.read_bytes() == b"abcdef"
    assert result["resumed_from_bytes"] == 0
    assert not sidecar.exists()
    assert (tmp_path / "archive.tgz.download.json").is_file()


def test_download_promotes_already_complete_partial_without_range_request(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "archive.tgz"
    partial = tmp_path / "archive.tgz.part"
    partial.write_bytes(b"abcdef")
    _write_download_sidecar(tmp_path / "archive.tgz.part.json")
    calls: list[str] = []

    class Response(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(b"")
            self.status = 200
            self.headers = {
                "Content-Length": "6",
                "Accept-Ranges": "bytes",
                "ETag": '"fixture"',
                "Last-Modified": "Sat, 15 Aug 2026 00:00:00 GMT",
            }

    def open_fixture(request, timeout):
        assert timeout == 60
        calls.append(request.get_method())
        assert request.get_method() == "HEAD"
        return Response()

    monkeypatch.setattr(download_aishell.urllib.request, "urlopen", open_fixture)
    result = _download("https://example.invalid/archive.tgz", destination)

    assert calls == ["HEAD"]
    assert destination.read_bytes() == b"abcdef"
    assert result["promoted_complete_partial"] is True
    assert not partial.exists()
    assert (tmp_path / "archive.tgz.download.json").is_file()


def test_download_retains_partial_when_server_rejects_range(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "archive.tgz"
    partial = tmp_path / "archive.tgz.part"
    partial.write_bytes(b"abc")
    sidecar = tmp_path / "archive.tgz.part.json"
    _write_download_sidecar(sidecar)

    class Response(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(b"")
            self.status = 200
            self.headers = {
                "Content-Length": "6",
                "Accept-Ranges": "bytes",
                "ETag": '"fixture"',
                "Last-Modified": "Sat, 15 Aug 2026 00:00:00 GMT",
            }

    def open_fixture(request, timeout):
        assert timeout == 60
        if request.get_method() == "HEAD":
            return Response()
        raise download_aishell.urllib.error.HTTPError(
            request.full_url, 416, "Range Not Satisfiable", {}, None
        )

    monkeypatch.setattr(download_aishell.urllib.request, "urlopen", open_fixture)
    with pytest.raises(RuntimeError, match="resume boundary"):
        _download("https://example.invalid/archive.tgz", destination)

    assert partial.read_bytes() == b"abc"
    assert sidecar.is_file()
    assert not destination.exists()


def test_download_rejects_non_regular_and_symlink_partial_state(tmp_path: Path) -> None:
    destination = tmp_path / "archive.tgz"
    partial = tmp_path / "archive.tgz.part"
    partial.mkdir()
    with pytest.raises(RuntimeError, match="regular file"):
        _download("https://example.invalid/archive.tgz", destination)

    partial.rmdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"do-not-touch")
    try:
        partial.symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links are unavailable on this host")
    with pytest.raises(RuntimeError, match="symbolic link"):
        _download("https://example.invalid/archive.tgz", destination)
    assert outside.read_bytes() == b"do-not-touch"


def test_completed_archive_migrates_legacy_validator_and_reuses_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "archive.tgz"
    destination.write_bytes(b"abcdef")
    sidecar = tmp_path / "archive.tgz.part.json"
    _write_download_sidecar(sidecar)
    monkeypatch.setattr(download_aishell, "_probe_download", lambda _url: _download_probe())

    migrated = download_aishell._verify_completed_download(
        "https://example.invalid/archive.tgz", destination
    )
    receipt = tmp_path / "archive.tgz.download.json"
    reused = download_aishell._verify_completed_download(
        "https://example.invalid/archive.tgz", destination
    )

    assert migrated["migrated_legacy_validator"] is True
    assert reused["migrated_legacy_validator"] is False
    assert receipt.is_file()
    assert json.loads(receipt.read_text(encoding="utf-8"))["dry_run"] is False
    assert not sidecar.exists()


def test_completed_archive_rejects_missing_or_changed_validator(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "archive.tgz"
    destination.write_bytes(b"abcdef")
    monkeypatch.setattr(download_aishell, "_probe_download", lambda _url: _download_probe())

    with pytest.raises(RuntimeError, match="no persistent validator"):
        download_aishell._verify_completed_download(
            "https://example.invalid/archive.tgz", destination
        )

    sidecar = tmp_path / "archive.tgz.part.json"
    _write_download_sidecar(sidecar)
    download_aishell._verify_completed_download("https://example.invalid/archive.tgz", destination)
    changed = _download_probe()
    changed["etag"] = '"same-size-new-object"'
    monkeypatch.setattr(download_aishell, "_probe_download", lambda _url: changed)
    with pytest.raises(RuntimeError, match="validator changed"):
        download_aishell._verify_completed_download(
            "https://example.invalid/archive.tgz", destination
        )


def test_completed_archive_receipt_requires_explicit_non_dry_run(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "archive.tgz"
    destination.write_bytes(b"abcdef")
    probe = _download_probe()
    receipt = download_aishell._receipt_payload(
        probe,
        archive_bytes=6,
        archive_sha256=hashlib.sha256(b"abcdef").hexdigest(),
    )
    receipt.pop("dry_run")
    (tmp_path / "archive.tgz.download.json").write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(download_aishell, "_probe_download", lambda _url: probe)

    with pytest.raises(RuntimeError, match="invalid validator receipt"):
        download_aishell._verify_completed_download(
            "https://example.invalid/archive.tgz", destination
        )


def test_receipt_failure_retains_legacy_sidecar_after_archive_promotion(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "archive.tgz"
    original_write_json = download_aishell._write_json_new

    class Response(io.BytesIO):
        def __init__(self, data: bytes, status: int) -> None:
            super().__init__(data)
            self.status = status
            self.headers = {
                "Content-Length": "6",
                "Accept-Ranges": "bytes",
                "ETag": '"fixture"',
                "Last-Modified": "Sat, 15 Aug 2026 00:00:00 GMT",
            }

    def open_fixture(request, timeout):
        assert timeout == 60
        return Response(b"" if request.get_method() == "HEAD" else b"abcdef", 200)

    def fail_receipt(path: Path, payload: dict[str, object]) -> None:
        if path.name.endswith(".download.json"):
            raise OSError("simulated receipt persistence failure")
        original_write_json(path, payload)

    monkeypatch.setattr(download_aishell.urllib.request, "urlopen", open_fixture)
    monkeypatch.setattr(download_aishell, "_write_json_new", fail_receipt)
    with pytest.raises(OSError, match="receipt persistence"):
        _download("https://example.invalid/archive.tgz", destination)

    assert destination.read_bytes() == b"abcdef"
    assert (tmp_path / "archive.tgz.part.json").is_file()
    assert not (tmp_path / "archive.tgz.download.json").exists()


def test_dry_run_plan_does_not_block_atomic_download_manifest(tmp_path: Path, monkeypatch) -> None:
    plan = download_aishell.run(tmp_path, accept_license=True, dry_run=True, extract=False)
    assert plan.name == "download_plan.json"
    assert json.loads(plan.read_text(encoding="utf-8"))["dry_run"] is True

    monkeypatch.setattr(download_aishell, "_remote_sha256", lambda _url: "a" * 64)

    def download_fixture(_url: str, destination: Path) -> dict[str, object]:
        destination.write_bytes(b"fixture")
        return {
            "bytes": 7,
            "sha256": hashlib.sha256(b"fixture").hexdigest(),
            "resumed_from_bytes": 0,
            "etag": '"fixture"',
            "last_modified": None,
        }

    monkeypatch.setattr(download_aishell, "_download", download_fixture)
    manifest = download_aishell.run(tmp_path, accept_license=True, dry_run=False, extract=False)

    assert manifest.name == "download_manifest.json"
    assert plan.is_file() and manifest.is_file()
    assert json.loads(manifest.read_text(encoding="utf-8"))["dry_run"] is False
    assert not (tmp_path / "download_manifest.json.tmp").exists()


def test_atomic_json_recovers_only_matching_complete_temporary_file(tmp_path: Path) -> None:
    output = tmp_path / "archive.tgz.download.json"
    temporary = tmp_path / "archive.tgz.download.json.tmp"
    payload: dict[str, object] = {"schema_version": "fixture/v1", "dry_run": False}
    temporary.write_text(json.dumps(payload), encoding="utf-8")

    download_aishell._write_json_new(output, payload)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not temporary.exists()

    mismatched = tmp_path / "download_manifest.json"
    mismatched_temporary = tmp_path / "download_manifest.json.tmp"
    mismatched_temporary.write_text(json.dumps({"different": True}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match expected evidence"):
        download_aishell._write_json_new(mismatched, payload)
    assert mismatched_temporary.is_file()
    assert not mismatched.exists()


def test_extraction_recovers_completed_and_incomplete_staging(tmp_path: Path) -> None:
    archive = tmp_path / "archive.tgz"
    with tarfile.open(archive, "w:gz") as target:
        info = tarfile.TarInfo("payload/file.txt")
        payload = b"fixture"
        info.size = len(payload)
        target.addfile(info, io.BytesIO(payload))
    download = {
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }

    first = download_aishell._extract_or_reuse(
        "audio", "https://example.invalid/audio.tgz", archive, tmp_path / "audio", download
    )
    second = download_aishell._extract_or_reuse(
        "audio", "https://example.invalid/audio.tgz", archive, tmp_path / "audio", download
    )
    assert first["extraction_reused"] is False
    assert second["extraction_reused"] is True
    assert (tmp_path / "audio" / "payload" / "file.txt").read_bytes() == b"fixture"

    incomplete = tmp_path / ".resources.extracting"
    incomplete.mkdir()
    (incomplete / "partial.txt").write_text("partial", encoding="utf-8")
    recovered = download_aishell._extract_or_reuse(
        "resources",
        "https://example.invalid/resources.tgz",
        archive,
        tmp_path / "resources",
        download,
    )
    assert recovered["extraction_reused"] is False
    assert not incomplete.exists()
    assert (tmp_path / "resources" / "payload" / "file.txt").is_file()


def test_extraction_rejects_wrong_hash_and_archive_mutation(tmp_path: Path, monkeypatch) -> None:
    archive = tmp_path / "archive.tgz"
    with tarfile.open(archive, "w:gz") as target:
        info = tarfile.TarInfo("payload.txt")
        payload = b"fixture"
        info.size = len(payload)
        target.addfile(info, io.BytesIO(payload))
    download = {
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    wrong = {**download, "sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="archive evidence mismatch"):
        download_aishell._extract_or_reuse(
            "audio", "https://example.invalid/audio.tgz", archive, tmp_path / "wrong", wrong
        )
    assert not (tmp_path / "wrong").exists()

    original_extract = download_aishell._safe_extract

    def mutate_after_extract(source: Path, target: Path) -> None:
        original_extract(source, target)
        source.write_bytes(source.read_bytes() + b"changed")

    monkeypatch.setattr(download_aishell, "_safe_extract", mutate_after_extract)
    with pytest.raises(RuntimeError, match="archive evidence mismatch"):
        download_aishell._extract_or_reuse(
            "resources",
            "https://example.invalid/resources.tgz",
            archive,
            tmp_path / "mutated",
            download,
        )
    assert not (tmp_path / "mutated").exists()
    assert (tmp_path / ".resources.extracting").is_dir()


def test_manifest_runner_keeps_hash_verified_fake_results_unscored(tmp_path: Path) -> None:
    audio_hash = _write_pcm16_wav(tmp_path / "utt-1.wav")
    manifest_path = tmp_path / "prepared.json"
    output_path = tmp_path / "frozen.json"
    manifest_path.write_text(json.dumps(_prepared_manifest(audio_hash)), encoding="utf-8")

    result = run_manifest(manifest_path, output_path, audio_root=tmp_path, backend="fake")

    assert result["frozen"] is False
    assert result["evaluation_authorized"] is False
    assert result["rows"][0]["final_stage"] == "dual_pass_final"
    assert result["rows"][0]["audio_sha256"] == audio_hash
    assert result["runtime_summary"]["rows"] == 1
    assert evaluate(output_path)["status"] == "not_yet_evaluated"
    with pytest.raises(ManifestRunnerError, match="output already exists"):
        run_manifest(manifest_path, output_path, audio_root=tmp_path, backend="fake")


def test_manifest_runner_rejects_audio_hash_mismatch(tmp_path: Path) -> None:
    _write_pcm16_wav(tmp_path / "utt-1.wav")
    manifest_path = tmp_path / "prepared.json"
    output_path = tmp_path / "frozen.json"
    manifest_path.write_text(json.dumps(_prepared_manifest("0" * 64)), encoding="utf-8")

    with pytest.raises(ManifestRunnerError, match="audio hash mismatch"):
        run_manifest(manifest_path, output_path, audio_root=tmp_path, backend="fake")
    assert not output_path.exists()


def test_manifest_runner_rejects_cross_split_speaker_overlap(tmp_path: Path) -> None:
    audio_hash = _write_pcm16_wav(tmp_path / "utt-1.wav")
    second_hash = _write_pcm16_wav(tmp_path / "utt-2.wav", frames=3201)
    manifest = _prepared_manifest(audio_hash)
    rows = manifest["rows"]
    assert isinstance(rows, list)
    rows.append(
        {
            "id": "utt-2",
            "speaker_id": "speaker-1",
            "split": "test",
            "audio_relpath": "utt-2.wav",
            "audio_sha256": second_hash,
            "reference": "你好世界",
        }
    )
    manifest_path = tmp_path / "prepared.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestRunnerError, match="multiple splits"):
        run_manifest(
            manifest_path,
            tmp_path / "frozen.json",
            audio_root=tmp_path,
            backend="fake",
        )


def test_prepare_aishell_manifest_is_deterministic_and_unscored(tmp_path: Path) -> None:
    wav_root = tmp_path / "wav"
    first = wav_root / "dev" / "S0001" / "U0001.wav"
    second = wav_root / "dev" / "S0002" / "U0002.wav"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first_hash = _write_pcm16_wav(first)
    _write_pcm16_wav(second)
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("U0002 世 界\nU0001 你 好\n", encoding="utf-8")
    download_manifest = tmp_path / "download.json"
    _write_download_manifest(download_manifest)
    output = tmp_path / "prepared.json"

    result = prepare_aishell_manifest(
        output,
        wav_root=wav_root,
        transcript_path=transcript,
        download_manifest_path=download_manifest,
        speaker_limit=1,
        utterances_per_speaker=1,
    )

    assert result["evaluation_authorized"] is False
    assert result["frozen"] is False
    assert [row["id"] for row in result["rows"]] == ["U0001"]
    assert result["rows"][0]["audio_sha256"] == first_hash
    assert result["rows"][0]["reference"] == "你好"


def test_prepare_aishell_manifest_rejects_cross_split_speakers(tmp_path: Path) -> None:
    wav_root = tmp_path / "wav"
    dev = wav_root / "dev" / "S0001" / "DEV.wav"
    test = wav_root / "test" / "S0001" / "TEST.wav"
    dev.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    _write_pcm16_wav(dev)
    _write_pcm16_wav(test, frames=3201)
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("DEV 你 好\nTEST 世 界\n", encoding="utf-8")
    download_manifest = tmp_path / "download.json"
    _write_download_manifest(download_manifest)

    with pytest.raises(ManifestRunnerError, match="multiple splits"):
        prepare_aishell_manifest(
            tmp_path / "prepared.json",
            wav_root=wav_root,
            transcript_path=transcript,
            download_manifest_path=download_manifest,
            splits=("dev", "test"),
        )
