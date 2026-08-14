from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from scripts.download_aishell import _safe_extract
from scripts.evaluate_manifest import evaluate


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


def test_evaluate_manifest_recomputes_character_counts(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "echoforge.eval-manifest/v1",
        "frozen": True,
        "normalization": "echoforge.zh-normalizer/v1",
        "rows": [
            {
                "id": "utt-1",
                "split": "test",
                "audio_sha256": "a" * 64,
                "reference": "你好",
                "hypothesis": "你啊",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = evaluate(path)
    assert report["status"] == "evaluated"
    assert report["counts"]["substitutions"] == 1
    assert report["cer"] == pytest.approx(0.5)


def test_evaluate_manifest_fails_closed_for_empty_normalized_references(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "echoforge.eval-manifest/v1",
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
