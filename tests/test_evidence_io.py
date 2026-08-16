from __future__ import annotations

import json
from pathlib import Path

import pytest

from echoforge.evaluation.evidence_io import (
    EvidenceJsonError,
    strict_json_dumps,
    strict_json_loads,
    write_json_new,
)


def test_strict_json_round_trip_and_no_overwrite_publication(tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    published = write_json_new(output, {"schema": "unit/v1", "value": 1})

    assert published == output
    assert json.loads(output.read_text(encoding="utf-8"))["value"] == 1
    assert not (tmp_path / "evidence.json.tmp").exists()
    with pytest.raises(FileExistsError):
        write_json_new(output, {"value": 2})


@pytest.mark.parametrize(
    "raw",
    [
        b'{"value": 1, "value": 2}',
        b'{"value": NaN}',
        b'{"value": 1e999}',
        b'{"value": "\\ud800"}',
        b"\xff",
    ],
)
def test_strict_json_rejects_ambiguous_or_nonstandard_input(raw: bytes) -> None:
    with pytest.raises(EvidenceJsonError):
        strict_json_loads(raw)


def test_strict_json_refuses_non_finite_output() -> None:
    with pytest.raises(EvidenceJsonError):
        strict_json_dumps({"value": float("nan")})


def test_writer_refuses_preexisting_temporary_path(tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    temporary = tmp_path / "evidence.json.tmp"
    temporary.write_text("operator-owned", encoding="utf-8")

    with pytest.raises(FileExistsError, match="temporary output"):
        write_json_new(output, {"value": 1})
    assert temporary.read_text(encoding="utf-8") == "operator-owned"


def test_writer_refuses_broken_symlink_target_when_available(tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    try:
        output.symlink_to(tmp_path / "missing.json")
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable on this platform: {exc}")

    with pytest.raises(FileExistsError, match="symbolic link"):
        write_json_new(output, {"value": 1})
