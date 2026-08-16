"""Strict JSON parsing and no-overwrite publication for evaluation evidence."""

from __future__ import annotations

import json
import math
import os
import stat
from pathlib import Path
from typing import Any


class EvidenceJsonError(ValueError):
    """Raised when evidence is not unambiguous RFC-compatible JSON."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceJsonError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise EvidenceJsonError(f"non-finite JSON number is not allowed: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise EvidenceJsonError(f"non-finite JSON number is not allowed: {value}")
    return parsed


def _validate_unicode(value: Any) -> None:
    if isinstance(value, str):
        if any("\ud800" <= character <= "\udfff" for character in value):
            raise EvidenceJsonError("unpaired UTF-16 surrogate is not allowed in evidence")
        return
    if isinstance(value, list):
        for item in value:
            _validate_unicode(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_unicode(key)
            _validate_unicode(item)


def strict_json_loads(raw: bytes | str) -> Any:
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    except UnicodeDecodeError as exc:
        raise EvidenceJsonError("evidence is not valid UTF-8") from exc
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except json.JSONDecodeError as exc:
        raise EvidenceJsonError("evidence is not valid JSON") from exc
    _validate_unicode(decoded)
    return decoded


def strict_json_dumps(payload: object) -> str:
    try:
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceJsonError("evidence cannot be encoded as strict JSON") from exc


def normalized_path(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def ensure_new_json_path(path: Path) -> Path:
    target = normalized_path(path)
    temporary = target.with_name(target.name + ".tmp")
    for candidate, label in ((target, "output"), (temporary, "temporary output")):
        metadata = _lstat(candidate)
        if metadata is not None:
            kind = "symbolic link" if stat.S_ISLNK(metadata.st_mode) else "existing path"
            raise FileExistsError(f"{label} is an {kind}: {candidate}")
    return target


def write_json_new(path: Path, payload: object) -> Path:
    target = ensure_new_json_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    encoded = strict_json_dumps(payload)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target, follow_symlinks=False)
        temporary.unlink()
    finally:
        metadata = _lstat(temporary)
        if metadata is not None and stat.S_ISREG(metadata.st_mode):
            temporary.unlink()
    return target


__all__ = [
    "EvidenceJsonError",
    "ensure_new_json_path",
    "normalized_path",
    "strict_json_dumps",
    "strict_json_loads",
    "write_json_new",
]
