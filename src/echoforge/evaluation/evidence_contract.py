"""Shared canonical predicates for frozen evaluation evidence."""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:+-]{0,255}$")
FLOATING_REVISIONS = {
    "latest",
    "unknown",
    "unversioned",
    "none",
    "n/a",
    "main",
    "master",
    "head",
    "dev",
    "develop",
}
PLACEHOLDER_REVISION_MARKERS = ("fixture", "mock", "fake", "deterministic")
REQUIRED_DATASET_TEXT_FIELDS = {
    "name",
    "source",
    "source_url",
    "license",
    "speaker_policy",
    "audio_protocol",
}
REQUIRED_DATASET_HASH_FIELDS = {
    "license_page_sha256",
    "download_manifest_sha256",
    "extraction_inventory_sha256",
    "extraction_marker_sha256",
    "transcript_sha256",
}


def canonical_identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and value == value.strip()
        and IDENTIFIER_PATTERN.fullmatch(value) is not None
    )


def valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def valid_revision(value: object) -> bool:
    if not isinstance(value, str) or REVISION_PATTERN.fullmatch(value) is None:
        return False
    lowered = value.casefold()
    revision_segments = set(re.split(r"[@/:]+", lowered))
    return (
        lowered not in FLOATING_REVISIONS
        and not revision_segments.intersection(FLOATING_REVISIONS)
        and not any(marker in lowered for marker in PLACEHOLDER_REVISION_MARKERS)
    )


def valid_audio_relpath(value: object) -> bool:
    if not isinstance(value, str) or value != value.strip() or not value:
        return False
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    return (
        "\x00" not in value
        and "\\" not in value
        and not path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def validate_dataset_provenance(
    dataset: object,
) -> tuple[list[str], dict[str, Any] | None]:
    reasons: list[str] = []
    if not isinstance(dataset, dict):
        return ["manifest has no dataset provenance"], None
    for field in sorted(REQUIRED_DATASET_TEXT_FIELDS):
        value = dataset.get(field)
        if not isinstance(value, str) or value != value.strip() or not value:
            reasons.append(f"dataset {field} is missing or invalid")
    source_url = dataset.get("source_url")
    if isinstance(source_url, str) and not source_url.startswith("https://"):
        reasons.append("dataset source_url must use https")
    for field in sorted(REQUIRED_DATASET_HASH_FIELDS):
        if not valid_sha256(dataset.get(field)):
            reasons.append(f"dataset {field} is missing or invalid")
    if dataset.get("raw_audio_in_repository") is not False:
        reasons.append("dataset raw_audio_in_repository must be false")
    selection = dataset.get("selection")
    if not isinstance(selection, dict):
        reasons.append("dataset has no deterministic selection evidence")
        return reasons, None
    for field, message in (
        ("speaker_limit_per_split", "authorized dataset selection must not limit speakers"),
        (
            "utterances_per_speaker",
            "authorized dataset selection must not limit utterances",
        ),
        (
            "extraction_speaker_limit_per_split",
            "authorized dataset extraction must not limit speakers",
        ),
    ):
        if field not in selection:
            reasons.append(f"dataset selection is missing {field}")
        elif selection[field] is not None:
            reasons.append(message)
    return reasons, dict(selection)


def validate_dataset_selection(
    selection: dict[str, Any] | None,
    *,
    splits: set[str],
    row_count: int,
    speakers: set[str],
) -> list[str]:
    reasons: list[str] = []
    if selection is None or len(splits) != 1:
        return reasons
    split = next(iter(splits))
    if selection.get("splits") != [split]:
        reasons.append("dataset selection split is inconsistent with rows")
    selected_rows = selection.get("rows")
    if (
        isinstance(selected_rows, bool)
        or not isinstance(selected_rows, int)
        or selected_rows != row_count
    ):
        reasons.append("dataset selection row count is inconsistent")
    selected_speakers = selection.get("selected_speakers")
    if not isinstance(selected_speakers, dict) or set(selected_speakers) != {split}:
        reasons.append("dataset selection speakers are missing or inconsistent")
        return reasons
    values = selected_speakers.get(split)
    if (
        not isinstance(values, list)
        or not values
        or any(not canonical_identifier(value) for value in values)
        or values != sorted(set(values))
        or set(values) != speakers
    ):
        reasons.append("dataset selection speakers are missing or inconsistent")
    return reasons


__all__ = [
    "canonical_identifier",
    "valid_audio_relpath",
    "valid_revision",
    "valid_sha256",
    "validate_dataset_provenance",
    "validate_dataset_selection",
]
