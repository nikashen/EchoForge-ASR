"""Recompute CER from a frozen row manifest, or fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from echoforge.evaluation.cer import edit_counts


def _valid_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def evaluate(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    reasons: list[str] = []
    if payload.get("schema_version") != "echoforge.eval-manifest/v1":
        reasons.append("unsupported manifest schema")
    if payload.get("frozen") is not True:
        reasons.append("manifest is not marked frozen")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        reasons.append("manifest has no evaluation rows")
        rows = []
    seen: set[str] = set()
    counts = {
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "reference_units": 0,
        "hypothesis_units": 0,
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            reasons.append(f"row {index} is not an object")
            continue
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id or row_id in seen:
            reasons.append(f"row {index} has a missing or duplicate id")
        else:
            seen.add(row_id)
        if not _valid_hash(row.get("audio_sha256")):
            reasons.append(f"row {index} has an invalid audio_sha256")
        reference = row.get("reference")
        hypothesis = row.get("hypothesis")
        if not isinstance(reference, str) or not isinstance(hypothesis, str):
            reasons.append(f"row {index} is missing reference/hypothesis")
            continue
        try:
            edit = edit_counts(reference, hypothesis)
        except (TypeError, ValueError) as exc:
            reasons.append(f"row {index} has invalid text normalization: {exc}")
            continue
        for key in counts:
            counts[key] += getattr(edit, key)
    if counts["reference_units"] == 0:
        reasons.append("normalized references contain no evaluation units")
    if reasons:
        return {
            "schema_version": "echoforge.report/v1",
            "status": "not_yet_evaluated",
            "reasons": sorted(set(reasons)),
            "rows": len(rows),
        }
    cer = (counts["substitutions"] + counts["deletions"] + counts["insertions"]) / counts[
        "reference_units"
    ]
    return {
        "schema_version": "echoforge.report/v1",
        "status": "evaluated",
        "rows": len(rows),
        "counts": counts,
        "cer": cer,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "normalization": payload.get("normalization", "echoforge.zh-normalizer/v1"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.manifest)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "evaluated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
