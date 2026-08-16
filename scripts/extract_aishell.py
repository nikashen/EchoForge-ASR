"""Verify and safely extract selected AISHELL-1 splits from nested OpenSLR archives."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from echoforge.evaluation.aishell_extract import AishellExtractionError, extract_aishell


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--audio-archive", type=Path, required=True)
    parser.add_argument("--resources-archive", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("dev", "test"),
        action="append",
        dest="splits",
        help="split to extract; repeat for both (default: dev and test)",
    )
    parser.add_argument(
        "--speaker-limit-per-split",
        type=int,
        help="extract only the first N speaker IDs in each selected split",
    )
    args = parser.parse_args(argv)
    try:
        result = extract_aishell(
            args.output,
            download_manifest_path=args.download_manifest,
            audio_archive_path=args.audio_archive,
            resources_archive_path=args.resources_archive,
            splits=tuple(args.splits or ("dev", "test")),
            speaker_limit_per_split=args.speaker_limit_per_split,
        )
    except (AishellExtractionError, OSError) as exc:
        print(f"AISHELL extraction failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
