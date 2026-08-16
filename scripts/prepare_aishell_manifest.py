"""Build a deterministic local AISHELL-1 prepared manifest from extracted data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from echoforge.evaluation.aishell import prepare_aishell_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wav-root", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--extraction-root", type=Path)
    parser.add_argument("--split", choices=("dev", "test"), action="append", dest="splits")
    parser.add_argument("--speaker-limit", type=int)
    parser.add_argument("--utterances-per-speaker", type=int)
    parser.add_argument("--authorize-evaluation", action="store_true")
    parser.add_argument("--protocol-id")
    args = parser.parse_args()
    output_path = args.output.expanduser().resolve()
    try:
        result = prepare_aishell_manifest(
            output_path,
            wav_root=args.wav_root,
            transcript_path=args.transcript,
            download_manifest_path=args.download_manifest,
            splits=tuple(args.splits or ("dev",)),
            speaker_limit=args.speaker_limit,
            utterances_per_speaker=args.utterances_per_speaker,
            evaluation_authorized=args.authorize_evaluation,
            protocol_id=args.protocol_id,
            extraction_root=args.extraction_root,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"AISHELL manifest preparation failed: {exc}", file=sys.stderr)
        return 1
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "status": "prepared",
                "output": str(output_path),
                "output_sha256": digest,
                "rows": len(result["rows"]),
                "evaluation_authorized": result["evaluation_authorized"],
                "protocol_id": result["protocol_id"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
