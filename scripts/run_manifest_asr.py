"""Run a prepared local ASR manifest and emit a new frozen result manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from echoforge.evaluation.runner import run_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("fake", "sherpa-onnx"), required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--verifier-model", type=Path)
    parser.add_argument(
        "--model-type", choices=("zipformer", "paraformer", "transducer"), default="zipformer"
    )
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-dual-pass", action="store_true")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--streaming-revision")
    parser.add_argument("--verifier-revision")
    parser.add_argument("--streaming-source-url")
    parser.add_argument("--streaming-license")
    parser.add_argument("--confirm-streaming-license-reviewed", action="store_true")
    parser.add_argument("--verifier-source-url")
    parser.add_argument("--verifier-license")
    parser.add_argument("--confirm-verifier-license-reviewed", action="store_true")
    parser.add_argument("--warmup-audio", type=Path)
    args = parser.parse_args()
    output_path = args.output.expanduser().resolve()
    try:
        result = run_manifest(
            args.manifest.expanduser().resolve(),
            output_path,
            audio_root=args.audio_root,
            backend=args.backend,
            model_dir=args.model_dir,
            verifier_model=args.verifier_model,
            model_type=args.model_type,
            provider=args.provider,
            dual_pass=not args.no_dual_pass,
            chunk_ms=args.chunk_ms,
            streaming_revision=args.streaming_revision,
            verifier_revision=args.verifier_revision,
            streaming_source_url=args.streaming_source_url,
            streaming_license=args.streaming_license,
            streaming_license_reviewed=args.confirm_streaming_license_reviewed,
            verifier_source_url=args.verifier_source_url,
            verifier_license=args.verifier_license,
            verifier_license_reviewed=args.confirm_verifier_license_reviewed,
            warmup_audio=args.warmup_audio,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"manifest ASR run failed: {exc}", file=sys.stderr)
        return 1
    output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "status": "completed_frozen" if result["frozen"] else "completed_unscored",
                "output": str(output_path),
                "output_sha256": output_hash,
                "rows": result["runtime_summary"]["rows"],
                "aggregate_utterance_rtf": result["runtime_summary"]["aggregate_utterance_rtf"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
