from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from echoforge.asr.factory import build_backend_factories
from echoforge.streaming.registry import SessionRegistry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="echoforge", description="EchoForge-ASR streaming speech lab"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the local HTTP/WebSocket service")
    serve.add_argument("--backend", choices=("fake", "sherpa-onnx"), default="fake")
    serve.add_argument("--host", default=os.getenv("ECHOFORGE_HOST", "127.0.0.1"))
    serve.add_argument("--port", type=int, default=8090)
    serve.add_argument("--model-dir", type=Path)
    serve.add_argument("--verifier-model", type=Path)
    serve.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    serve.add_argument("--no-dual-pass", action="store_true")
    serve.add_argument("--allowed-origin", action="append", dest="allowed_origins")

    smoke = sub.add_parser("smoke", help="run a deterministic in-process protocol smoke test")
    smoke.add_argument("--seconds", type=float, default=0.5)
    smoke.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _run_smoke(seconds: float) -> dict[str, object]:
    if not 0.05 <= seconds <= 30:
        raise ValueError("seconds must be in [0.05, 30]")
    factories = build_backend_factories("fake")
    registry = SessionRegistry(max_sessions=1)
    session = registry.create(
        "smoke-session", factories.recognizer_factory, factories.finalizer_factory
    )
    samples = np.zeros(round(seconds * 16_000), dtype=np.float32)
    # Two chunks exercise partial and final revision paths without a microphone.
    midpoint = max(1, samples.size // 2)
    session.ingest(0, samples[:midpoint])
    session.ingest(1, samples[midpoint:])
    result = session.flush(expected_generation=session.generation)
    snapshot = session.snapshot()
    registry.close_all()
    return {
        "backend": factories.name,
        "utterance_id": result.utterance_id,
        "revision_stages": [revision.stage.value for revision in result.revisions],
        "revision_count": len(result.revisions),
        "verifier_degraded": result.verifier_degraded,
        "session_state": snapshot.state.value,
    }


def _run_serve(args: argparse.Namespace) -> int:
    try:
        factories = build_backend_factories(
            args.backend,
            model_dir=args.model_dir,
            verifier_model=args.verifier_model,
            provider=args.provider,
            dual_pass=not args.no_dual_pass,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        import uvicorn

        from echoforge.api.app import create_app
    except ImportError:
        print("serve requires echoforge-asr[serve]", file=sys.stderr)
        return 2
    try:
        app = create_app(
            factories=factories,
            allowed_origins=tuple(
                args.allowed_origins
                or (
                    f"http://{args.host}:{args.port}",
                    f"http://127.0.0.1:{args.port}",
                    f"http://localhost:{args.port}",
                )
            ),
        )
    except ImportError:
        print("serve requires echoforge-asr[serve]", file=sys.stderr)
        return 2
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "smoke":
        try:
            result = _run_smoke(args.seconds)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"smoke failed: {exc}", file=sys.stderr)
            return 1
        print(
            json.dumps(result, ensure_ascii=False, sort_keys=True)
            if args.as_json
            else "EchoForge smoke OK: "
            + ", ".join(f"{key}={value}" for key, value in result.items())
        )
        return 0
    if args.command == "serve":
        return _run_serve(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
