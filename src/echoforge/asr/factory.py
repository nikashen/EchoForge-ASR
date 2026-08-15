"""Backend construction without import-time model side effects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .base import EndpointFinalizer, StreamingRecognizer
from .fake import ScriptedFinalizer, ScriptedStreamingRecognizer
from .faster_whisper import FasterWhisperFinalizer
from .preflight import run_preflight
from .sherpa_onnx import SherpaOnnxConfig, SherpaOnnxStreamingRecognizer


@dataclass(frozen=True, slots=True)
class BackendFactories:
    recognizer_factory: Callable[[], StreamingRecognizer]
    finalizer_factory: Callable[[], EndpointFinalizer | None]
    name: str
    evidence: dict[str, str]
    static_preflight_ok: bool = True


def _make_whisper_finalizer(
    verifier_model: str | Path | None,
    *,
    dual_pass: bool,
    allow_download: bool,
) -> EndpointFinalizer | None:
    if not dual_pass or verifier_model is None:
        return None
    return FasterWhisperFinalizer(
        verifier_model,
        device="cpu",
        allow_download=allow_download,
    )


def build_backend_factories(
    backend: str = "fake",
    *,
    model_dir: str | Path | None = None,
    verifier_model: str | Path | None = None,
    provider: str = "cpu",
    model_type: str = "zipformer",
    allow_download: bool = False,
    dual_pass: bool = True,
) -> BackendFactories:
    """Return factories suitable for ``SessionRegistry``.

    ``fake`` is deterministic and is the only backend used by CI/Pages.  Real
    backends require explicit local model paths and are never selected from a
    client message.
    """

    normalized = backend.strip().lower()
    if normalized == "fake":
        return BackendFactories(
            recognizer_factory=ScriptedStreamingRecognizer,
            finalizer_factory=(ScriptedFinalizer if dual_pass else (lambda: None)),
            name="deterministic-fake",
            evidence={
                "streaming_model": "deterministic-protocol-fixture",
                "verifier": "deterministic-verifier-fixture",
            },
        )
    if normalized not in {"sherpa", "sherpa-onnx"}:
        raise ValueError("backend must be fake or sherpa-onnx")
    if model_dir is None:
        raise ValueError("model_dir is required for sherpa-onnx backend")
    if dual_pass and verifier_model is None:
        raise ValueError("verifier_model is required when dual_pass is enabled")
    if allow_download:
        raise ValueError(
            "automatic model downloads are disabled for the serving factory; use local paths"
        )
    preflight = run_preflight(
        normalized,
        model_dir=model_dir,
        verifier_model=verifier_model,
        model_type=model_type,
        provider=provider,
        dual_pass=dual_pass,
    )
    if not preflight["ok"]:
        checks = preflight.get("checks", [])
        details = "; ".join(
            str(check.get("detail", "failed"))
            for check in checks
            if isinstance(check, dict) and not check.get("ok", False)
        )
        raise ValueError(f"backend preflight failed: {details or 'unknown prerequisite failure'}")
    config = SherpaOnnxConfig(model_dir=Path(model_dir), model_type=model_type, provider=provider)
    return BackendFactories(
        recognizer_factory=lambda: SherpaOnnxStreamingRecognizer(config),
        finalizer_factory=lambda: _make_whisper_finalizer(
            verifier_model,
            dual_pass=dual_pass,
            allow_download=allow_download,
        ),
        name="sherpa-onnx" + ("+faster-whisper" if dual_pass else ""),
        evidence={
            "streaming_model": "sherpa-onnx",
            "streaming_model_dir": str(Path(model_dir).expanduser()),
            "model_type": model_type,
            "verifier": "faster-whisper" if dual_pass else "disabled",
            "preflight": "static_passed",
        },
        static_preflight_ok=True,
    )
