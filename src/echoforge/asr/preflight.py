"""Fail-closed checks for operator-provided real ASR runtimes."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .faster_whisper import LOCAL_MODEL_REQUIRED_FILES
from .sherpa_onnx import (
    DECODER_CANDIDATES,
    ENCODER_CANDIDATES,
    JOINER_CANDIDATES,
    TOKENS_CANDIDATES,
    resolve_model_file,
)


def _runtime_check(name: str, required_symbols: tuple[str, ...]) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - DLL/API failures must fail closed
        detail = str(exc).strip() or type(exc).__name__
        return False, f"{name} import failed: {type(exc).__name__}: {detail}"
    missing = tuple(symbol for symbol in required_symbols if not hasattr(module, symbol))
    if missing:
        return False, f"{name} is missing required API: {', '.join(missing)}"
    version = getattr(module, "__version__", None)
    suffix = f" (version {version})" if version else ""
    return True, f"{name} import available{suffix}"


def _check_directory(path_value: str | Path | None, label: str) -> tuple[bool, str, Path | None]:
    if path_value is None or not str(path_value).strip():
        return False, f"{label} is required", None
    path = Path(path_value).expanduser()
    if not path.is_dir():
        return False, f"{label} directory does not exist: {path}", path
    return True, str(path), path


def _check_files(path: Path, candidates: tuple[str, ...], label: str) -> tuple[bool, str]:
    try:
        resolved = resolve_model_file(path, None, candidates=candidates)
    except (FileNotFoundError, ValueError) as exc:
        return False, str(exc)
    if resolved is not None:
        return True, resolved.name
    return False, f"{label} missing; expected one of: {', '.join(candidates)}"


def run_preflight(
    backend: str = "fake",
    *,
    model_dir: str | Path | None = None,
    verifier_model: str | Path | None = None,
    model_type: str = "zipformer",
    dual_pass: bool = True,
    provider: str = "cpu",
) -> dict[str, Any]:
    """Inspect local runtime prerequisites without loading model weights."""

    normalized = backend.strip().lower()
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    if normalized == "fake":
        add("backend", True, "deterministic-fake requires no model files")
        return {
            "ok": True,
            "backend": "deterministic-fake",
            "verification_level": "fixture",
            "model_load_verified": False,
            "checks": checks,
        }
    if normalized not in {"sherpa", "sherpa-onnx"}:
        add("backend", False, "backend must be fake or sherpa-onnx")
        return {"ok": False, "backend": normalized, "checks": checks}
    if model_type not in {"zipformer", "paraformer", "transducer"}:
        add("model_type", False, "model_type must be zipformer, paraformer, or transducer")
    if provider not in {"cpu", "cuda"}:
        add("provider", False, "provider must be cpu or cuda")
    elif provider == "cuda":
        add(
            "cuda_compatibility",
            True,
            "not verified by static preflight; run an end-to-end model load probe",
        )

    model_ok, model_detail, model_path = _check_directory(model_dir, "model_dir")
    add("model_dir", model_ok, model_detail)
    sherpa_runtime_ok, sherpa_runtime_detail = _runtime_check("sherpa_onnx", ("OnlineRecognizer",))
    add("sherpa_onnx_runtime", sherpa_runtime_ok, sherpa_runtime_detail)
    if model_path is not None and model_ok:
        tokens_ok, tokens_detail = _check_files(model_path, TOKENS_CANDIDATES, "tokens")
        add("tokens", tokens_ok, tokens_detail)
        if model_type == "paraformer":
            encoder_ok, encoder_detail = _check_files(model_path, ENCODER_CANDIDATES, "encoder")
            decoder_ok, decoder_detail = _check_files(model_path, DECODER_CANDIDATES, "decoder")
            add("encoder", encoder_ok, encoder_detail)
            add("decoder", decoder_ok, decoder_detail)
        else:
            for label, candidates in (
                ("encoder", ENCODER_CANDIDATES),
                ("decoder", DECODER_CANDIDATES),
                ("joiner", JOINER_CANDIDATES),
            ):
                file_ok, file_detail = _check_files(model_path, candidates, label)
                add(label, file_ok, file_detail)

    if dual_pass:
        verifier_ok, verifier_detail, verifier_path = _check_directory(
            verifier_model, "verifier_model"
        )
        add("verifier_model", verifier_ok, verifier_detail)
        whisper_runtime_ok, whisper_runtime_detail = _runtime_check(
            "faster_whisper", ("WhisperModel",)
        )
        add("faster_whisper_runtime", whisper_runtime_ok, whisper_runtime_detail)
        if verifier_path is not None and verifier_ok:
            for filename in LOCAL_MODEL_REQUIRED_FILES:
                present = (verifier_path / filename).is_file()
                check_name = "verifier_" + filename.replace(".", "_")
                add(check_name, present, f"{filename} {'present' if present else 'missing'}")

    return {
        "ok": all(bool(check["ok"]) for check in checks),
        "backend": "sherpa-onnx",
        "model_type": model_type,
        "provider": provider,
        "dual_pass": dual_pass,
        "verification_level": "static",
        "model_load_verified": False,
        "checks": checks,
    }


__all__ = ["run_preflight"]
