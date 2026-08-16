"""Fail-closed checks for operator-provided real ASR runtimes."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
from pathlib import Path
from typing import Any

from .faster_whisper import (
    LOCAL_MODEL_REQUIRED_FILES,
    LOCAL_MODEL_VOCABULARY_FILES,
    is_safe_local_model_file,
)
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


def _ctranslate2_check() -> tuple[bool, str]:
    runtime_ok, runtime_detail = _runtime_check("ctranslate2", ("models",))
    if not runtime_ok:
        return runtime_ok, runtime_detail
    try:
        version = importlib.metadata.version("ctranslate2")
    except importlib.metadata.PackageNotFoundError:
        return False, "ctranslate2 distribution metadata is unavailable"
    if platform.system() == "Windows" and version != "4.5.0":
        return (
            False,
            f"ctranslate2 {version} is unsupported by the Windows verifier contract; expected 4.5.0",
        )
    return True, f"ctranslate2 import available (version {version})"


def _check_directory(path_value: str | Path | None, label: str) -> tuple[bool, str, Path | None]:
    if path_value is None or not str(path_value).strip():
        return False, f"{label} is required", None
    path = Path(path_value).expanduser()
    if path.is_symlink():
        return False, f"{label} must not be a symbolic link: {path}", path
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

    def add(name: str, ok: bool | None, detail: str) -> None:
        """Record a passed, failed, or deliberately unverified check.

        ``None`` is reserved for facts a static inspection cannot establish.
        Keeping that state distinct avoids presenting CUDA compatibility as a
        green check while still allowing the static file/import contract to
        pass.
        """

        status = "not_verified" if ok is None else ("passed" if ok else "failed")
        checks.append({"name": name, "ok": ok, "status": status, "detail": detail})

    if normalized == "fake":
        add("backend", True, "deterministic-fake requires no model files")
        return {
            "ok": True,
            "static_requirements_ok": True,
            "backend": "deterministic-fake",
            "verification_level": "fixture",
            "model_load_verified": False,
            "runtime_probe_required": False,
            "checks": checks,
        }
    if normalized not in {"sherpa", "sherpa-onnx"}:
        add("backend", False, "backend must be fake or sherpa-onnx")
        return {
            "ok": False,
            "static_requirements_ok": False,
            "backend": normalized,
            "verification_level": "none",
            "model_load_verified": False,
            "runtime_probe_required": True,
            "checks": checks,
        }
    if model_type not in {"zipformer", "paraformer", "transducer"}:
        add("model_type", False, "model_type must be zipformer, paraformer, or transducer")
    if provider not in {"cpu", "cuda"}:
        add("provider", False, "provider must be cpu or cuda")
    else:
        add("provider", True, f"{provider} selected")
        if provider == "cuda":
            add(
                "cuda_compatibility",
                None,
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
        ctranslate2_ok, ctranslate2_detail = _ctranslate2_check()
        add("ctranslate2_runtime", ctranslate2_ok, ctranslate2_detail)
        if verifier_path is not None and verifier_ok:
            for filename in LOCAL_MODEL_REQUIRED_FILES:
                present = is_safe_local_model_file(verifier_path, filename)
                check_name = "verifier_" + filename.replace(".", "_")
                add(check_name, present, f"{filename} {'present' if present else 'missing'}")
            vocabulary = next(
                (
                    filename
                    for filename in LOCAL_MODEL_VOCABULARY_FILES
                    if is_safe_local_model_file(verifier_path, filename)
                ),
                None,
            )
            add(
                "verifier_vocabulary",
                vocabulary is not None,
                f"{vocabulary} present"
                if vocabulary is not None
                else "vocabulary.json/txt missing",
            )

    static_requirements_ok = all(check["ok"] is not False for check in checks)
    return {
        "ok": static_requirements_ok,
        "static_requirements_ok": static_requirements_ok,
        "backend": "sherpa-onnx",
        "model_type": model_type,
        "provider": provider,
        "dual_pass": dual_pass,
        "verification_level": "static",
        "model_load_verified": False,
        "runtime_probe_required": True,
        "checks": checks,
    }


__all__ = ["run_preflight"]
