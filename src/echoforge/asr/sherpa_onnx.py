"""Optional sherpa-onnx streaming adapter.

The adapter is intentionally lazy: importing EchoForge never imports sherpa or
opens a model.  A model directory/file must be supplied by the operator; this
module never downloads weights from a request or at import time.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from echoforge.contracts.domain import Hypothesis, RevisionStage

FloatAudio = NDArray[np.float32]

TOKENS_CANDIDATES = ("tokens.txt", "tokens.txt.gz")
ENCODER_CANDIDATES = (
    "encoder.int8.onnx",
    "encoder.onnx",
    "encoder-*.int8.onnx",
    "encoder-*.onnx",
    "encoder_*.int8.onnx",
    "encoder_*.onnx",
)
DECODER_CANDIDATES = (
    "decoder.int8.onnx",
    "decoder.onnx",
    "decoder-*.int8.onnx",
    "decoder-*.onnx",
    "decoder_*.int8.onnx",
    "decoder_*.onnx",
)
JOINER_CANDIDATES = (
    "joiner.int8.onnx",
    "joiner.onnx",
    "joiner-*.int8.onnx",
    "joiner-*.onnx",
    "joiner_*.int8.onnx",
    "joiner_*.onnx",
)


def resolve_model_file(
    model_dir: Path,
    value: str | None,
    *,
    candidates: tuple[str, ...],
) -> Path | None:
    """Resolve an explicit or uniquely matched model file inside ``model_dir``."""

    root = Path(model_dir).expanduser().resolve()
    if value:
        path = (root / value).resolve()
        if root not in path.parents:
            raise FileNotFoundError(f"sherpa model file escapes model_dir: {value}")
        if not path.is_file():
            raise FileNotFoundError(f"sherpa model file does not exist: {path}")
        return path
    for candidate in candidates:
        matches = (
            sorted(path for path in root.glob(candidate) if path.is_file())
            if any(marker in candidate for marker in "*?[")
            else [root / candidate]
        )
        matches = [path for path in matches if path.is_file()]
        if len(matches) > 1:
            names = ", ".join(path.name for path in matches)
            raise ValueError(f"ambiguous sherpa model files for {candidate}: {names}")
        if matches:
            return matches[0]
    return None


@dataclass(frozen=True, slots=True)
class SherpaOnnxConfig:
    """Files and decoder settings for a local sherpa-onnx model."""

    model_dir: Path
    model_type: str = "zipformer"
    tokens: str | None = None
    encoder: str | None = None
    decoder: str | None = None
    joiner: str | None = None
    paraformer_encoder: str | None = None
    num_threads: int = 2
    provider: str = "cpu"
    sample_rate: int = 16_000
    decoding_method: str = "greedy_search"
    hotwords: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        model_dir = Path(self.model_dir).expanduser()
        if not model_dir.exists() or not model_dir.is_dir():
            raise FileNotFoundError(f"sherpa model_dir does not exist: {model_dir}")
        if self.sample_rate != 16_000:
            raise ValueError("EchoForge streaming v1 requires 16 kHz audio")
        if self.num_threads < 1 or self.num_threads > 64:
            raise ValueError("num_threads must be in [1, 64]")
        if self.provider not in {"cpu", "cuda"}:
            raise ValueError("provider must be cpu or cuda")
        if self.model_type not in {"zipformer", "paraformer", "transducer"}:
            raise ValueError("model_type must be zipformer, paraformer, or transducer")
        object.__setattr__(self, "model_dir", model_dir)

    def resolve(self, value: str | None, *, candidates: tuple[str, ...]) -> str | None:
        path = resolve_model_file(self.model_dir, value, candidates=candidates)
        return str(path) if path is not None else None


class SherpaOnnxUnavailable(RuntimeError):
    """Raised when the optional runtime/model is not available."""


class SherpaOnnxStreamingRecognizer:
    """StreamingRecognizer implementation backed by sherpa-onnx.

    ``recognizer_factory`` is useful for integration tests and for pinning a
    project-specific sherpa constructor.  It receives the imported module and
    config and must return an object exposing ``create_stream``.
    """

    model_id = "sherpa-onnx"

    def __init__(
        self,
        config: SherpaOnnxConfig,
        *,
        recognizer_factory: Callable[[Any, SherpaOnnxConfig], Any] | None = None,
    ) -> None:
        self.config = config
        self._recognizer_factory = recognizer_factory
        self._recognizer: Any | None = None
        self._stream: Any | None = None
        self._last_text = ""
        self._samples = 0
        self._hotwords = tuple(config.hotwords)

    def _load(self) -> None:
        if self._recognizer is not None:
            return
        try:
            module = importlib.import_module("sherpa_onnx")
        except (ImportError, OSError) as exc:
            raise SherpaOnnxUnavailable(
                "sherpa-onnx is not installed; install echoforge-asr[streaming]"
            ) from exc
        if self._recognizer_factory is not None:
            recognizer = self._recognizer_factory(module, self.config)
        else:
            recognizer = self._default_factory(module)
        if not hasattr(recognizer, "create_stream"):
            raise SherpaOnnxUnavailable("sherpa recognizer does not expose create_stream()")
        self._recognizer = recognizer
        hotwords = " ".join(self._hotwords) or None
        try:
            self._stream = recognizer.create_stream(hotwords=hotwords)
        except TypeError:
            self._stream = recognizer.create_stream()

    def _default_factory(self, module: Any) -> Any:
        """Build common Zipformer/Paraformer configurations across sherpa versions."""

        cfg = self.config
        tokens = cfg.resolve(cfg.tokens, candidates=TOKENS_CANDIDATES)
        if tokens is None:
            raise FileNotFoundError("could not resolve tokens.txt in model_dir")
        recognizer_cls = getattr(module, "OnlineRecognizer", None)
        if recognizer_cls is None:
            raise SherpaOnnxUnavailable("installed sherpa-onnx lacks OnlineRecognizer")
        if cfg.model_type == "paraformer":
            encoder = cfg.resolve(
                cfg.paraformer_encoder or cfg.encoder,
                candidates=ENCODER_CANDIDATES,
            )
            decoder = cfg.resolve(
                cfg.decoder,
                candidates=DECODER_CANDIDATES,
            )
            if encoder is None or decoder is None:
                raise FileNotFoundError("could not resolve Paraformer encoder/decoder in model_dir")
            return recognizer_cls.from_paraformer(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                num_threads=cfg.num_threads,
                sample_rate=cfg.sample_rate,
                feature_dim=80,
                decoding_method=cfg.decoding_method,
                provider=cfg.provider,
            )
        else:
            encoder = cfg.resolve(cfg.encoder, candidates=ENCODER_CANDIDATES)
            decoder = cfg.resolve(cfg.decoder, candidates=DECODER_CANDIDATES)
            joiner = cfg.resolve(cfg.joiner, candidates=JOINER_CANDIDATES)
            if not all((encoder, decoder, joiner)):
                raise FileNotFoundError("could not resolve transducer encoder/decoder/joiner")
            return recognizer_cls.from_transducer(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                num_threads=cfg.num_threads,
                sample_rate=cfg.sample_rate,
                feature_dim=80,
                decoding_method=cfg.decoding_method,
                provider=cfg.provider,
            )

    def _result_text(self, stream: Any) -> str:
        result = None
        if self._recognizer is not None and hasattr(self._recognizer, "get_result"):
            result = self._recognizer.get_result(stream)
        if result is None:
            result = stream.result if hasattr(stream, "result") else None
        if callable(result):
            result = result()
        if result is None:
            result = stream
        if isinstance(result, str):
            return result.strip()
        text = getattr(
            result,
            "text",
            result.get("text", "") if isinstance(result, dict) else getattr(stream, "text", ""),
        )
        return str(text or "").strip()

    def _decode_ready(self) -> None:
        assert self._recognizer is not None and self._stream is not None
        while bool(self._recognizer.is_ready(self._stream)):
            self._recognizer.decode_stream(self._stream)

    def accept_audio(self, samples: FloatAudio, sample_rate: int) -> Hypothesis | None:
        if sample_rate != self.config.sample_rate:
            raise ValueError("sherpa adapter expects 16 kHz audio")
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError("audio samples must be finite non-empty mono float32")
        self._load()
        assert self._stream is not None
        self._stream.accept_waveform(sample_rate, array)
        self._samples += int(array.size)
        self._decode_ready()
        text = self._result_text(self._stream)
        if not text or text == self._last_text:
            return None
        self._last_text = text
        return Hypothesis(
            text=text,
            stage=RevisionStage.PARTIAL,
            model_id=self.model_id,
            audio_end_ms=round(self._samples * 1000 / sample_rate),
        )

    def finalize(self) -> Hypothesis:
        self._load()
        assert self._recognizer is not None and self._stream is not None
        self._stream.input_finished()
        self._decode_ready()
        text = self._result_text(self._stream) or self._last_text
        return Hypothesis(
            text=text,
            stage=RevisionStage.STREAM_FINAL,
            model_id=self.model_id,
            audio_end_ms=round(self._samples * 1000 / self.config.sample_rate),
        )

    def reset(self) -> None:
        if self._recognizer is not None:
            hotwords = " ".join(self._hotwords) or None
            try:
                self._stream = self._recognizer.create_stream(hotwords=hotwords)
            except TypeError:
                self._stream = self._recognizer.create_stream()
        self._last_text = ""
        self._samples = 0

    def update_hotwords(self, hotwords: tuple[str, ...]) -> None:
        # sherpa versions differ in hotword APIs.  Keep the update explicit and
        # fail closed instead of pretending that a decoder accepted it.
        if not all(isinstance(word, str) and word.strip() for word in hotwords):
            raise ValueError("hotwords must be non-empty strings")
        if self._samples:
            raise SherpaOnnxUnavailable("hotwords can change only before audio is accepted")
        normalized = tuple(dict.fromkeys(word.strip() for word in hotwords))
        self._hotwords = normalized
        if self._recognizer is not None:
            try:
                self._stream = self._recognizer.create_stream(hotwords=" ".join(normalized) or None)
            except TypeError as exc:
                raise SherpaOnnxUnavailable(
                    "this sherpa build does not support hotword streams"
                ) from exc


__all__ = [
    "DECODER_CANDIDATES",
    "ENCODER_CANDIDATES",
    "JOINER_CANDIDATES",
    "TOKENS_CANDIDATES",
    "SherpaOnnxConfig",
    "SherpaOnnxStreamingRecognizer",
    "SherpaOnnxUnavailable",
    "resolve_model_file",
]
