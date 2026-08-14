from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from echoforge.asr.base import EndpointFinalizer, StreamingRecognizer
from echoforge.audio.ring_buffer import AudioRingBuffer
from echoforge.cascade.revisions import RevisionLog
from echoforge.contracts.domain import Hypothesis, RevisionStage, TranscriptRevision
from echoforge.contracts.enums import StringEnum
from echoforge.contracts.errors import (
    BackendFailureError,
    GenerationConflictError,
    ResourceLimitError,
    SequenceConflictError,
    SessionStateError,
)


class SessionState(StringEnum):
    READY = "ready"
    LISTENING = "listening"
    DRAINING = "draining"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class IngestResult:
    highest_contiguous_sequence: int
    queued_ms: int
    duplicate: bool
    revision: TranscriptRevision | None


@dataclass(frozen=True, slots=True)
class FlushResult:
    utterance_id: str
    revisions: tuple[TranscriptRevision, ...]
    verifier_degraded: bool


@dataclass(frozen=True, slots=True)
class ResetResult:
    reset_id: str
    generation: int
    state_version: int
    removed_samples: int
    idempotent_replay: bool


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    session_id: str
    state: SessionState
    generation: int
    state_version: int
    next_audio_sequence: int
    total_samples: int
    buffered_samples: int
    completed_utterances: int
    failed_code: str | None


class StreamingSession:
    """Thread-safe protocol state for one ordered PCM stream."""

    def __init__(
        self,
        session_id: str,
        recognizer: StreamingRecognizer,
        finalizer: EndpointFinalizer | None = None,
        *,
        sample_rate: int = 16_000,
        max_utterance_seconds: int = 30,
        max_session_seconds: int = 15 * 60,
        fingerprint_capacity: int = 4_096,
        reset_cache_capacity: int = 128,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not session_id or len(session_id) > 96:
            raise ValueError("session_id must be non-empty and at most 96 characters")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if sample_rate != 16_000:
            raise ValueError("streaming v1 requires 16 kHz")
        if (
            isinstance(max_utterance_seconds, bool)
            or not isinstance(max_utterance_seconds, int)
            or isinstance(max_session_seconds, bool)
            or not isinstance(max_session_seconds, int)
            or max_utterance_seconds < 1
            or max_session_seconds < max_utterance_seconds
        ):
            raise ValueError("session duration limits are invalid")
        if isinstance(fingerprint_capacity, bool) or fingerprint_capacity < 1:
            raise ValueError("fingerprint_capacity must be positive")
        if isinstance(reset_cache_capacity, bool) or reset_cache_capacity < 1:
            raise ValueError("reset_cache_capacity must be positive")
        self.session_id = session_id
        self.recognizer = recognizer
        self.finalizer = finalizer
        self.sample_rate = sample_rate
        self.max_utterance_samples = max_utterance_seconds * sample_rate
        self.max_session_samples = max_session_seconds * sample_rate
        self._fingerprint_capacity = fingerprint_capacity
        self._reset_cache_capacity = reset_cache_capacity
        self._clock = clock
        self._lock = threading.RLock()
        self._audio = AudioRingBuffer(self.max_utterance_samples)
        self._fingerprints: OrderedDict[int, str] = OrderedDict()
        self._reset_cache: OrderedDict[str, ResetResult] = OrderedDict()
        self._completed: list[tuple[TranscriptRevision, ...]] = []
        self._utterance_index = 0
        self._log = self._new_revision_log()
        self.state = SessionState.READY
        self.generation = 0
        self.state_version = 0
        self.next_audio_sequence = 0
        self.total_samples = 0
        self.failed_code: str | None = None

    def _new_revision_log(self) -> RevisionLog:
        return RevisionLog(self.session_id, f"utt-{self._utterance_index:06d}")

    def snapshot(self) -> SessionSnapshot:
        with self._lock:
            return SessionSnapshot(
                session_id=self.session_id,
                state=self.state,
                generation=self.generation,
                state_version=self.state_version,
                next_audio_sequence=self.next_audio_sequence,
                total_samples=self.total_samples,
                buffered_samples=len(self._audio),
                completed_utterances=len(self._completed),
                failed_code=self.failed_code,
            )

    @staticmethod
    def _fingerprint(samples: NDArray[np.float32]) -> str:
        return hashlib.sha256(np.ascontiguousarray(samples).tobytes()).hexdigest()

    def _remember_fingerprint(self, sequence: int, fingerprint: str) -> None:
        self._fingerprints[sequence] = fingerprint
        while len(self._fingerprints) > self._fingerprint_capacity:
            self._fingerprints.popitem(last=False)

    def ingest(self, sequence: int, samples: NDArray[np.float32]) -> IngestResult:
        with self._lock:
            if self.state not in {SessionState.READY, SessionState.LISTENING}:
                raise SessionStateError(f"cannot ingest audio while session is {self.state}")
            if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
                raise SequenceConflictError("audio sequence must be a non-negative integer")
            array = np.asarray(samples, dtype=np.float32)
            if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
                raise ValueError("audio samples must be finite non-empty mono float32")
            fingerprint = self._fingerprint(array)
            if sequence < self.next_audio_sequence:
                previous = self._fingerprints.get(sequence)
                if previous is None or previous != fingerprint:
                    raise SequenceConflictError(
                        "audio sequence was already committed with a different payload"
                    )
                return IngestResult(
                    highest_contiguous_sequence=self.next_audio_sequence - 1,
                    queued_ms=round(len(self._audio) * 1000 / self.sample_rate),
                    duplicate=True,
                    revision=None,
                )
            if sequence > self.next_audio_sequence:
                raise SequenceConflictError(
                    f"audio sequence gap: expected {self.next_audio_sequence}, received {sequence}"
                )
            if self.total_samples + array.size > self.max_session_samples:
                raise ResourceLimitError("session audio duration limit exceeded")
            if len(self._audio) + array.size > self.max_utterance_samples:
                raise ResourceLimitError("utterance audio duration limit exceeded; flush first")

            try:
                hypothesis = self.recognizer.accept_audio(array, self.sample_rate)
                if hypothesis is not None and hypothesis.stage is not RevisionStage.PARTIAL:
                    raise ValueError("streaming accept_audio must emit only partial hypotheses")
                revision = self._log.append(hypothesis, lane="streaming") if hypothesis else None
            except Exception as exc:
                self.state = SessionState.FAILED
                self.failed_code = "STREAMING_BACKEND_FAILED"
                self.state_version += 1
                raise BackendFailureError(
                    "streaming backend failed; session is fail-closed"
                ) from exc

            self._audio.append(array)
            self._remember_fingerprint(sequence, fingerprint)
            self.next_audio_sequence += 1
            self.total_samples += array.size
            self.state = SessionState.LISTENING
            self.state_version += 1
            return IngestResult(
                highest_contiguous_sequence=sequence,
                queued_ms=round(len(self._audio) * 1000 / self.sample_rate),
                duplicate=False,
                revision=revision,
            )

    def flush(self, *, expected_generation: int | None = None) -> FlushResult:
        with self._lock:
            if expected_generation is not None and expected_generation != self.generation:
                raise GenerationConflictError("stale session generation")
            if self.state not in {SessionState.READY, SessionState.LISTENING}:
                raise SessionStateError(f"cannot flush while session is {self.state}")
            if len(self._audio) == 0:
                raise SessionStateError("cannot flush an empty utterance")
            self.state = SessionState.DRAINING
            audio = self._audio.to_array()
            try:
                stream_hypothesis = self.recognizer.finalize()
                if stream_hypothesis.stage is not RevisionStage.STREAM_FINAL:
                    raise ValueError("streaming finalize must emit a stream_final hypothesis")
                stream_revision = self._log.append(stream_hypothesis, lane="streaming")
            except Exception as exc:
                self.state = SessionState.FAILED
                self.failed_code = "STREAMING_FINALIZE_FAILED"
                self.state_version += 1
                raise BackendFailureError(
                    "streaming finalization failed; session is fail-closed"
                ) from exc

            degraded = False
            if self.finalizer is not None:
                started = self._clock()
                try:
                    verified = self.finalizer.transcribe(audio, self.sample_rate)
                    elapsed_ms = (self._clock() - started) * 1000
                    verified = Hypothesis(
                        text=verified.text,
                        stage=RevisionStage.DUAL_PASS_FINAL,
                        model_id=verified.model_id,
                        decoder_score=verified.decoder_score,
                        audio_start_ms=verified.audio_start_ms,
                        audio_end_ms=verified.audio_end_ms,
                        degraded=verified.degraded,
                        degradation_code=verified.degradation_code,
                    )
                    self._log.append(
                        verified,
                        lane="verifier",
                        server_compute_ms=elapsed_ms,
                        endpoint_to_final_ms=elapsed_ms,
                    )
                    degraded = verified.degraded
                except Exception:  # noqa: BLE001 - verifier failures degrade to stream-only
                    degraded = True
                    fallback = Hypothesis(
                        text=stream_revision.text,
                        stage=RevisionStage.STREAM_ONLY,
                        model_id=stream_revision.model_id,
                        audio_start_ms=stream_revision.audio_start_ms,
                        audio_end_ms=stream_revision.audio_end_ms,
                        degraded=True,
                        degradation_code="VERIFIER_FAILED",
                    )
                    self._log.append(fallback, lane="streaming")

            revisions = self._log.items
            utterance_id = self._log.utterance_id
            self._completed.append(revisions)
            self._audio.clear()
            self.recognizer.reset()
            self._utterance_index += 1
            self._log = self._new_revision_log()
            self.state = SessionState.READY
            self.state_version += 1
            return FlushResult(
                utterance_id=utterance_id,
                revisions=revisions,
                verifier_degraded=degraded,
            )

    def reset(
        self,
        reset_id: str,
        *,
        expected_state_version: int,
        expected_generation: int,
    ) -> ResetResult:
        with self._lock:
            cached = self._reset_cache.get(reset_id)
            if cached is not None:
                return ResetResult(
                    reset_id=cached.reset_id,
                    generation=cached.generation,
                    state_version=cached.state_version,
                    removed_samples=cached.removed_samples,
                    idempotent_replay=True,
                )
            if expected_generation != self.generation:
                raise GenerationConflictError("stale session generation")
            if expected_state_version != self.state_version:
                raise GenerationConflictError("stale session state version")
            if not reset_id or len(reset_id) > 96:
                raise ValueError("reset_id must be non-empty and at most 96 characters")
            removed = len(self._audio)
            self._audio.clear()
            self._fingerprints.clear()
            self._completed.clear()
            self.recognizer.reset()
            self._utterance_index = 0
            self._log = self._new_revision_log()
            self.next_audio_sequence = 0
            self.total_samples = 0
            self.failed_code = None
            self.generation += 1
            self.state_version += 1
            self.state = SessionState.READY
            result = ResetResult(
                reset_id=reset_id,
                generation=self.generation,
                state_version=self.state_version,
                removed_samples=removed,
                idempotent_replay=False,
            )
            self._reset_cache[reset_id] = result
            while len(self._reset_cache) > self._reset_cache_capacity:
                self._reset_cache.popitem(last=False)
            return result

    def stop(self) -> tuple[TranscriptRevision, ...]:
        with self._lock:
            if self.state == SessionState.CLOSED:
                return ()
            if self.state == SessionState.FAILED:
                self.state = SessionState.CLOSED
                self.state_version += 1
                return ()
            revisions: tuple[TranscriptRevision, ...] = ()
            if len(self._audio):
                revisions = self.flush(expected_generation=self.generation).revisions
            self.state = SessionState.CLOSED
            self.state_version += 1
            return revisions
