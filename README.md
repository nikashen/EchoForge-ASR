# EchoForge-ASR

Evidence-first Chinese streaming ASR laboratory with a real PCM WebSocket
transport, deterministic VAD, dual-pass transcript repair, controlled audio
degradation primitives, and an auditable VoiceLab UI.

**Live demo:** https://nikashen.github.io/EchoForge-ASR/

The Pages build is an explicit deterministic replay. It demonstrates the UI,
protocol states, self-healing diff, and evidence boundary; it does not run a
speech model or claim CER, RTF, production latency, accent coverage, or business
impact.

On Pages, only the Demo source is enabled. Mic and File become available when
the same UI is served by the local API and its readiness probe succeeds. The
repository currently publishes no frozen benchmark report or speech-quality
number.

## What is implemented

- `EFA1` binary WebSocket frames: ordered 16 kHz mono PCM16LE with strict size,
  sequence, duplicate, generation, and state checks.
- Per-session bounded queue, single consumer, heartbeat, backpressure, explicit
  Origin/subprotocol policy, close-code mapping, and cleanup on disconnect.
- Deterministic energy VAD with double-threshold hysteresis, speech start/end
  debounce, state versions, and chunk-boundary invariance.
- Monotonic transcript revisions:
  `partial -> stream_final -> dual_pass_final`, or explicit
  `stream_only / VERIFIER_FAILED` degradation.
- Optional lazy adapters for `sherpa-onnx==1.13.5` streaming recognition and
  `faster-whisper==1.2.1` endpoint verification. Importing EchoForge never
  downloads weights or opens a device.
- Browser Demo, Mic, and File sources; AudioWorklet capture, 16 kHz resampling,
  live hotword messages, waveform/VAD telemetry, repair diff, timeline, and
  JSON/SRT/VTT export.
- Controlled fixed-seed noise/SNR and telephone-channel transforms, Chinese
  normalization, edit counts, and fail-closed frozen-manifest evaluation.
- Responsive 1440 px, 390 px, and 320 px layouts with deterministic Pages
  transport separated from the local API transport.

## Architecture

```text
Browser AudioWorklet / decoded file
              |
              | EFA1 PCM16LE frames
              v
       WebSocket /api/v1/stream
              |
       bounded single-consumer queue
              |
      deterministic VAD + session fences
          /                     \
 sherpa-onnx streaming     endpoint verifier
      partial/final         faster-whisper
          \                     /
       revision log + character diff
              |
      VoiceLab / timeline / evidence
```

## Demo tour

1. Open the [Pages demo](https://nikashen.github.io/EchoForge-ASR/) and choose
   **Live Lab > Start** to replay the fixed partial, streaming-final, and
   dual-pass-final revision chain.
2. Open **Robustness Arena** to inspect a declared fixture condition. The
   displayed CER stays `N/A` until a reference-backed frozen evaluation exists.
3. Open **Session Timeline** to inspect event provenance and export JSON, SRT,
   or VTT. **Evidence** shows the runtime and publication boundary.

The replay is intentionally useful for reviewing protocol behavior and UI
states, not for measuring an ASR model. Hotword controls are capability
negotiated: the deterministic backend reports when an override is not applied,
and a live sherpa session can only accept updates before audio is consumed.

## Quick start

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,serve]"
.\.venv\Scripts\echoforge.exe smoke --json
.\.venv\Scripts\echoforge.exe serve --backend fake --port 8090
```

Open http://127.0.0.1:8090/. The local fake backend accepts real microphone and
file audio (browser permission and a localhost/127.0.0.1 or HTTPS secure
context are required) but returns a declared deterministic transcript fixture. This
proves the transport and state machine without presenting fixture output as ASR
quality.

## Real local models

Install optional runtimes and point the server at operator-controlled local
model directories:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[serve,streaming,verifier]"
.\.venv\Scripts\echoforge.exe serve `
  --backend sherpa-onnx `
  --model-dir D:\models\sherpa-zipformer `
  --verifier-model D:\models\faster-whisper-small `
  --provider cpu `
  --port 8090
```

Run the fail-closed prerequisite check before starting a real backend. It
checks the model files and optional runtime imports without loading weights:

```powershell
.\.venv\Scripts\echoforge.exe preflight `
  --backend sherpa-onnx `
  --model-dir D:\models\sherpa-zipformer `
  --verifier-model D:\models\faster-whisper-small `
  --json
```

Use `--no-dual-pass` when the endpoint verifier is intentionally disabled.
`preflight` returning `PASSED` means only that local paths, required model
artifacts, and imports satisfy the static contract; it is not a speech-quality
benchmark and does not replace a short end-to-end audio probe. JSON output
records `verification_level=static` and `model_load_verified=false`. A CUDA
selection emits a `cuda_compatibility` check with `status=not_verified` (and
`ok=null`) instead of a misleading green check until that probe succeeds.
The readiness endpoint keeps service lifecycle separate: for a real backend,
`verification_level=static` and
`model_verification_status=static_requirements_passed_model_unverified` still
mean no model load has been observed. The legacy `model_status` field remains
for API compatibility. Only successful streaming inference changes the
streaming-specific load flag; it does not prove endpoint-verifier load or ASR
quality.

The local faster-whisper contract requires `model.bin`, `config.json`,
`tokenizer.json`, and one of `vocabulary.json` / `vocabulary.txt`. With the
default no-download policy, the adapter also passes `local_files_only=True` to
faster-whisper so a missing artifact cannot silently trigger network access.

CPU is the default. GPU use is opt-in and depends on the installed sherpa wheel,
CUDA/cuDNN compatibility, and available VRAM. `--provider cuda` selects the
sherpa streaming runtime; the current faster-whisper endpoint verifier is
constructed on CPU. Model weights remain outside Git, Releases, and browser
requests.

On Windows, the verifier extra pins `ctranslate2==4.5.0`. Versions 4.6.3 and
4.8.1 caused a reproducible native access violation while loading the pinned
`faster-whisper-small` CTranslate2 model on the validated Windows host; static
preflight rejects a different Windows version instead of allowing the service
to crash during the first endpoint revision.

## Evaluation

The evaluation commands are repository workflows run from a source checkout.
Review the current OpenSLR terms before downloading AISHELL-1. A dry-run writes
only a plan; a real download writes hash-bound receipts and a completed local
manifest:

```powershell
python scripts/download_aishell.py `
  --accept-license --dry-run `
  --output F:\ASR\data\aishell1-download
python scripts/download_aishell.py `
  --accept-license `
  --output F:\ASR\data\aishell1-download
```

OpenSLR's audio archive contains nested per-speaker archives. Download-time
generic extraction is disabled. The dedicated extractor verifies both tar
layers, rejects links/devices/traversal, checks every WAV, and atomically
publishes a hash-bound tree. This limited command is suitable for an
**unscored compatibility smoke only**:

```powershell
python scripts/extract_aishell.py `
  --output F:\ASR\data\aishell1-dev-smoke `
  --download-manifest F:\ASR\data\aishell1-download\download_manifest.json `
  --audio-archive F:\ASR\data\aishell1-download\audio.tgz `
  --resources-archive F:\ASR\data\aishell1-download\resources.tgz `
  --split dev `
  --speaker-limit-per-split 1
python scripts/prepare_aishell_manifest.py `
  --output F:\ASR\manifests\aishell-dev-smoke-prepared.json `
  --extraction-root F:\ASR\data\aishell1-dev-smoke `
  --wav-root F:\ASR\data\aishell1-dev-smoke\wav `
  --transcript F:\ASR\data\aishell1-dev-smoke\transcript\aishell_transcript_v0.8.txt `
  --download-manifest F:\ASR\data\aishell1-download\download_manifest.json `
  --split dev `
  --speaker-limit 1 `
  --utterances-per-speaker 1
```

Preparation verifies the extraction marker before and after scanning the tree,
uses the same bytes for WAV/transcript hashes, and emits an unfrozen manifest.
The runner checks model artifacts before loading and after inference, rejects
truncated audio and verifier degradation, and never overwrites evidence:

```powershell
python scripts/run_manifest_asr.py F:\ASR\manifests\aishell-dev-smoke-prepared.json `
  --output F:\ASR\manifests\aishell-dev-smoke-unscored.json `
  --audio-root F:\ASR\data\aishell1-dev-smoke\wav `
  --backend sherpa-onnx `
  --model-dir F:\ASR\models\sherpa-streaming-zh `
  --verifier-model F:\ASR\models\faster-whisper-small `
  --streaming-revision sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30 `
  --verifier-revision Systran/faster-whisper-small@536b0662 `
  --warmup-audio F:\ASR\smoke\warmup.wav
python scripts/evaluate_manifest.py F:\ASR\manifests\aishell-dev-smoke-unscored.json
```

That evaluator invocation intentionally exits `2` with
`status=not_yet_evaluated`: limited extraction/selection is never authorized.
For an authorized benchmark, extract one complete dev or test split without a
speaker limit, prepare it without row limits using `--authorize-evaluation`
and a pre-registered `--protocol-id`, and use separate warm-up audio. The
runner also requires explicit reviewed model-weight provenance for every
active model:

```text
--streaming-source-url <immutable HTTPS source>
--streaming-license <reviewed weight license>
--confirm-streaming-license-reviewed
--verifier-source-url <immutable HTTPS source>
--verifier-license <reviewed weight license>
--confirm-verifier-license-reviewed
```

Authorization currently supports CPU evidence only. Do not pass review flags
unless the weight licenses—not merely runtime software licenses—were
independently checked. The locally validated sherpa archive has no separate
weight-license file, so this repository keeps its real-model runs
`evaluation_authorized=false` and publishes no CER, RTF, or latency number.

An authorized report includes normalized CER sufficient statistics plus
sanitized dataset/model/artifact/package/device/warm-up/timing provenance; it
never includes row text or audio paths. Runner timing is offline, unpaced,
sequential `perf_counter` wall time. `first_partial_wall_ms` is not real-time
playback TTFT, and the resulting RTF/latencies are not production SLAs.

The example evaluator returns `not_yet_evaluated` because it is intentionally
unfrozen and contains no rows. A report is emitted only when IDs, audio hashes,
references, hypotheses, and the frozen flag all validate.

The evaluator exits with code `0` for an evaluated report and code `2` for the
intentional `not_yet_evaluated` result; the latter is a fail-closed state, not a
passing benchmark.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m mypy src/echoforge
.\.venv\Scripts\python.exe scripts/build_pages.py --output dist/pages
node --check src/echoforge/web/app.js
node --check src/echoforge/web/pcm-worklet.js
.\.venv\Scripts\python.exe -m build
```

The current local verification is **264 tests passed**. CI covers Linux Python
3.10/3.11/3.12, a Windows verifier/CT2 4.5.0 contract job, wheel imports, Pages,
and frontend syntax without downloading model weights or evaluation audio.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [WebSocket protocol](docs/WEBSOCKET_PROTOCOL.md)
- [Evaluation protocol](docs/EVALUATION_PROTOCOL.md)
- [Data and license protocol](docs/DATA_PROTOCOL.md)
- [Security and resource limits](docs/SECURITY.md)
- [Publication boundary](docs/PUBLICATION_BOUNDARY.md)
- [Demo and publishing guide](docs/PUBLISHING.md)
- [中文简历与面试案例](docs/RESUME_PROJECT_ZH.md)
