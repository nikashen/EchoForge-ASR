# Architecture

EchoForge keeps the deterministic protocol core independent of FastAPI, model
packages, and browser code. A model adapter is injected at application startup;
importing the package never downloads weights or opens a device.

## Boundaries

1. `contracts`: strict wire models and binary framing.
2. `audio` / `vad`: finite PCM validation, accounting, and deterministic VAD.
3. `asr`: streaming and endpoint verifier protocols plus optional adapters.
4. `cascade`: revision history, stable prefixes, and repair diff.
5. `streaming`: per-session sequencing, bounded buffers, and failure states.
6. `evaluation`: text normalization, CER/edit counts, perturbations, and reports.
7. `api`: REST health and WebSocket transport only.
8. `web`: explicit local WebSocket transport or deterministic Pages demo transport.

The browser has two explicit transports. `WebSocketTransport` captures local
microphone or decoded file audio and sends real `EFA1` frames. `DemoTransport`
replays fixed protocol events for GitHub Pages and never presents itself as a
speech recognizer.

The session is fail-closed after a streaming backend failure because a stateful
decoder cannot be rolled back safely. A verifier failure preserves the already
confirmed streaming transcript and emits a degraded final event instead of
claiming that a repair succeeded. Model adapters are lazy and accept only
operator-supplied local paths; client messages cannot select checkpoints.
