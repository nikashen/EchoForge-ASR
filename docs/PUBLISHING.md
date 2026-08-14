# Demo and Publishing Guide

This project has two intentionally separate runtimes:

- **Pages:** a static `DemoTransport` replay. It needs no model, API key,
  microphone permission, or hosted inference service. Mic and File controls are
  disabled by design.
- **Local API:** the FastAPI/WebSocket service started by `echoforge serve`.
  It enables real browser microphone/file streaming when the readiness probe is
  healthy. The fake backend still returns fixture text; local model paths are
  required for speech recognition.

## Review the demo

Use **Live Lab** to replay the revision chain, **Robustness Arena** to inspect a
declared fixture condition, **Session Timeline** to inspect event provenance and
export subtitles, and **Evidence** to see the runtime and claim boundary. A
Pages replay is an interactive sample, not a CER, WER, RTF, latency, or
robustness benchmark.

## Enable Pages deployment

The repository includes `.github/workflows/pages.yml`. In the repository
settings, set **Pages > Build and deployment > Source** to **GitHub Actions**.
The workflow builds `dist/pages`, uploads it as a Pages artifact, and deploys it
with the least required repository and Pages permissions. The generated
`manifest.json` records `deterministic_demo_transport` and
`metrics: not_yet_evaluated`.

## Release checklist

Before creating a release:

1. Run the CI-equivalent checks from the README and inspect the wheel contents.
2. Keep model weights, raw recordings, transcripts, credentials, and local
   manifests out of Git, source distributions, wheels, and Pages artifacts.
3. Re-check the current OpenSLR/AISHELL-1 terms before downloading or
   redistributing any evaluation data.
4. Publish CER or latency numbers only with a frozen, speaker-disjoint manifest,
   model/file hashes, normalization version, and a report that another operator
   can recompute.
5. Tag the release with the same version recorded in `pyproject.toml`; attach
   only build artifacts whose provenance was reviewed.

See [Publication boundary](PUBLICATION_BOUNDARY.md), [Evaluation protocol](EVALUATION_PROTOCOL.md),
and [Data and license protocol](DATA_PROTOCOL.md) for the evidence rules.
