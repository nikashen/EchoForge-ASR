# Publication Boundary

The deterministic backend and Pages fixture establish protocol and UI behavior
only. They do not establish speech-recognition accuracy, robustness, latency
SLA, accent coverage, or business impact.

At the current repository snapshot, the Pages manifest is explicitly marked
`metrics: not_yet_evaluated`, and no frozen evaluation report is published.
Single-session timings or deterministic fixture timings shown in the UI are
observations of that runtime, not model or production-service benchmarks.

Real metrics may be published only when an independent verifier can recompute
them from the frozen manifest, model hashes, normalization rules, and report.
Unfinished human review, missing public data, or a failed artifact check remains
visible as a limitation.
