# Evaluation Protocol v1

Metrics are authorized only for a frozen, speaker-disjoint manifest. The report
must record source URLs and licenses, every audio SHA-256, split assignment,
model revision and file hashes, package versions, normalization rules, device,
warm-up policy, and latency clock definitions.

The checked-in evaluator currently reports **normalized** Chinese CER with
substitutions, deletions, insertions, and reference length. It does not emit a
raw (pre-normalization) CER column; any raw-vs-normalized comparison must be
generated separately and recorded in the report. The available deterministic
audio primitives are fixed-seed additive noise at a requested SNR and a 16 kHz
telephone-channel band-limit/resampling transform. Gain/resampling helpers
exist in the audio package, but there is no checked-in benchmark runner that
combines them, and reverberation is not implemented. Do not present the Arena
fixture as a measured robustness result.

When a benchmark is built, robustness conditions must be paired with clean
samples and must record the transform seed and parameters. RTF is total model
compute time divided by active audio duration; first-partial and
endpoint-to-final latency are reported separately.

No threshold, decoder setting, or model choice may be tuned on the frozen Test
split. Incomplete manifests or missing sufficient statistics fail closed.

`scripts/evaluate_manifest.py` recomputes aggregate counts from row-level
reference/hypothesis pairs. It validates the schema, frozen flag, unique IDs,
and 64-character audio SHA-256 values before emitting normalized CER. It does
not independently verify every provenance field listed above, so a release
review must check those fields as well. Invalid or empty evidence produces
`status: not_yet_evaluated` and no numeric score.

Interactive VoiceLab values are single-session observations. Percentiles may
appear only in a frozen report. Endpoint-to-final latency begins when an
explicit flush/VAD endpoint is committed and ends when the final revision is
available; it is not a production SLA.
