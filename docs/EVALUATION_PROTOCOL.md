# Evaluation Protocol v1

Metrics are authorized only for a frozen, speaker-disjoint manifest derived
from a hash-bound extraction of one complete dev or test split. Authorized
preparation forbids extraction, speaker, and utterance limits and requires a
canonical protocol ID. The runner additionally requires CPU execution,
separate warm-up audio, immutable model revisions, artifact hashes, and an
explicitly reviewed weight-license/source record for every active model.

`scripts/run_manifest_asr.py` consumes an explicitly unfrozen local manifest
and atomically creates a no-overwrite result. Before inference it validates the
dataset/extraction selection, canonical IDs and relative paths, duplicate audio
hashes, normalized references, strict complete 16 kHz mono PCM16LE decoding,
model files, and provenance. It hashes model artifacts before adapter creation
and again after inference. A model mutation, verifier degradation, truncated
WAV, ambiguous JSON, or evidence publication race aborts the run.

The checked-in evaluator independently recomputes **normalized** Chinese CER
substitutions, deletions, insertions, reference length, and hypothesis length.
It validates authorization/frozen state; exact normalization version; dataset,
extraction and model provenance; real-backend whitelist; decoder settings;
artifact labels, plausible sizes and hashes; package/device/warm-up/clock
evidence; final-stage/text consistency; per-row timing arithmetic; and runtime
totals. Invalid, incomplete, limited, fake, or unauthorized evidence returns
`status: not_yet_evaluated` with no numeric score.

An evaluated aggregate report contains CER sufficient statistics and sanitized
dataset/model/artifact/package/device/warm-up/timing evidence plus the frozen
manifest hash. It excludes row text and audio paths. The private frozen
manifest retains every row ID, speaker/split, audio SHA-256, reference,
hypothesis, revision stage and timing needed for independent recomputation.

Runner timing is offline, unpaced, sequential `time.perf_counter` wall time.
`utterance_rtf` is sequential ASR processing wall time divided by decoded audio
duration. `first_partial_wall_ms` starts at the first immediate decoder feed and
is **not** real-time playback TTFT. Endpoint-to-final starts immediately before
streaming finalize and includes the optional endpoint verifier. These local
observations are not production latency or SLA claims.

The evaluator emits normalized CER only; raw-vs-normalized comparisons must be
generated separately. Fixed-seed additive noise/SNR and telephone-channel
primitives exist, but the current manifest runner accepts clean WAV input and
does not yet apply those perturbations or reverberation. Robustness conditions
must eventually be paired with clean samples and record transform seeds and
parameters. Do not present the Arena fixture as a measured result.

No threshold, decoder setting, or model choice may be tuned on a frozen test
split. Software licenses do not imply model-weight licenses: if the actual
weights lack independently reviewed terms, keep the run unauthorized and do
not publish CER, RTF, latency, robustness, accent, or production claims.
