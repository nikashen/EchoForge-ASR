# Data and License Protocol

Raw recordings and model weights stay outside Git and Releases. The opt-in
downloader records the source URL, archive SHA-256, license-page hash, byte
counts, and receipt/completion status in a local manifest. Speaker-disjoint policy,
model provenance, and evaluation-tool versions belong in the separate frozen
evaluation manifest; they are not silently inferred by the downloader.

The first planned public benchmark is AISHELL-1 from OpenSLR 33. The repository
will contain only the downloader, a schema, and aggregate evidence after the
license and redistribution terms are independently checked. No row-level
recording or transcript is copied into the public repository.

User microphone audio is ephemeral by default. Logs contain session IDs,
durations, counters, and error codes, never raw audio or transcript text.

`scripts/download_aishell.py` requires `--accept-license`, supports a no-network
`--dry-run` plan that does not block a later real manifest. Archives stream to
an operator-selected local directory through resumable `.part` files whose
sidecars pin URL, length, ETag/Last-Modified and are sent back with conditional
range requests. The downloader strictly checks Content-Range boundaries,
fsyncs content, and atomically promotes only a complete regular file; links and
non-regular state files are rejected. Promotion retains the temporary validator
until a durable sibling `*.download.json` receipt binds URL, length,
ETag/Last-Modified and a stable local size/SHA-256. A completed archive without
that receipt (or a matching legacy `.part.json` that can be migrated) is never
silently associated with the current upstream object. The downloader does not
extract AISHELL: `--extract` is disabled because the upstream audio tar contains
nested per-speaker archives. Plan, receipt, and final manifest JSON are strict,
fsynced, and atomically published without overwrite. The OpenSLR page
describes Apache-2.0 terms and academic use; operators must re-check the current
upstream page before commercial use or redistribution.

Completed downloads created by the historical resumable downloader before
durable receipts were introduced require an explicit migration; the normal
download path remains fail-closed. After reviewing the current license, run:

```console
python scripts/download_aishell.py --accept-license \
  --migrate-legacy-manifest /evidence/old/download_manifest.json \
  --archive-root /datasets/aishell1 \
  --migration-output /evidence/new/download_manifest.json
```

This mode accepts only the historical `echoforge.download/v1` form in which
`dry_run` is absent. It requires complete audio/resources records with the
official URLs, canonical hashes, positive sizes, and both ETag and
Last-Modified validators. It ignores the recorded archive paths and reads only
regular `audio.tgz` and `resources.tgz` files directly under the explicit
archive root. Before writing anything, it matches both records against current
HEAD validators/length and stable local size/SHA-256 evidence. A matching
legacy `audio.tgz.part.json` is accepted and retired after receipt publication;
the complete legacy manifest is the only supported way to attest a historical
`resources.tgz` that has no sidecar.

Migration never edits or replaces the source manifest. It atomically publishes
no-overwrite `audio.tgz.download.json` and `resources.tgz.download.json`
receipts that carry the source-manifest SHA-256, then atomically publishes a
separate completed manifest with explicit `dry_run: false` and
`echoforge.download-migration/v1` provenance. Matching temporary evidence can
be recovered after interruption; conflicting receipts, temporary files,
validators, hashes, links, or an existing output fail closed. Migration cannot
be combined with `--dry-run` or `--extract` and downloads no archive payload.

Use `scripts/extract_aishell.py` as the single extraction entry point. It
verifies the completed download manifest and both stable archive hashes, streams
the outer audio tar and every nested speaker tar, rejects traversal, links,
devices, duplicate paths and unexpected split/speaker layouts, and validates
all selected WAV files as complete 16 kHz mono PCM16LE. It publishes only dev
and/or test through a controlled staging directory, then writes a strict
hash-bound marker containing the download-manifest hash, archive evidence,
selected splits/speakers, transcript hash and complete tree inventory hash:

```powershell
python scripts/extract_aishell.py `
  --output F:\ASR\data\aishell1-extracted `
  --download-manifest F:\ASR\data\aishell1-download\download_manifest.json `
  --audio-archive F:\ASR\data\aishell1-download\audio.tgz `
  --resources-archive F:\ASR\data\aishell1-download\resources.tgz `
  --split dev
```

`--speaker-limit-per-split` is permitted only for an unscored compatibility
smoke. Existing completed trees are reused only after recomputing the tree and
matching the marker; incomplete controlled staging is rebuilt, and untrusted
existing output is never overwritten.

`scripts/prepare_aishell_manifest.py` reads only explicitly supplied extracted
WAV/transcript paths, the extraction root, and a completed download manifest.
It validates the marker before and after scanning to close preparation-time
TOCTOU gaps, and requires the supplied WAV/transcript paths to be inside that
root. It deterministically
selects sorted dev/test speakers and utterances, rejects cross-split speakers,
rejects linked or out-of-root speaker/audio paths, validates PCM format, and
records every selected audio hash plus the transcript and download-manifest
hashes and the marker/inventory hashes. The completed download evidence must
contain exactly the official
OpenSLR audio and resource URLs, positive byte counts, and canonical SHA-256
values; dry-run plans are rejected. Prepared manifests are unscored by default.
Authorization requires a full, unlimited extraction of exactly one dev or test
split, no preparation row limits, an explicit canonical protocol ID, and
verified extraction evidence. Limited trees/manifests remain permanently
unscored.
