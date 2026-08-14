# Data and License Protocol

Raw recordings and model weights stay outside Git and Releases. The opt-in
downloader records the source URL, archive SHA-256, license-page hash, byte
counts, and extraction status in a local manifest. Speaker-disjoint policy,
model provenance, and evaluation-tool versions belong in the separate frozen
evaluation manifest; they are not silently inferred by the downloader.

The first planned public benchmark is AISHELL-1 from OpenSLR 33. The repository
will contain only the downloader, a schema, and aggregate evidence after the
license and redistribution terms are independently checked. No row-level
recording or transcript is copied into the public repository.

User microphone audio is ephemeral by default. Logs contain session IDs,
durations, counters, and error codes, never raw audio or transcript text.

`scripts/download_aishell.py` requires `--accept-license`, supports a no-network
`--dry-run`, streams archives to an operator-selected local directory, records
archive and license-page SHA-256 values plus byte counts, rejects archive path
traversal, links, and device files, and never adds the data to Git. The OpenSLR
page describes Apache-2.0 terms and academic use; operators must re-check the
current upstream page before commercial use or redistribution.
