# Local Evaluation Data

This directory is intentionally empty in Git. Use the opt-in downloader to
place AISHELL-1 outside the repository's tracked files and record archive
hashes in a local manifest:

```powershell
.\.venv\Scripts\python.exe scripts/download_aishell.py --accept-license --output .cache/aishell1 --dry-run
```

Run the command from the repository's activated/installed virtual environment;
the script writes only to the operator-selected local cache.

The downloader never uploads or publishes recordings. Confirm the current
OpenSLR license and redistribution terms before a real download.
