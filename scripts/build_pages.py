"""Build the deterministic GitHub Pages artifact without a bundler."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def build(output: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    source = root / "src" / "echoforge" / "web"
    if not source.joinpath("index.html").is_file():
        raise FileNotFoundError("web/index.html is missing")
    output = output.resolve()
    if root not in output.parents and output != root:
        raise ValueError("Pages output must remain inside the repository")
    if output.exists():
        shutil.rmtree(output)
    static = output / "static"
    static.mkdir(parents=True)
    index = source.joinpath("index.html").read_text(encoding="utf-8")
    index = index.replace('href="/static/', 'href="./static/').replace(
        'src="/static/', 'src="./static/'
    )
    (output / "index.html").write_text(index, encoding="utf-8")
    for name in ("style.css", "app.js", "pcm-worklet.js"):
        shutil.copy2(source / name, static / name)
    (output / ".nojekyll").write_text("", encoding="ascii")
    manifest = {
        "schema_version": "echoforge.pages/v1",
        "runtime": "deterministic_demo_transport",
        "evidence_scope": "interactive_sample",
        "metrics": "not_yet_evaluated",
        "assets": [
            "index.html",
            "static/style.css",
            "static/app.js",
            "static/pcm-worklet.js",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist/pages"))
    args = parser.parse_args()
    print(build(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
