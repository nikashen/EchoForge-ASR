from __future__ import annotations

import re
import unicodedata

NORMALIZER_VERSION = "echoforge.zh-normalizer/v1"
ALLOWED = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]")


def normalize_zh(text: str) -> str:
    """NFKC, lower-case, and retain CJK/alphanumerics for frozen CER v1."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(ALLOWED.findall(normalized))
