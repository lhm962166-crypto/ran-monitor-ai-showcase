"""Small .env reader that preserves quoted secrets and Windows paths."""
from __future__ import annotations

import os
import re
from pathlib import Path


def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text[7:].lstrip()
        if "=" not in text:
            raise ValueError(f".env 第 {number} 行缺少等号。")
        key, raw = text.split("=", 1)
        key, raw = key.strip(), raw.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f".env 第 {number} 行的变量名无效。")
        if raw.startswith(("'", '"')):
            quote = raw[0]
            end = raw.find(quote, 1)
            if end < 0 or (raw[end + 1:].strip() and not raw[end + 1:].strip().startswith("#")):
                raise ValueError(f".env 第 {number} 行的引号不完整。")
            value = raw[1:end]
        else:
            value = re.split(r"\s+#", raw, maxsplit=1)[0].strip()
        values[key] = value
    return values


def load_env_file(path: Path) -> None:
    for key, value in read_env(path).items():
        os.environ.setdefault(key, value)
