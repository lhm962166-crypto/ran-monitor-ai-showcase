"""Bound default native thread pools for local inference processes."""
from __future__ import annotations

import os


def configure_threads(limit: int = 4) -> dict[str, str]:
    """Call before importing numeric libraries; respect explicitly supplied settings."""
    if not isinstance(limit, int) or not 1 <= limit <= 32:
        raise ValueError("默认线程上限须为 1–32 的整数。")
    count = str(min(limit, os.cpu_count() or 1))
    variables = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    for name in variables:
        os.environ.setdefault(name, count)
    return {name: os.environ[name] for name in variables}
