"""Safe spreadsheet-friendly exports for user supplied labels."""
from __future__ import annotations

import pandas as pd


FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Return UTF-8 CSV while preventing text cells from becoming spreadsheet formulas."""
    safe = frame.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(
            lambda value: "'" + value
            if isinstance(value, str) and value.lstrip().startswith(FORMULA_PREFIXES)
            else value
        )
    return safe.to_csv(index=False).encode("utf-8-sig")
