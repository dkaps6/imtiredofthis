from __future__ import annotations

from typing import Iterable

import pandas as pd


def coerce_merge_keys(df: pd.DataFrame, cols: Iterable[str], *, as_str: bool = True) -> pd.DataFrame:
    """
    Return a copy of df with selected columns coerced to safe merge dtypes.
    - When as_str=True, cast to pandas 'string' then strip; fills NA as "".
    - When as_str=False, cast to NumPy int64 (not pandas extension Int64Dtype).
    This prevents pandas merge from hitting Int64Dtype() coercion paths.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        if as_str:
            out[c] = out[c].astype("string").fillna("").str.strip()
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(-1).astype("int64")
    return out

