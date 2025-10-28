#!/usr/bin/env python3
import os
import pandas as pd
from typing import List

def ensure_header(path: str, columns: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        pd.DataFrame(columns=columns).to_csv(path, index=False)

def write_safely(df: pd.DataFrame, path: str, columns_if_empty: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if df is None or df.empty:
        pd.DataFrame(columns=columns_if_empty).to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)
