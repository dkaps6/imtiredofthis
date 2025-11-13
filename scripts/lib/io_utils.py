from __future__ import annotations

import os

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_concat(to_path: str, df: pd.DataFrame, index: bool = False) -> None:
    if os.path.exists(to_path):
        prev = pd.read_csv(to_path)
        out = pd.concat([prev, df], ignore_index=True)
    else:
        out = df
    out.to_csv(to_path, index=index)


def write_atomic(path: str, df: pd.DataFrame) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
