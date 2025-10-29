#!/usr/bin/env python3
import os
import pandas as pd

PF = "data/player_form_consensus.csv"
OM = "data/opponent_map_from_props.csv"
OUT = PF


def safe_read_csv(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return df if len(df.columns) else None


def norm(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[^A-Za-z\-\.' ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )


pf = safe_read_csv(PF)
om = safe_read_csv(OM)
if pf is None or om is None:
    print(
        f"[merge_opponent] SKIP: missing/empty inputs (pf={pf is not None}, om={om is not None})"
    )
    raise SystemExit(0)

if "player" in pf.columns:
    pf["player_key"] = norm(pf["player"])
if "player" in om.columns:
    om["player_key"] = norm(om["player"])
else:
    om["player_key"] = ""

keys1 = [
    k
    for k in ("player_key", "team", "week")
    if k in pf.columns and k in om.columns
]
merge_cols = [
    c for c in ("player_key", "team", "week", "opponent") if c in om.columns
]
merged = pf.merge(
    om[merge_cols].drop_duplicates(),
    on=keys1,
    how="left",
    suffixes=("", "_m1"),
)
if "opponent_m1" in merged.columns:
    merged["opponent"] = merged.get("opponent").combine_first(merged["opponent_m1"])
    merged = merged.drop(columns=["opponent_m1"])

if "opponent" not in merged.columns:
    merged["opponent"] = pd.NA

na_mask = merged["opponent"].isna()
if na_mask.any():
    keys2 = [k for k in ("player_key", "week") if k in merged.columns and k in om.columns]
    merge_cols2 = [c for c in ("player_key", "week", "opponent") if c in om.columns]
    second = merged.merge(
        om[merge_cols2].drop_duplicates(),
        on=keys2,
        how="left",
        suffixes=("", "_m2"),
    )
    if "opponent_m2" in second.columns:
        second["opponent"] = second["opponent"].combine_first(second["opponent_m2"])
        second = second.drop(columns=["opponent_m2"])
    merged = second

merged.to_csv(OUT, index=False)
print(
    f"[merge_opponent] wrote {OUT} with {(merged['opponent'].notna()).sum()} mapped opponents."
)
