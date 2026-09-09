#!/usr/bin/env python3
"""Fail closed if final RB rush+receiving means contradict final components."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
OUTPUTS = Path("outputs")
PRICED = OUTPUTS / "props_priced_clean.csv"
CONSERVATION = DATA / "rb_rush_rec_conservation_input_audit.json"
OUT = DATA / "rb_rush_rec_conservation_final_audit.csv"
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "vi", "vii"}


def _key(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    parts = [p for p in re.sub(r"[^a-z0-9]+", " ", text).split() if p]
    while parts and parts[-1] in SUFFIXES:
        parts.pop()
    return "".join(parts)


def main() -> int:
    if not CONSERVATION.exists() or CONSERVATION.stat().st_size == 0:
        raise RuntimeError("RB rush+receiving conservation input audit missing")
    meta = json.loads(CONSERVATION.read_text(encoding="utf-8"))
    if meta.get("disposition") != "RB_RUSH_REC_DISTRIBUTION_CONSERVED_WITH_PROMOTED_P3":
        raise RuntimeError(f"RB rush+receiving input conservation not certified: {meta.get('disposition')}")
    if meta.get("sportsbook_inputs_used") is not False:
        raise RuntimeError("RB rush+receiving conservation reports sportsbook inputs")

    priced = pd.read_csv(PRICED, low_memory=False)
    priced.columns = [str(c).strip().lower() for c in priced.columns]
    required = {"player", "team", "source_market", "side", "model_proj"}
    missing = required - set(priced.columns)
    if missing:
        raise RuntimeError(f"priced output missing conservation columns: {sorted(missing)}")
    x = priced.loc[priced["side"].astype(str).str.upper().eq("OVER")].copy()
    x["player_base_key"] = x["player"].map(_key)
    # Model mean must be invariant to sportsbook book/line for one player/market.
    spread = x.groupby(["team", "player_base_key", "source_market"])["model_proj"].nunique()
    if (spread > 1).any():
        raise RuntimeError(f"model projection varies by book/line: {spread.loc[spread>1].head(20).to_dict()}")
    one = x.drop_duplicates(["team", "player_base_key", "source_market"])
    pivot = one.pivot_table(
        index=["team", "player_base_key"],
        columns="source_market",
        values="model_proj",
        aggfunc="first",
    ).reset_index()
    need = ["player_rush_yds", "player_reception_yds", "player_rush_reception_yds"]
    present = pivot.dropna(subset=need).copy()
    if present.empty:
        raise RuntimeError("no players have all three RB conservation markets in final priced output")
    present["component_sum"] = present["player_rush_yds"] + present["player_reception_yds"]
    present["conservation_gap"] = present["player_rush_reception_yds"] - present["component_sum"]
    max_gap = float(present["conservation_gap"].abs().max())
    if not np.isfinite(max_gap) or max_gap > 1e-6:
        sample = present.loc[present["conservation_gap"].abs().gt(1e-6)].head(20).to_dict("records")
        raise RuntimeError(f"final RB rush+receiving projection conservation failed max_gap={max_gap} sample={sample}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    present.to_csv(OUT, index=False)
    print(
        "[rb_rush_rec_final] "
        + json.dumps({
            "disposition": "FINAL_RB_RUSH_REC_PROJECTIONS_CONSERVED",
            "players_checked": int(len(present)),
            "max_gap": max_gap,
        }, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
