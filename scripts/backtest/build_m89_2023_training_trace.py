#!/usr/bin/env python3
"""Build the corrected 2023 stable-primary QB trace used to train M89 synthesis.

The trace is created entirely from the corrected 2023 walk-forward component
predictions. It does not reuse the pre-M89 M88 prediction artifact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.modeling.ensemble_v2 import apply_ensemble, fit_market_weights

SEASON = 2023
MIN_ENSEMBLE_ROWS = 40


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def pkey(v) -> str:
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def load_predictions(path: Path) -> pd.DataFrame:
    x = lower(pd.read_csv(path, low_memory=False))
    required = {
        "season", "week", "team", "opponent", "player_clean_key", "market",
        "actual", "mc_proj", "ml_proj", "state_proj", "mc_expected_pass_attempts",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"M89 2023 predictions missing {sorted(missing)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[x["season"].eq(SEASON) & x["week"].between(1, 18)].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    x["team"] = x["team"].map(canon_team)
    x["opponent"] = x["opponent"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].map(pkey)
    q = x.loc[x["market"].astype(str).str.lower().eq("pass_yards")].copy()
    if q.empty or q.duplicated(["season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError("M89 2023 pass-yard prediction identity failure")
    return q


def build_oos_ensemble(q: pd.DataFrame) -> pd.DataFrame:
    out = []
    for week in sorted(q["week"].dropna().astype(int).unique()):
        target = q.loc[q["week"].eq(week)].copy()
        history = q.loc[q["week"].lt(week)].dropna(subset=["actual", "mc_proj", "ml_proj", "state_proj"])
        if len(history) >= MIN_ENSEMBLE_ROWS:
            weights = fit_market_weights(history, min_rows=MIN_ENSEMBLE_ROWS)
            applied = apply_ensemble(target, weights=weights)
        else:
            applied = apply_ensemble(target, weights=pd.DataFrame())
        out.append(applied)
    z = pd.concat(out, ignore_index=True)
    if z["ensemble_proj"].isna().any():
        raise RuntimeError("M89 2023 OOS ensemble has missing predictions")
    return z


def load_logs(path: Path) -> pd.DataFrame:
    x = lower(pd.read_csv(path, low_memory=False))
    required = {"season", "week", "team", "pass_att", "pass_yards"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"M89 2023 logs missing {sorted(missing)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[x["season"].eq(SEASON) & x["week"].between(1, 18)].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    x["team"] = x["team"].map(canon_team)
    if "player_clean_key" in x.columns:
        x["player_clean_key"] = x["player_clean_key"].map(pkey)
    elif "player" in x.columns:
        x["player_clean_key"] = x["player"].map(pkey)
    else:
        raise RuntimeError("M89 2023 logs missing player identity")
    x["pass_att_num"] = pd.to_numeric(x["pass_att"], errors="coerce").fillna(0)
    x["pass_yards_num"] = pd.to_numeric(x["pass_yards"], errors="coerce")
    return x


def stable_primary_trace(pred: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    totals = (
        logs.groupby(["week", "team"], as_index=False)["pass_att_num"]
        .sum()
        .rename(columns={"pass_att_num": "team_qb_attempts"})
    )
    lp = logs.merge(totals, on=["week", "team"], how="left")
    lp["actual_qb_attempt_share"] = lp["pass_att_num"] / lp["team_qb_attempts"].replace(0, np.nan)
    prim = (
        lp.sort_values(["week", "team", "pass_att_num"], ascending=[True, True, False])
        .drop_duplicates(["week", "team"])
        [["week", "team", "player_clean_key", "actual_qb_attempt_share"]]
        .rename(columns={"player_clean_key": "actual_primary_key"})
    )
    actual = (
        logs[["week", "team", "player_clean_key", "pass_att_num", "pass_yards_num"]]
        .drop_duplicates(["week", "team", "player_clean_key"])
        .rename(columns={"pass_att_num": "actual_attempts", "pass_yards_num": "actual_pass_yards"})
    )
    q = pred.merge(prim, on=["week", "team"], how="left", validate="many_to_one")
    q = q.loc[q["player_clean_key"].eq(q["actual_primary_key"]) & q["actual_qb_attempt_share"].ge(0.80)].copy()
    q = q.merge(actual, on=["week", "team", "player_clean_key"], how="left", validate="one_to_one")
    q["pred_attempts"] = pd.to_numeric(q["mc_expected_pass_attempts"], errors="coerce")
    q["mc_proj"] = pd.to_numeric(q["mc_proj"], errors="coerce")
    q["implied_pred_ypa"] = q["mc_proj"] / q["pred_attempts"].replace(0, np.nan)
    q["actual_ypa"] = pd.to_numeric(q["actual_pass_yards"], errors="coerce") / pd.to_numeric(q["actual_attempts"], errors="coerce").replace(0, np.nan)
    core = ["ensemble_proj", "actual_pass_yards", "actual_attempts", "pred_attempts", "implied_pred_ypa"]
    if q[core].isna().any().any():
        raise RuntimeError(f"M89 2023 training trace missing core values: {q[core].isna().sum().to_dict()}")
    if len(q) < 400:
        raise RuntimeError(f"M89 corrected 2023 stable-primary cohort unexpectedly small: {len(q)}")
    return q.sort_values(["week", "team", "player_clean_key"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--player-logs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    q = build_oos_ensemble(load_predictions(args.predictions))
    out = stable_primary_trace(q, load_logs(args.player_logs))
    out.to_csv(args.out, index=False)
    print(f"[m89_2023_trace] rows={len(out)} -> {args.out}")
    print(out[["week", "team", "player_clean_key", "actual_attempts", "pred_attempts", "actual_pass_yards", "ensemble_proj"]].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
