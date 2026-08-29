#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.backtest.validate_qb_possession_dropback_generative as m


def prepare_pbp_fixed(raw: pd.DataFrame) -> pd.DataFrame:
    """Use nflverse's corrected drive identifier when available.

    The original M64 implementation preferred raw `drive` over `fixed_drive`.
    For the authoritative rerun, preserve the frozen model but correct the
    possession identifier before the original preparation logic executes.
    """
    x = raw.copy()
    if "fixed_drive" in x.columns:
        if "drive" in x.columns:
            x["fixed_drive"] = x["fixed_drive"].where(x["fixed_drive"].notna(), x["drive"])
            x = x.drop(columns=["drive"])
    return _ORIG_PREPARE_PBP(x)


def frozen_verdict_fixed(att: pd.DataFrame, pas: pd.DataFrame, weeks: pd.DataFrame) -> pd.DataFrame:
    """Apply the unchanged frozen M64 gates with collision-safe column access."""
    a = att[(att["season"].eq("combined")) & att["candidate"].eq("generative_gamescript")].iloc[0]
    ar = att[(att["season"].eq("combined")) & att["candidate"].eq("raw")].iloc[0]
    p = pas[(pas["season"].eq("combined")) & pas["candidate"].eq("generative_gamescript")].iloc[0]
    pr = pas[(pas["season"].eq("combined")) & pas["candidate"].eq("raw")].iloc[0]

    year_att_nonworse = True
    year_pass_nonworse = True
    for season in ("2024", "2025"):
        ag = att[(att["season"].eq(season)) & att["candidate"].eq("generative_gamescript")].iloc[0]
        ab = att[(att["season"].eq(season)) & att["candidate"].eq("raw")].iloc[0]
        pg = pas[(pas["season"].eq(season)) & pas["candidate"].eq("generative_gamescript")].iloc[0]
        pb = pas[(pas["season"].eq(season)) & pas["candidate"].eq("raw")].iloc[0]
        year_att_nonworse &= bool(float(ag["mae"]) <= float(ab["mae"]) + 1e-12)
        year_pass_nonworse &= bool(float(pg["mae"]) <= float(pb["mae"]) + 1e-12)

    gates = {
        "attempt_mae_gain_ge_0_40": float(ar["mae"] - a["mae"]) >= 0.40,
        "attempt_mae_nonworse_both_years": year_att_nonworse,
        "attempt_corr_gain_ge_0_03": float(a["corr"] - ar["corr"]) >= 0.03,
        "attempt_10plus_misses_reduce_10pct": int(a["miss_10plus"]) <= int(np.floor(ar["miss_10plus"] * 0.90)),
        "actual_40plus_attempt_mae_gain_ge_0_75": float(ar["actual_40plus_mae"] - a["actual_40plus_mae"]) >= 0.75,
        "pass_mae_gain_ge_1_50": float(pr["mae"] - p["mae"]) >= 1.50,
        "pass_mae_nonworse_both_years": year_pass_nonworse,
        "pass_corr_gain_ge_0_03": float(p["corr"] - pr["corr"]) >= 0.03,
        "pass_100plus_misses_reduce_10pct": int(p["miss_100plus"]) <= int(np.floor(pr["miss_100plus"] * 0.90)),
    }
    n_week_wins = int(weeks["pass_win"].sum())
    total_weeks = int(len(weeks))
    all_pass = bool(all(gates.values()))
    return pd.DataFrame([{
        **gates,
        "weekly_pass_wins": n_week_wins,
        "weekly_total": total_weeks,
        "raw_attempt_mae": float(ar["mae"]),
        "gen_attempt_mae": float(a["mae"]),
        "attempt_mae_gain": float(ar["mae"] - a["mae"]),
        "raw_attempt_corr": float(ar["corr"]),
        "gen_attempt_corr": float(a["corr"]),
        "attempt_corr_gain": float(a["corr"] - ar["corr"]),
        "raw_pass_mae": float(pr["mae"]),
        "gen_pass_mae": float(p["mae"]),
        "pass_mae_gain": float(pr["mae"] - p["mae"]),
        "raw_pass_corr": float(pr["corr"]),
        "gen_pass_corr": float(p["corr"]),
        "pass_corr_gain": float(p["corr"] - pr["corr"]),
        "raw_100plus": int(pr["miss_100plus"]),
        "gen_100plus": int(p["miss_100plus"]),
        "m64_architecture_actionable": all_pass,
        "interpretation": "eligible_for_next_stage" if all_pass else "hold_architecture",
    }])


_ORIG_PREPARE_PBP = m.prepare_pbp
m.prepare_pbp = prepare_pbp_fixed
m.frozen_verdict = frozen_verdict_fixed

if __name__ == "__main__":
    raise SystemExit(m.main())
