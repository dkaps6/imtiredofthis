#!/usr/bin/env python3
"""Step 1 of docs/research/MARKET_IMPLIED_GAME_SCRIPT_V1_PLAN.md.

Diagnostic only: does Vegas's game-level spread/total (a different, more
liquid market than player props, known pregame from the nflverse schedule)
carry genuine incremental predictive value for actual team plays and actual
team pass rate, beyond a simple historical-tendency baseline -- the same two
quantities scripts/modeling/rules_v2.py::project_game_script() estimates for
every position (QB pass-rate/plays directly; RB/WR/TE opportunity shares are
computed against the same play-count ceiling).

Does not touch rules_v2.py, simulation_rules.py, or any live pricing path.
The historical baseline here is a simple prior-N-week rolling average of the
team's own actual plays/pass-rate, a proxy for "what the existing
historical-tendency approach would predict" -- not a literal reimplementation
of estimate_plays()/success_diff(), which would require the full TeamContext
pipeline. Disclosed simplification, not a hidden one.

Single-direction holdout only (fit 2024, freeze, evaluate blind on 2025):
a genuine reverse direction (fit 2025, test 2024) would require the training
fold's own rolling-history feature to never draw on the test season's
outcomes, which it cannot avoid here since 2025's early weeks' prior-8
history reaches directly into 2024 -- exactly the season being held out in
that direction. Fixing that would require pulling additional backstop
seasons (2022/2023) purely for history, which this diagnostic does not
attempt; noted as a real limitation rather than silently run anyway.

Three arms compared on the 2025 holdout, all frozen on 2024 only:
(1) the raw historical rolling baseline itself (no fit at all);
(2) that same baseline re-fit through a plain linear regression (isolates
    how much of any apparent improvement is just correcting the baseline's
    own scale/bias, independent of the market columns);
(3) the baseline plus market_team_implied/market_abs_spread, fit jointly.
Incremental market value is (2) vs (3), not (1) vs (3).

Research only. No production/model/weight/threshold change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from scripts._opponent_map import canon_team

ROLLING_WINDOW = 8
MIN_PRIOR_WEEKS = 3


def num(v):
    return pd.to_numeric(v, errors="coerce")


def load_market_schedule(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    rows = []
    for season in seasons:
        s = nfl.load_schedules(int(season))
        if hasattr(s, "to_pandas"):
            s = s.to_pandas()
        s = pd.DataFrame(s)
        s.columns = [str(c).strip().lower() for c in s.columns]
        if "game_type" in s.columns:
            s = s.loc[s["game_type"].astype(str).str.upper().eq("REG")].copy()
        required = {"season", "week", "home_team", "away_team", "spread_line", "total_line"}
        missing = required - set(s.columns)
        if missing:
            raise RuntimeError(f"{season} schedule missing required market columns: {sorted(missing)}")
        for _, r in s.iterrows():
            total = num(pd.Series([r.get("total_line")])).iloc[0]
            spread = num(pd.Series([r.get("spread_line")])).iloc[0]
            # nflverse's spread_line convention (verified empirically against
            # real 2023 results, e.g. DAL home spread_line=+17.5, won 49-17):
            # POSITIVE means the HOME team is favored by that many points,
            # negative means home is the underdog. team_spread below is
            # therefore "points this team is favored by" for each side
            # (positive=favored), and the favored team's implied total must
            # be the LARGER half of the total -- (total + team_spread) / 2,
            # not (total - team_spread) / 2.
            for side, team, opp in [("home", r["home_team"], r["away_team"]), ("away", r["away_team"], r["home_team"])]:
                team_spread = spread if side == "home" else (-spread if pd.notna(spread) else np.nan)
                implied = (total + team_spread) / 2.0 if pd.notna(total) and pd.notna(team_spread) else np.nan
                rows.append({
                    "season": int(r["season"]), "week": int(r["week"]),
                    "team": canon_team(team), "opponent": canon_team(opp),
                    "market_total": total, "market_team_spread": team_spread,
                    "market_team_implied": implied,
                    "market_abs_spread": abs(team_spread) if pd.notna(team_spread) else np.nan,
                })
    out = pd.DataFrame(rows)
    if out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate market schedule rows")
    return out


def add_rolling_baseline(team_weekly: pd.DataFrame, col: str) -> pd.DataFrame:
    """Strictly-prior rolling mean of the team's own actual outcome.

    Crosses season boundaries within the combined multi-season frame (the
    same team-week ordering carries the identity), mirroring how last8-style
    history is already handled elsewhere tonight (e.g. RB-PD2 multiseason).
    """
    x = team_weekly.sort_values(["team", "season", "week"]).copy()
    x[f"{col}_prior_avg"] = x.groupby("team")[col].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=MIN_PRIOR_WEEKS).mean()
    )
    return x


def build_cohort(team_weekly: pd.DataFrame, market: pd.DataFrame, target_col: str) -> pd.DataFrame:
    x = add_rolling_baseline(team_weekly, target_col)
    x = x.merge(market, on=["season", "week", "team"], how="inner", validate="one_to_one")
    x = x.loc[x[f"{target_col}_prior_avg"].notna() & x["market_team_implied"].notna() & x[target_col].notna()].copy()
    return x


def fit_and_evaluate(cohort: pd.DataFrame, *, target_col: str, fit_season: int, test_season: int) -> dict:
    train = cohort.loc[cohort["season"].eq(fit_season)]
    test = cohort.loc[cohort["season"].eq(test_season)]
    if len(train) < 50 or len(test) < 50:
        return {"target": target_col, "fit_season": fit_season, "test_season": test_season, "status": "INSUFFICIENT_ROWS"}

    baseline_col = f"{target_col}_prior_avg"
    y_train = train[target_col].to_numpy(dtype=float)
    y_test = test[target_col].to_numpy(dtype=float)

    # Arm 1: raw historical rolling baseline, unfit, carried through as-is.
    raw_baseline_pred = test[baseline_col].to_numpy(dtype=float)
    raw_baseline_mae = float(np.mean(np.abs(raw_baseline_pred - y_test)))

    # Arm 2: the SAME baseline re-fit through a plain linear regression
    # (intercept + scale correction only, no market columns). This is the
    # correct comparator for isolating market value -- Arm 3 must beat this,
    # not the unfit Arm 1, since Arm 3 also gets to fit the baseline's own
    # scale/bias and would otherwise get undue credit for doing so.
    x_train_baseline_only = train[[baseline_col]].to_numpy(dtype=float)
    baseline_model = LinearRegression()
    baseline_model.fit(x_train_baseline_only, y_train)
    x_test_baseline_only = test[[baseline_col]].to_numpy(dtype=float)
    fitted_baseline_pred = baseline_model.predict(x_test_baseline_only)
    fitted_baseline_mae = float(np.mean(np.abs(fitted_baseline_pred - y_test)))

    # Arm 3: baseline + market signal, fit jointly, frozen, applied blind.
    x_train = train[[baseline_col, "market_team_implied", "market_abs_spread"]].to_numpy(dtype=float)
    model = LinearRegression()
    model.fit(x_train, y_train)
    x_test = test[[baseline_col, "market_team_implied", "market_abs_spread"]].to_numpy(dtype=float)
    blended_pred_test = model.predict(x_test)
    blended_mae = float(np.mean(np.abs(blended_pred_test - y_test)))

    return {
        "target": target_col, "fit_season": fit_season, "test_season": test_season, "status": "OK",
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "raw_baseline_mae": raw_baseline_mae,
        "fitted_baseline_only_mae": fitted_baseline_mae,
        "baseline_plus_market_mae": blended_mae,
        # The real incremental-value comparison: fitted baseline-only vs
        # baseline+market, both given equal opportunity to correct the
        # baseline's own scale/bias.
        "incremental_market_mae_improvement": fitted_baseline_mae - blended_mae,
        "market_coefficients": {
            "baseline_weight": float(model.coef_[0]),
            "market_implied_weight": float(model.coef_[1]),
            "market_abs_spread_weight": float(model.coef_[2]),
            "intercept": float(model.intercept_),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--team-weekly", type=Path, required=True, help="team_weekly_history.csv (has actual plays_est/dropback_rate)")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    if not a.team_weekly.exists() or not a.team_weekly.stat().st_size:
        raise RuntimeError(f"missing team-weekly file: {a.team_weekly}")
    team_weekly = pd.read_csv(a.team_weekly, low_memory=False)
    team_weekly.columns = [str(c).strip().lower() for c in team_weekly.columns]
    required = {"season", "week", "team", "plays_est", "dropback_rate"}
    missing = required - set(team_weekly.columns)
    if missing:
        raise RuntimeError(f"team-weekly file missing required columns: {sorted(missing)}")
    team_weekly["team"] = team_weekly["team"].map(canon_team)

    seasons = sorted(pd.to_numeric(team_weekly["season"], errors="raise").astype(int).unique().tolist())
    if len(seasons) != 2:
        raise RuntimeError(f"expected exactly 2 seasons in team-weekly file, found {seasons}")

    market = load_market_schedule(seasons)

    # Single direction only: fit_season must be the earlier season, so the
    # training fold's rolling history (which crosses season boundaries) can
    # never reach into the test season. The reverse direction cannot make
    # this guarantee with only two seasons of history available -- see the
    # module docstring.
    fit_season, test_season = seasons[0], seasons[1]

    a.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for target_col in ["plays_est", "dropback_rate"]:
        cohort = build_cohort(team_weekly, market, target_col)
        cohort.to_csv(a.out_dir / f"market_game_script_cohort_{target_col}.csv", index=False)
        results.append(fit_and_evaluate(cohort, target_col=target_col, fit_season=fit_season, test_season=test_season))

    out = pd.DataFrame(results)
    out.to_csv(a.out_dir / "market_game_script_diagnosis_summary.csv", index=False)
    print("=== MARKET-IMPLIED GAME SCRIPT, INCREMENTAL-VALUE DIAGNOSIS ===")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
