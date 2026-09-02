#!/usr/bin/env python3
"""RB STACK6N: strict-prior deep-late conditional rushing-tendency qualification."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

START_WEEK = 6
ALPHA = 0.75
TEAM_WINDOW = 8
PSEUDO_DEEP_LATE_PLAYS = 24.0
RATE_MIN = 0.05
RATE_MAX = 0.75
EXPECTED_N = 388
EXPECTED_OCCUPANCY_MAE = 5.518381962346741


def num(v):
    return pd.to_numeric(v, errors="coerce")


def lower(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    z.columns = [str(c).strip().lower() for c in z.columns]
    return z


def one(root: Path, name: str) -> pd.DataFrame:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name}; found {len(hits)}")
    return lower(pd.read_csv(hits[0], low_memory=False))


def metric(y, p) -> dict:
    y, p = num(y), num(p)
    ok = y.notna() & p.notna()
    y, p = y[ok], p[ok]
    e = p - y
    return {
        "n": int(len(y)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(e.mean()),
        "corr": float(p.corr(y)) if len(y) >= 3 and p.nunique() > 1 and y.nunique() > 1 else np.nan,
    }


def load_pbp_team_games() -> pd.DataFrame:
    import nflreadpy as nfl

    p = lower(nfl.load_pbp(seasons=[2023, 2024, 2025]).to_pandas())
    if "season_type" in p.columns:
        reg = p.loc[p["season_type"].astype(str).str.upper().eq("REG")].copy()
        if len(reg):
            p = reg
    required = {"season", "week", "posteam", "rush_attempt", "qb_dropback", "qtr"}
    miss = required - set(p.columns)
    if miss:
        raise RuntimeError(f"STACK6N PBP missing columns: {sorted(miss)}")

    p["team"] = p["posteam"].map(canon_team)
    p["rush_attempt"] = num(p["rush_attempt"]).fillna(0)
    p["qb_dropback"] = num(p["qb_dropback"]).fillna(0)
    p = p.loc[(p["rush_attempt"].eq(1) | p["qb_dropback"].eq(1)) & p["team"].ne("")].copy()

    if "score_differential" in p.columns:
        diff = num(p["score_differential"])
    elif {"posteam_score", "defteam_score"}.issubset(p.columns):
        diff = num(p["posteam_score"]) - num(p["defteam_score"])
    else:
        raise RuntimeError("STACK6N PBP has no score differential fields")

    p["score_diff"] = diff.fillna(0.0)
    p["qtr_num"] = num(p["qtr"]).fillna(0)
    p["trail"] = p["score_diff"].lt(-3)
    p["deep_late"] = p["score_diff"].le(-9) & p["qtr_num"].ge(4)

    rows = []
    for (season, week, team), g in p.groupby(["season", "week", "team"], dropna=False):
        n = float(len(g))
        trail = g.loc[g["trail"]]
        dl = g.loc[g["deep_late"]]
        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "team": canon_team(team),
                "off_plays": n,
                "trail_plays": float(len(trail)),
                "trail_rushes": float(trail["rush_attempt"].sum()),
                "deep_late_plays": float(len(dl)),
                "deep_late_rushes": float(dl["rush_attempt"].sum()),
                "deep_late_share": float(len(dl) / n) if n else np.nan,
                "deep_late_rush_rate": float(dl["rush_attempt"].mean()) if len(dl) else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values(["season", "week", "team"]).reset_index(drop=True)
    if out.empty or out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("STACK6N PBP team-game aggregation invalid")
    return out


def strict_prior(history: pd.DataFrame, season: int, week: int, team: str | None = None) -> pd.DataFrame:
    s = num(history["season"])
    w = num(history["week"])
    mask = s.lt(int(season)) | (s.eq(int(season)) & w.lt(int(week)))
    if team is not None:
        mask &= history["team"].astype(str).eq(str(team))
    return history.loc[mask].sort_values(["season", "week"])


def rate(rushes: float, plays: float) -> float:
    return float(rushes / plays) if plays > 0 else np.nan


def build_prior_estimates(history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in targets.iterrows():
        season, week, team = int(r["season"]), int(r["week"]), str(r["team"])
        league = strict_prior(history, season, week)
        if league.empty:
            raise RuntimeError(f"no strict-prior league history for {season} W{week} {team}")

        lg_trail_plays = float(league["trail_plays"].sum())
        lg_trail_rushes = float(league["trail_rushes"].sum())
        lg_dl_plays = float(league["deep_late_plays"].sum())
        lg_dl_rushes = float(league["deep_late_rushes"].sum())
        lg_trail_rate = rate(lg_trail_rushes, lg_trail_plays)
        lg_dl_rate = rate(lg_dl_rushes, lg_dl_plays)
        if not np.isfinite(lg_trail_rate) or not np.isfinite(lg_dl_rate):
            raise RuntimeError(f"missing strict-prior league rates for {season} W{week} {team}")
        lg_delta = float(lg_dl_rate - lg_trail_rate)

        team_prior = strict_prior(history, season, week, team).tail(TEAM_WINDOW)
        t_trail_plays = float(team_prior["trail_plays"].sum())
        t_trail_rushes = float(team_prior["trail_rushes"].sum())
        t_dl_plays = float(team_prior["deep_late_plays"].sum())
        t_dl_rushes = float(team_prior["deep_late_rushes"].sum())
        t_trail_rate = rate(t_trail_rushes, t_trail_plays)
        t_dl_rate = rate(t_dl_rushes, t_dl_plays)
        usable_team_delta = np.isfinite(t_trail_rate) and np.isfinite(t_dl_rate)
        team_delta = float(t_dl_rate - t_trail_rate) if usable_team_delta else lg_delta
        weight = float(t_dl_plays / (t_dl_plays + PSEUDO_DEEP_LATE_PLAYS)) if t_dl_plays > 0 else 0.0
        shrunk_delta = float(lg_delta + weight * (team_delta - lg_delta))

        rows.append(
            {
                "season": season,
                "week": week,
                "team": team,
                "league_trail_rate_prior": lg_trail_rate,
                "league_deep_late_rate_prior": lg_dl_rate,
                "league_context_delta_prior": lg_delta,
                "league_deep_late_plays_prior": lg_dl_plays,
                "team_prior_games": int(len(team_prior)),
                "team_trail_plays_prior8": t_trail_plays,
                "team_deep_late_plays_prior8": t_dl_plays,
                "team_trail_rate_prior8": t_trail_rate,
                "team_deep_late_rate_prior8": t_dl_rate,
                "team_context_delta_prior8": team_delta,
                "team_deep_late_weight": weight,
                "team_shrunk_context_delta": shrunk_delta,
                "strict_prior_ok": 1,
            }
        )
    return pd.DataFrame(rows)


def add_arm(t: pd.DataFrame, name: str, dl_rate: pd.Series) -> None:
    non_dl_trail = (t["trail_play_share"] - t["deep_late_share"]).clip(lower=0.0)
    contribution = (
        t["lead_play_share"] * t["gs_team_lead_rush_rate_shrunk"]
        + t["neutral_play_share"] * t["gs_team_neutral_rush_rate_shrunk"]
        + non_dl_trail * t["gs_team_trail_rush_rate_shrunk"]
        + t["deep_late_share"] * dl_rate
    )
    t[name] = (1.0 - ALPHA) * t["baseline_team_rush_att"] + ALPHA * t["pred_off_plays"] * contribution


def pop_scores(w: pd.DataFrame, label: str, mask: pd.Series) -> pd.DataFrame:
    q = w.loc[mask].copy()
    base = metric(q["actual_team_rush_att"], q["OCCUPANCY_BASE"])
    rows = []
    for arm in ["OCCUPANCY_BASE", "LEAGUE_DEEP_LATE_CONTEXT", "TEAM_SHRUNK_DEEP_LATE", "ORACLE_DEEP_LATE"]:
        m = metric(q["actual_team_rush_att"], q[arm])
        rows.append({"population": label, "arm": arm, **m, "mae_gain_vs_base": float(base["mae"] - m["mae"])})
    return pd.DataFrame(rows)


def gate_arm(scores: pd.DataFrame, arm: str) -> dict:
    def row(pop, which):
        return scores.loc[(scores["population"].eq(pop)) & (scores["arm"].eq(which))].iloc[0]

    base = row("ALL_W6_18", "OCCUPANCY_BASE")
    cur = row("ALL_W6_18", arm)
    oracle = row("ALL_W6_18", "ORACLE_DEEP_LATE")
    over = row("POOL_OVER_5", arm)
    under = row("POOL_UNDER_5", arm)
    non = row("NON_EXTREME_ABS_LT3", arm)
    headroom = float(base["mae"] - oracle["mae"])
    gain = float(base["mae"] - cur["mae"])
    frac = float(gain / headroom) if headroom > 1e-12 else np.nan
    checks = {
        "overall_mae_gain_ge_0_10": gain >= 0.10,
        "oracle_headroom_fraction_ge_0_20": np.isfinite(frac) and frac >= 0.20,
        "pool_over5_gain_gt_0_20": float(over["mae_gain_vs_base"]) > 0.20,
        "pool_under5_regression_le_0_10": float(under["mae_gain_vs_base"]) >= -0.10,
        "non_extreme_regression_le_0_05": float(non["mae_gain_vs_base"]) >= -0.05,
        "overall_rmse_not_worse": float(cur["rmse"]) <= float(base["rmse"]) + 1e-12,
        "abs_bias_le_0_75": abs(float(cur["bias"])) <= 0.75,
    }
    return {
        "arm": arm,
        "overall_mae_gain": gain,
        "oracle_deep_late_headroom": headroom,
        "headroom_fraction_recovered": frac,
        "pool_over5_mae_gain": float(over["mae_gain_vs_base"]),
        "pool_under5_mae_gain": float(under["mae_gain_vs_base"]),
        "non_extreme_mae_gain": float(non["mae_gain_vs_base"]),
        "overall_rmse_gain": float(base["rmse"] - cur["rmse"]),
        "overall_abs_bias": abs(float(cur["bias"])),
        **{k: int(v) for k, v in checks.items()},
        "passes_all_gates": int(all(checks.values())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m94c-root", type=Path, required=True)
    ap.add_argument("--stack6h-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    m = one(a.m94c_root, "m94c_2025_team_trace.csv")
    h = one(a.stack6h_root, "stack6h_team_trace.csv")
    pbp = load_pbp_team_games()

    for d in (m, h, pbp):
        d["season"] = num(d["season"]).astype(int)
        d["week"] = num(d["week"]).astype(int)
        d["team"] = d["team"].map(canon_team)

    req = [
        "season", "week", "team", "actual_team_rush_att", "baseline_team_rush_att", "pred_off_plays",
        "lead_play_share", "neutral_play_share", "trail_play_share",
        "gs_team_lead_rush_rate_shrunk", "gs_team_neutral_rush_rate_shrunk", "gs_team_trail_rush_rate_shrunk",
    ]
    bins = ["pool_over_5", "pool_under_5", "pool_abs_5", "non_extreme_abs_lt3"]
    target_pbp = pbp.loc[pbp["season"].eq(2025), ["season", "week", "team", "off_plays", "deep_late_plays", "deep_late_rushes", "deep_late_share", "deep_late_rush_rate"]].copy()
    t = (
        m[req]
        .merge(h[["season", "week", "team", *bins]], on=["season", "week", "team"], how="inner", validate="one_to_one")
        .merge(target_pbp, on=["season", "week", "team"], how="inner", validate="one_to_one")
    )
    if len(t) != 544:
        raise RuntimeError(f"expected 544 joined 2025 rows; got {len(t)}")

    prior = build_prior_estimates(pbp, t[["season", "week", "team"]])
    t = t.merge(prior, on=["season", "week", "team"], how="left", validate="one_to_one")
    if t["strict_prior_ok"].fillna(0).ne(1).any():
        raise RuntimeError("STACK6N strict-prior coverage failure")

    numeric_cols = [c for c in t.columns if c not in {"team"}]
    for c in numeric_cols:
        t[c] = num(t[c])

    add_arm(t, "OCCUPANCY_BASE", t["gs_team_trail_rush_rate_shrunk"])
    league_rate = (t["gs_team_trail_rush_rate_shrunk"] + t["league_context_delta_prior"]).clip(RATE_MIN, RATE_MAX)
    team_rate = (t["gs_team_trail_rush_rate_shrunk"] + t["team_shrunk_context_delta"]).clip(RATE_MIN, RATE_MAX)
    oracle_rate = t["deep_late_rush_rate"].where(t["deep_late_plays"].gt(0), t["gs_team_trail_rush_rate_shrunk"])
    add_arm(t, "LEAGUE_DEEP_LATE_CONTEXT", league_rate)
    add_arm(t, "TEAM_SHRUNK_DEEP_LATE", team_rate)
    add_arm(t, "ORACLE_DEEP_LATE", oracle_rate)
    t["pred_deep_late_rate_league"] = league_rate
    t["pred_deep_late_rate_team"] = team_rate

    w = t.loc[t["week"].ge(START_WEEK)].copy()
    if len(w) != EXPECTED_N:
        raise RuntimeError(f"expected {EXPECTED_N} W6-18 rows; got {len(w)}")

    score_parts = [pop_scores(w, "ALL_W6_18", pd.Series(True, index=w.index))]
    masks = {
        "POOL_OVER_5": w["pool_over_5"].eq(1),
        "POOL_UNDER_5": w["pool_under_5"].eq(1),
        "POOL_ABS_5": w["pool_abs_5"].eq(1),
        "NON_EXTREME_ABS_LT3": w["non_extreme_abs_lt3"].eq(1),
    }
    score_parts += [pop_scores(w, label, mask) for label, mask in masks.items()]
    scores = pd.concat(score_parts, ignore_index=True)

    occ_mae = float(scores.loc[(scores["population"].eq("ALL_W6_18")) & (scores["arm"].eq("OCCUPANCY_BASE")), "mae"].iloc[0])
    strict_prior_coverage = float(w["strict_prior_ok"].mean())
    integrity_pass = int(abs(occ_mae - EXPECTED_OCCUPANCY_MAE) <= 1e-9 and strict_prior_coverage == 1.0)

    dl = w.loc[w["deep_late_plays"].gt(0)].copy()
    rate_rows = []
    for name, col in [
        ("PARENT_TRAIL_RATE", "gs_team_trail_rush_rate_shrunk"),
        ("LEAGUE_DEEP_LATE_CONTEXT", "pred_deep_late_rate_league"),
        ("TEAM_SHRUNK_DEEP_LATE", "pred_deep_late_rate_team"),
    ]:
        err = (dl[col] - dl["deep_late_rush_rate"]).abs()
        wt = dl["deep_late_plays"].fillna(0)
        rate_rows.append(
            {
                "arm": name,
                "target_games_with_deep_late_plays": int(len(dl)),
                "deep_late_rate_mae_game_weighted": float(err.mean()),
                "deep_late_rate_mae_play_weighted": float(np.average(err, weights=wt)) if wt.sum() > 0 else np.nan,
            }
        )
    rate_scores = pd.DataFrame(rate_rows)

    coverage = pd.DataFrame([
        {
            "w6_18_n": len(w),
            "strict_prior_coverage": strict_prior_coverage,
            "target_games_with_deep_late_plays": int(w["deep_late_plays"].gt(0).sum()),
            "target_deep_late_game_rate": float(w["deep_late_plays"].gt(0).mean()),
            "team_prior8_deep_late_positive_games": int(w["team_deep_late_plays_prior8"].gt(0).sum()),
            "team_prior8_deep_late_positive_rate": float(w["team_deep_late_plays_prior8"].gt(0).mean()),
            "mean_team_prior8_deep_late_plays": float(w["team_deep_late_plays_prior8"].mean()),
            "median_team_prior8_deep_late_plays": float(w["team_deep_late_plays_prior8"].median()),
            "mean_team_deep_late_weight": float(w["team_deep_late_weight"].mean()),
            "team_window_games": TEAM_WINDOW,
            "pseudo_deep_late_plays": PSEUDO_DEEP_LATE_PLAYS,
        }
    ])

    gates = pd.DataFrame([gate_arm(scores, "LEAGUE_DEEP_LATE_CONTEXT"), gate_arm(scores, "TEAM_SHRUNK_DEEP_LATE")])
    if not integrity_pass:
        disposition = "STACK6N_INTEGRITY_FAILURE_DO_NOT_INTERPRET"
    else:
        lg = gates.loc[gates["arm"].eq("LEAGUE_DEEP_LATE_CONTEXT")].iloc[0]
        tm = gates.loc[gates["arm"].eq("TEAM_SHRUNK_DEEP_LATE")].iloc[0]
        lg_pass = bool(lg["passes_all_gates"])
        tm_pass = bool(tm["passes_all_gates"])
        team_vs_league = float(
            scores.loc[(scores["population"].eq("ALL_W6_18")) & (scores["arm"].eq("LEAGUE_DEEP_LATE_CONTEXT")), "mae"].iloc[0]
            - scores.loc[(scores["population"].eq("ALL_W6_18")) & (scores["arm"].eq("TEAM_SHRUNK_DEEP_LATE")), "mae"].iloc[0]
        )
        if tm_pass and team_vs_league >= 0.03:
            disposition = "TEAM_DEEP_LATE_PERSISTENCE_RETAINED"
        elif lg_pass or tm_pass:
            disposition = "LEAGUE_DEEP_LATE_CONTEXT_RETAINED"
        else:
            disposition = "DEEP_LATE_HISTORY_NOT_RETAINABLE"

    league_mae = float(scores.loc[(scores["population"].eq("ALL_W6_18")) & (scores["arm"].eq("LEAGUE_DEEP_LATE_CONTEXT")), "mae"].iloc[0])
    team_mae = float(scores.loc[(scores["population"].eq("ALL_W6_18")) & (scores["arm"].eq("TEAM_SHRUNK_DEEP_LATE")), "mae"].iloc[0])
    disposition_df = pd.DataFrame([
        {
            "disposition": disposition,
            "integrity_pass": integrity_pass,
            "team_mae_gain_vs_league": league_mae - team_mae,
            "production_change": 0,
            "player_recomposition_authorized": 0,
            "conditional_mechanism_only": 1,
        }
    ])

    integrity = pd.DataFrame([
        {
            "m94c_rows": len(m),
            "stack6h_rows": len(h),
            "pbp_team_games_all_seasons": len(pbp),
            "joined_2025_rows": len(t),
            "w6_18_n": len(w),
            "expected_occupancy_mae": EXPECTED_OCCUPANCY_MAE,
            "observed_occupancy_mae": occ_mae,
            "strict_prior_coverage": strict_prior_coverage,
            "integrity_pass": integrity_pass,
            "fitted_models": 0,
            "feature_search": 0,
            "hyperparameter_search": 0,
            "threshold_search": 0,
            "window_search": 0,
            "sportsbook_inputs": 0,
            "target_game_pbp_used_for_candidate": 0,
            "target_game_pbp_used_for_labels_and_conditional_scaffold_only": 1,
        }
    ])

    t.to_csv(a.out_dir / "stack6n_team_trace.csv", index=False)
    scores.to_csv(a.out_dir / "stack6n_scores.csv", index=False)
    rate_scores.to_csv(a.out_dir / "stack6n_rate_scores.csv", index=False)
    coverage.to_csv(a.out_dir / "stack6n_coverage.csv", index=False)
    gates.to_csv(a.out_dir / "stack6n_gates.csv", index=False)
    integrity.to_csv(a.out_dir / "stack6n_integrity.csv", index=False)
    disposition_df.to_csv(a.out_dir / "stack6n_disposition.csv", index=False)

    print("=== STACK6N integrity ===")
    print(integrity.to_string(index=False))
    print("=== STACK6N coverage ===")
    print(coverage.to_string(index=False))
    print("=== STACK6N scores ===")
    print(scores.to_string(index=False))
    print("=== STACK6N direct rate scores ===")
    print(rate_scores.to_string(index=False))
    print("=== STACK6N gates ===")
    print(gates.to_string(index=False))
    print("=== STACK6N disposition ===")
    print(disposition_df.to_string(index=False))
    print(f"STACK6N_DISPOSITION={disposition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
