#!/usr/bin/env python3
"""Migration 56: diagnose whether richer weekly defense context explains QB residuals.

Diagnostic only. This script does not alter production projections. It uses only
2024-2025 as scored development seasons; 2023/2024 are prior-history inputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def num(value) -> pd.Series:
    return pd.to_numeric(value, errors="coerce")


def corr(a, b) -> float:
    z = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    return float(z.a.corr(z.b)) if len(z) > 2 and z.a.nunique() > 1 and z.b.nunique() > 1 else np.nan


def metrics(actual, pred) -> dict:
    z = pd.DataFrame({"a": num(actual), "p": num(pred)}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "correlation": np.nan}
    e = z.p - z.a
    return {
        "n": len(z),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(e.mean()),
        "correlation": float(z.p.corr(z.a)) if len(z) > 2 and z.p.nunique() > 1 else np.nan,
    }


def weighted_mean(g: pd.DataFrame, col: str, tail: int) -> float:
    if col not in g.columns or g.empty:
        return np.nan
    z = g.tail(tail)
    v = num(z[col])
    ok = v.notna()
    if not ok.any():
        return np.nan
    w = np.arange(1, len(z) + 1, dtype=float)
    return float(np.average(v[ok], weights=w[ok]))


def build_defense_observations(team_weekly: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    h = team_weekly.copy()
    s = schedule.copy()
    for frame in (h, s):
        frame["season"] = num(frame["season"])
        frame["week"] = num(frame["week"])
        frame["team"] = frame["team"].astype(str).str.upper().str.strip()
    s["opponent"] = s["opponent"].astype(str).str.upper().str.strip()
    s = s[["season", "week", "team", "opponent"]].drop_duplicates(["season", "week", "team"])
    base = h.merge(s, on=["season", "week", "team"], how="left", validate="one_to_one")

    offense_cols = [c for c in [
        "plays_est", "dropback_rate", "pass_attempts_per_dropback", "proe",
        "success_rate_off", "pressure_rate_allowed", "neutral_pace",
    ] if c in h.columns]
    opp = h[["season", "week", "team", *offense_cols]].rename(
        columns={"team": "opponent", **{c: f"opp_off_{c}" for c in offense_cols}}
    )
    base = base.merge(opp, on=["season", "week", "opponent"], how="left", validate="one_to_one")

    plays = num(base.get("opp_off_plays_est"))
    drop_rate = num(base.get("opp_off_dropback_rate"))
    att_per_drop = num(base.get("opp_off_pass_attempts_per_dropback"))
    base["opp_plays_faced"] = plays
    base["opp_dropbacks_faced"] = plays * drop_rate
    base["opp_pass_attempts_faced"] = plays * drop_rate * att_per_drop
    base["opp_dropback_rate_faced"] = drop_rate
    base["opp_proe_faced"] = num(base.get("opp_off_proe"))
    base["opp_success_rate_faced"] = num(base.get("opp_off_success_rate_off"))
    base["opp_neutral_pace_faced"] = num(base.get("opp_off_neutral_pace"))
    base["pass_funnel_epa"] = num(base.get("def_pass_epa")) - num(base.get("def_rush_epa"))
    return base.sort_values(["season", "week", "team"]).reset_index(drop=True)


BASE_DEF = [
    "pressure_rate_generated", "def_pass_epa", "success_rate_def",
    "explosive_play_rate_allowed", "coverage_man_rate", "coverage_zone_rate",
]
OPPORTUNITY = [
    "opp_plays_faced", "opp_dropbacks_faced", "opp_pass_attempts_faced",
    "opp_dropback_rate_faced", "opp_proe_faced", "opp_success_rate_faced",
    "opp_neutral_pace_faced", "pass_funnel_epa", "def_rush_epa",
]
COVERAGE = [
    "cover_0_rate", "cover_1_rate", "cover_2_rate", "cover_3_rate", "cover_4_rate",
    "cover_6_rate", "cover_9_rate", "2_man_rate", "avg_defenders_in_box",
    "light_box_rate", "heavy_box_rate",
]


def add_defense_context(frame: pd.DataFrame, observations: pd.DataFrame, season: int) -> pd.DataFrame:
    x = frame.copy()
    features = [c for c in [*BASE_DEF, *OPPORTUNITY, *COVERAGE] if c in observations.columns]
    rows = []
    for _, r in x.iterrows():
        prior = observations[
            observations.team.astype(str).eq(str(r.opponent).upper())
            & ((num(observations.season) < season) | ((num(observations.season) == season) & (num(observations.week) < int(r.week))))
        ].sort_values(["season", "week"])
        rec = {"_row": r.name, "def_prior_games": int(len(prior))}
        for c in features:
            rec[f"def8_{c}"] = weighted_mean(prior, c, 8)
            rec[f"def3_{c}"] = weighted_mean(prior, c, 3)
            a, b = rec[f"def3_{c}"], rec[f"def8_{c}"]
            rec[f"deftrend_{c}"] = a - b if np.isfinite(a) and np.isfinite(b) else np.nan
        rows.append(rec)
    return x.join(pd.DataFrame(rows).set_index("_row"))


def season_frame(root: Path, season: int) -> pd.DataFrame:
    base = root / str(season)
    stable = read(base / "qb_joint_attempts_ypa_mc/qb_joint_attempts_ypa_mc_stable.csv")
    cand = read(base / "qb_joint_attempts_ypa/qb_joint_attempts_ypa_trace.csv")
    team = read(base / "team_weekly_history.csv")
    sched = read(base / "schedule_history.csv")
    observations = build_defense_observations(team, sched)

    keys = ["week", "team", "player_clean_key"]
    cur = stable[stable.candidate.eq("current")][keys + ["actual", "mc_proj"]].rename(
        columns={"actual": "actual_pass_yards", "mc_proj": "current_mc_proj"}
    )
    joint = stable[stable.candidate.eq("joint")][keys + ["actual", "mc_proj"]].rename(
        columns={"actual": "actual_joint", "mc_proj": "joint_mc_proj"}
    )
    x = cur.merge(joint, on=keys, how="inner", validate="one_to_one")
    if not np.allclose(num(x.actual_pass_yards), num(x.actual_joint), equal_nan=True):
        raise RuntimeError(f"actual mismatch in season {season}")
    keep = [c for c in [
        *keys, "opponent", "actual_pass_att", "actual_ypa", "pred_attempts", "attempts_gamescript",
        "pred_ypa", "ypa_contextual", "qb_recent_ypa", "qb_recent_pass_att",
    ] if c in cand.columns]
    x = x.merge(cand[keep].drop_duplicates(keys), on=keys, how="left", validate="one_to_one")
    x["target_season"] = season
    x = add_defense_context(x, observations, season)
    x["joint_pass_error"] = num(x.joint_mc_proj) - num(x.actual_pass_yards)
    x["joint_abs_error"] = x.joint_pass_error.abs()
    x["attempt_residual"] = num(x.actual_pass_att) - num(x.attempts_gamescript)
    x["ypa_residual"] = num(x.actual_ypa) - num(x.ypa_contextual)
    x["pass_residual"] = num(x.actual_pass_yards) - num(x.joint_mc_proj)
    x["actual_40plus"] = num(x.actual_pass_att).ge(40).astype(int)
    x["catastrophic_100plus"] = x.joint_abs_error.ge(100).astype(int)

    # Matchup interactions that the current joint model does not explicitly represent.
    interaction_specs = [
        ("qb_recent_pass_att", "def8_opp_pass_attempts_faced"),
        ("pred_attempts", "def8_opp_dropback_rate_faced"),
        ("qb_recent_ypa", "def8_pressure_rate_generated"),
        ("pred_ypa", "def8_def_pass_epa"),
        ("qb_recent_ypa", "def8_cover_1_rate"),
        ("qb_recent_ypa", "def8_cover_2_rate"),
        ("qb_recent_ypa", "def8_cover_3_rate"),
        ("qb_recent_ypa", "def8_cover_4_rate"),
        ("qb_recent_ypa", "def8_cover_6_rate"),
    ]
    for a, b in interaction_specs:
        if a in x.columns and b in x.columns:
            x[f"int_{a}_x_{b}"] = num(x[a]) * num(x[b])
    return x


def effect_size(a: pd.Series, b: pd.Series) -> float:
    a, b = num(a).dropna(), num(b).dropna()
    if len(a) < 10 or len(b) < 10:
        return np.nan
    pooled = np.sqrt((a.var() + b.var()) / 2.0)
    return float((a.mean() - b.mean()) / pooled) if pooled and np.isfinite(pooled) else np.nan


def signal_ranking(x: pd.DataFrame) -> pd.DataFrame:
    features = [c for c in x.columns if c.startswith(("def8_", "def3_", "deftrend_", "int_"))]
    rows = []
    for c in features:
        v = num(x[c])
        if v.notna().sum() < 50 or v.nunique() < 2:
            continue
        high = v[x.actual_40plus.eq(1)]
        normal = v[x.actual_40plus.eq(0)]
        cat = v[x.catastrophic_100plus.eq(1)]
        small = v[x.joint_abs_error.lt(50)]
        rows.append({
            "feature": c,
            "n": int(v.notna().sum()),
            "coverage": float(v.notna().mean()),
            "corr_actual_attempts": corr(v, x.actual_pass_att),
            "corr_attempt_residual": corr(v, x.attempt_residual),
            "corr_actual_ypa": corr(v, x.actual_ypa),
            "corr_ypa_residual": corr(v, x.ypa_residual),
            "corr_actual_pass_yards": corr(v, x.actual_pass_yards),
            "corr_pass_residual": corr(v, x.pass_residual),
            "corr_abs_pass_error": corr(v, x.joint_abs_error),
            "effect_40plus_vs_under40": effect_size(high, normal),
            "effect_catastrophic_vs_lt50": effect_size(cat, small),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["residual_signal_score"] = out[["corr_attempt_residual", "corr_ypa_residual", "corr_pass_residual"]].abs().max(axis=1)
    return out.sort_values(["residual_signal_score", "effect_40plus_vs_under40"], ascending=False)


def feature_groups(x: pd.DataFrame) -> dict[str, list[str]]:
    existing = [f"def8_{c}" for c in BASE_DEF if f"def8_{c}" in x.columns]
    opportunity = []
    for c in OPPORTUNITY:
        opportunity += [n for n in (f"def8_{c}", f"def3_{c}", f"deftrend_{c}") if n in x.columns]
    coverage = []
    for c in COVERAGE:
        coverage += [n for n in (f"def8_{c}", f"def3_{c}", f"deftrend_{c}") if n in x.columns]
    interactions = [c for c in x.columns if c.startswith("int_")]
    return {
        "bias_only": [],
        "existing_defense": existing,
        "existing_plus_opportunity": list(dict.fromkeys(existing + opportunity)),
        "existing_plus_coverage": list(dict.fromkeys(existing + coverage)),
        "existing_plus_new_defense": list(dict.fromkeys(existing + opportunity + coverage)),
        "existing_plus_new_interactions": list(dict.fromkeys(existing + opportunity + coverage + interactions)),
    }


def fit_correction(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, cap: float, shrink: float) -> pd.Series:
    y = num(train[target])
    if not features:
        raw = pd.Series(float(y.mean()), index=test.index)
    else:
        usable = [c for c in features if c in train.columns and num(train[c]).notna().sum() >= 20 and num(train[c]).nunique() > 1]
        if not usable:
            raw = pd.Series(float(y.mean()), index=test.index)
        else:
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=40.0))
            model.fit(train[usable], y)
            raw = pd.Series(model.predict(test[usable]), index=test.index)
    return raw.clip(-cap, cap) * shrink


def run_oos(x: pd.DataFrame, min_train: int) -> pd.DataFrame:
    groups = feature_groups(x)
    specs = {
        "pass_yards": ("pass_residual", "actual_pass_yards", "joint_mc_proj", 80.0, 0.50),
        "attempts": ("attempt_residual", "actual_pass_att", "attempts_gamescript", 8.0, 0.60),
        "ypa": ("ypa_residual", "actual_ypa", "ypa_contextual", 2.0, 0.60),
    }
    rows = []
    for season in sorted(num(x.target_season).dropna().astype(int).unique()):
        sx = x[num(x.target_season).eq(season)].copy()
        for week in sorted(num(sx.week).dropna().astype(int).unique()):
            train, test = sx[num(sx.week) < week], sx[num(sx.week) == week]
            if len(train) < min_train or test.empty:
                continue
            for target_name, (resid_col, actual_col, base_col, cap, shrink) in specs.items():
                # same-row baseline
                for idx, r in test.iterrows():
                    rows.append({
                        "season": season, "week": int(week), "team": r.team, "player_clean_key": r.player_clean_key,
                        "target": target_name, "group": "joint_baseline", "actual": num(pd.Series([r[actual_col]])).iloc[0],
                        "pred": num(pd.Series([r[base_col]])).iloc[0], "correction": 0.0,
                    })
                for name, features in groups.items():
                    correction = fit_correction(train, test, features, resid_col, cap, shrink)
                    pred = num(test[base_col]) + correction
                    for idx in test.index:
                        rows.append({
                            "season": season, "week": int(week), "team": test.loc[idx, "team"], "player_clean_key": test.loc[idx, "player_clean_key"],
                            "target": target_name, "group": name, "actual": num(test.loc[[idx], actual_col]).iloc[0],
                            "pred": pred.loc[idx], "correction": correction.loc[idx],
                        })
    return pd.DataFrame(rows)


def summarize_oos(trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (season, target, group), g in trace.groupby(["season", "target", "group"]):
        rows.append({"season": str(int(season)), "target": target, "group": group, **metrics(g.actual, g.pred)})
    for (target, group), g in trace.groupby(["target", "group"]):
        rows.append({"season": "combined", "target": target, "group": group, **metrics(g.actual, g.pred)})
    out = pd.DataFrame(rows)
    baseline = out[out.group.eq("joint_baseline")][["season", "target", "mae"]].rename(columns={"mae": "baseline_mae"})
    out = out.merge(baseline, on=["season", "target"], how="left")
    out["mae_improvement_vs_joint"] = out.baseline_mae - out.mae
    return out


def verdict(summary: pd.DataFrame) -> pd.DataFrame:
    new_groups = ["existing_plus_opportunity", "existing_plus_coverage", "existing_plus_new_defense", "existing_plus_new_interactions"]
    rows = []
    def mae(season: str, group: str) -> float:
        z = summary[(summary.season.astype(str).eq(str(season))) & summary.target.eq("pass_yards") & summary.group.eq(group)]
        return float(z.iloc[0].mae) if len(z) == 1 else np.nan
    for group in new_groups:
        combined_existing, combined_new = mae("combined", "existing_defense"), mae("combined", group)
        imp = combined_existing - combined_new
        imp24 = mae("2024", "existing_defense") - mae("2024", group)
        imp25 = mae("2025", "existing_defense") - mae("2025", group)
        qualifies = bool(np.isfinite(imp) and imp >= 0.50 and np.isfinite(imp24) and imp24 >= 0 and np.isfinite(imp25) and imp25 >= 0)
        rows.append({
            "group": group,
            "combined_mae_improvement_vs_existing_defense": imp,
            "season_2024_improvement_vs_existing_defense": imp24,
            "season_2025_improvement_vs_existing_defense": imp25,
            "qualifies_for_migration57_candidate_test": qualifies,
        })
    out = pd.DataFrame(rows)
    out["migration56_actionable_defense_signal"] = bool(out.qualifies_for_migration57_candidate_test.any())
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/backtests/qb_defensive_matchup"))
    p.add_argument("--seasons", default="2024,2025")
    p.add_argument("--min-train", type=int, default=80)
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/qb_defensive_matchup/summary"))
    a = p.parse_args()
    seasons = [int(v.strip()) for v in a.seasons.split(",") if v.strip()]
    if seasons != [2024, 2025]:
        raise RuntimeError("Migration 56 is locked to scored development seasons 2024,2025")

    frames = [season_frame(a.root, season) for season in seasons]
    x = pd.concat(frames, ignore_index=True, sort=False)
    ranking = signal_ranking(x)
    oos = run_oos(x, a.min_train)
    if oos.empty:
        raise RuntimeError("Migration 56 produced no OOS residual rows")
    summary = summarize_oos(oos)
    decision = verdict(summary)

    feature_cols = [c for c in x.columns if c.startswith(("def8_", "def3_", "deftrend_", "int_"))]
    coverage = pd.DataFrame({
        "feature": feature_cols,
        "n": len(x),
        "non_null": [int(x[c].notna().sum()) for c in feature_cols],
        "coverage": [float(x[c].notna().mean()) for c in feature_cols],
    }).sort_values("coverage")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir / "qb_defensive_matchup_frame.csv", index=False)
    ranking.to_csv(a.out_dir / "qb_defensive_signal_ranking.csv", index=False)
    oos.to_csv(a.out_dir / "qb_defensive_oos_trace.csv", index=False)
    summary.to_csv(a.out_dir / "qb_defensive_oos_summary.csv", index=False)
    coverage.to_csv(a.out_dir / "qb_defensive_feature_coverage.csv", index=False)
    decision.to_csv(a.out_dir / "qb_defensive_audit_verdict.csv", index=False)

    print("=== MIGRATION 56 OOS SUMMARY ===")
    print(summary[summary.target.eq("pass_yards")].to_string(index=False))
    print("\n=== TOP RESIDUAL DEFENSIVE SIGNALS ===")
    print(ranking.head(30).to_string(index=False))
    print("\n=== PRECOMMITTED M57 ELIGIBILITY ===")
    print(decision.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
