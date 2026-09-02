#!/usr/bin/env python3
"""RB STACK6F: qualify a team-level RB carry-pool forecast before player recomposition.

Frozen protocol:
  docs/migrations/RB_STACK6F_TEAM_RB_POOL_PLAN.md

Research only. Fit 2024, evaluate 2025. No sportsbook data are loaded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtest import evaluate_rb_stack2_enriched_allocation as s2

RIDGE_ALPHA = 10.0
START_WEEK = 6
RB_POS = {"RB", "HB", "FB"}
QB_POS = {"QB"}
FEATURES = [
    "team_prior1_rb_carries",
    "team_prior3_rb_carries",
    "team_prior5_rb_carries",
    "team_prior1_total_rush",
    "team_prior3_total_rush",
    "team_prior3_qb_rush_share",
    "team_prior3_pass_att",
    "team_prior3_play_proxy",
    "opp_prior1_rb_carries_allowed",
    "opp_prior3_rb_carries_allowed",
    "opp_prior3_total_rush_allowed",
    "opp_prior3_pass_att_faced",
]
ARMS = ["HISTORY_POOL", "P3_HISTORY_50"]


def num(x):
    return pd.to_numeric(x, errors="coerce")


def lower(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    z.columns = [str(c).strip().lower() for c in z.columns]
    return z


def one(root: Path, name: str) -> pd.DataFrame:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name} under {root}; found {len(hits)}")
    return lower(pd.read_csv(hits[0], low_memory=False))


def metric(y, p):
    q = pd.DataFrame({"y": num(y), "p": num(p)}).dropna()
    if q.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan}
    err = q.p - q.y
    corr = q.y.corr(q.p) if len(q) >= 3 and q.y.nunique() > 1 and q.p.nunique() > 1 else np.nan
    return {
        "n": int(len(q)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.square(err).mean())),
        "bias": float(err.mean()),
        "corr": float(corr) if pd.notna(corr) else np.nan,
    }


def schedule_long(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    x = lower(s2.pdx(nfl.load_schedules(seasons=seasons)))
    x["season"] = num(x.season)
    x["week"] = num(x.week)
    x = x.loc[x.season.isin(seasons) & x.week.between(1, 18)].copy()
    if "game_type" in x.columns:
        x = x.loc[x.game_type.fillna("").astype(str).str.upper().eq("REG")].copy()
    elif "season_type" in x.columns:
        st = x.season_type.fillna("").astype(str).str.upper()
        if st.eq("REG").any():
            x = x.loc[st.eq("REG")].copy()
    x["season"] = x.season.astype(int)
    x["week"] = x.week.astype(int)
    x["home_team"] = x.home_team.map(s2.tm)
    x["away_team"] = x.away_team.map(s2.tm)
    rows = []
    for _, r in x.iterrows():
        rows.append({"season": int(r.season), "week": int(r.week), "team": r.home_team, "opponent": r.away_team})
        rows.append({"season": int(r.season), "week": int(r.week), "team": r.away_team, "opponent": r.home_team})
    z = pd.DataFrame(rows).drop_duplicates(["season", "week", "team"])
    z["order"] = z.season * 100 + z.week
    return z


def team_weekly(logs: pd.DataFrame, sched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = logs.copy()
    for c in ["rushes", "pass_att"]:
        x[c] = num(x.get(c, 0)).fillna(0.0)
    x["position"] = x.position.fillna("").astype(str).str.upper()
    keys = ["season", "week", "team"]
    total = x.groupby(keys, as_index=False).agg(
        team_total_rush=("rushes", "sum"),
        team_pass_att=("pass_att", "sum"),
    )
    rb = x.loc[x.position.isin(RB_POS)].groupby(keys, as_index=False).agg(team_rb_carries=("rushes", "sum"))
    qb = x.loc[x.position.isin(QB_POS)].groupby(keys, as_index=False).agg(team_qb_carries=("rushes", "sum"))
    t = total.merge(rb, on=keys, how="left").merge(qb, on=keys, how="left")
    t[["team_rb_carries", "team_qb_carries"]] = t[["team_rb_carries", "team_qb_carries"]].fillna(0.0)
    t["team_qb_rush_share"] = np.where(t.team_total_rush.gt(0), t.team_qb_carries / t.team_total_rush, 0.0)
    t["team_play_proxy"] = t.team_total_rush + t.team_pass_att
    t = t.merge(sched[["season", "week", "team", "opponent", "order"]], on=keys, how="inner", validate="one_to_one")

    # Defense-allowed history: each offensive game becomes one observation for the defense faced.
    d = t[[
        "season", "week", "opponent", "team", "team_rb_carries", "team_total_rush", "team_pass_att", "order"
    ]].rename(columns={
        "opponent": "defense",
        "team": "offense",
        "team_rb_carries": "rb_carries_allowed",
        "team_total_rush": "total_rush_allowed",
        "team_pass_att": "pass_att_faced",
    })
    return t, d


def last_mean(g: pd.DataFrame, col: str, n: int) -> float:
    if g.empty or col not in g.columns:
        return np.nan
    s = num(g[col]).tail(n)
    return float(s.mean()) if s.notna().any() else np.nan


def last_value(g: pd.DataFrame, col: str) -> float:
    if g.empty or col not in g.columns:
        return np.nan
    s = num(g[col]).dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def make_features(target_games: pd.DataFrame, team_hist: pd.DataFrame, def_hist: pd.DataFrame) -> pd.DataFrame:
    tp = {k: g.sort_values("order") for k, g in team_hist.groupby("team") if k}
    dp = {k: g.sort_values("order") for k, g in def_hist.groupby("defense") if k}
    rows = []
    for _, r in target_games.iterrows():
        order = int(r.season) * 100 + int(r.week)
        team = str(r.team)
        opp = str(r.opponent)
        th = tp.get(team, pd.DataFrame())
        th = th.loc[num(th.order).lt(order)].copy() if not th.empty else th
        dh = dp.get(opp, pd.DataFrame())
        dh = dh.loc[num(dh.order).lt(order)].copy() if not dh.empty else dh
        rows.append({
            "season": int(r.season),
            "week": int(r.week),
            "team": team,
            "opponent": opp,
            "target_order": order,
            "team_max_source_order": float(num(th.order).max()) if len(th) else np.nan,
            "opp_max_source_order": float(num(dh.order).max()) if len(dh) else np.nan,
            "team_history_games": int(len(th)),
            "opp_history_games": int(len(dh)),
            "team_prior1_rb_carries": last_value(th, "team_rb_carries"),
            "team_prior3_rb_carries": last_mean(th, "team_rb_carries", 3),
            "team_prior5_rb_carries": last_mean(th, "team_rb_carries", 5),
            "team_prior1_total_rush": last_value(th, "team_total_rush"),
            "team_prior3_total_rush": last_mean(th, "team_total_rush", 3),
            "team_prior3_qb_rush_share": last_mean(th, "team_qb_rush_share", 3),
            "team_prior3_pass_att": last_mean(th, "team_pass_att", 3),
            "team_prior3_play_proxy": last_mean(th, "team_play_proxy", 3),
            "opp_prior1_rb_carries_allowed": last_value(dh, "rb_carries_allowed"),
            "opp_prior3_rb_carries_allowed": last_mean(dh, "rb_carries_allowed", 3),
            "opp_prior3_total_rush_allowed": last_mean(dh, "total_rush_allowed", 3),
            "opp_prior3_pass_att_faced": last_mean(dh, "pass_att_faced", 3),
        })
    z = pd.DataFrame(rows)
    safe = pd.Series(True, index=z.index)
    for c in ["team_max_source_order", "opp_max_source_order"]:
        v = num(z[c])
        safe &= v.isna() | v.lt(num(z.target_order))
    z["asof_leakage_safe"] = safe.astype(int)
    return z


def prepare_p3_team_pool(casebook: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    z = casebook.copy()
    z["season"] = num(z.get("season", 2025)).fillna(2025).astype(int)
    z["week"] = num(z.week).astype(int)
    z["team"] = z.team.map(s2.tm)
    if "parent_att" not in z.columns:
        if "enriched_att" not in z.columns:
            raise RuntimeError("STACK6F casebook missing parent_att/enriched_att")
        z["parent_att"] = np.where(z.week.eq(1), num(z.get("stack_att")), num(z.enriched_att))
    z["parent_att"] = num(z.parent_att)
    p = z.groupby(["season", "week", "team"], as_index=False).agg(
        p3_team_rb_pool=("parent_att", "sum"),
        p3_player_rows=("parent_att", "size"),
    )
    p = p.merge(sched[["season", "week", "team", "opponent"]], on=["season", "week", "team"], how="left", validate="one_to_one")
    if p.opponent.isna().any():
        raise RuntimeError("STACK6F P3 team-week missing schedule opponent")
    return p


def ridge():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=RIDGE_ALPHA)),
    ])


def score_scopes(z: pd.DataFrame) -> pd.DataFrame:
    masks = {
        "w6_18": z.week.ge(START_WEEK),
        "w13_18": z.week.ge(13),
        "all_weeks": pd.Series(True, index=z.index),
    }
    preds = {
        "P3_PARENT_POOL": "p3_team_rb_pool",
        "HISTORY_POOL": "pred_history_pool",
        "P3_HISTORY_50": "pred_p3_history_50",
    }
    rows = []
    for scope, mask in masks.items():
        g = z.loc[mask]
        for arm, col in preds.items():
            rows.append({"scope": scope, "arm": arm, **metric(g.actual_team_rb_carries, g[col])})
    return pd.DataFrame(rows)


def retention(scores: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    def r(scope, arm):
        q = scores.loc[scores.scope.eq(scope) & scores.arm.eq(arm)]
        if q.empty:
            raise RuntimeError(f"STACK6F missing score {scope}/{arm}")
        return q.iloc[0]

    base = r("w6_18", "P3_PARENT_POOL")
    late_base = r("w13_18", "P3_PARENT_POOL")
    rows = []
    for arm in ARMS:
        cur = r("w6_18", arm)
        late = r("w13_18", arm)
        mae_gain = float(base.mae - cur.mae)
        rmse_gain = float(base.rmse - cur.rmse)
        corr_gain = float(cur.corr - base.corr)
        late_gain = float(late_base.mae - late.mae)
        abs_bias = abs(float(cur.bias))
        passed = int(
            mae_gain >= 0.30
            and rmse_gain > 0
            and corr_gain >= 0.05
            and abs_bias <= 0.50
            and late_gain > 0
        )
        rows.append({
            "arm": arm,
            "team_carry_mae_gain": mae_gain,
            "team_carry_rmse_gain": rmse_gain,
            "team_carry_corr_gain": corr_gain,
            "team_carry_abs_bias": abs_bias,
            "late_team_carry_mae_gain": late_gain,
            "gate_mae_gain_ge_030": int(mae_gain >= 0.30),
            "gate_rmse_gain_gt_0": int(rmse_gain > 0),
            "gate_corr_gain_ge_005": int(corr_gain >= 0.05),
            "gate_abs_bias_le_050": int(abs_bias <= 0.50),
            "gate_late_gain_gt_0": int(late_gain > 0),
            "gate_pass": passed,
        })
    g = pd.DataFrame(rows)
    passing = g.loc[g.gate_pass.eq(1)].copy()
    selected = "NONE"
    if len(passing):
        best = float(passing.team_carry_mae_gain.max())
        blend = passing.loc[passing.arm.eq("P3_HISTORY_50")]
        if len(blend) and float(blend.iloc[0].team_carry_mae_gain) >= best - 0.10:
            selected = "P3_HISTORY_50"
        else:
            selected = str(passing.sort_values(["team_carry_mae_gain", "arm"], ascending=[False, True]).iloc[0].arm)
    g["selected_arm"] = selected
    return g, selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack6-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    seasons = [2023, 2024, 2025]
    logs = s2.load_weekly_logs(seasons)
    sched = schedule_long(seasons)
    team_hist, def_hist = team_weekly(logs, sched)

    train_games = sched.loc[sched.season.eq(2024), ["season", "week", "team", "opponent"]].copy()
    train = make_features(train_games, team_hist, def_hist)
    actual24 = team_hist.loc[team_hist.season.eq(2024), ["season", "week", "team", "team_rb_carries"]].rename(columns={"team_rb_carries": "actual_team_rb_carries"})
    train = train.merge(actual24, on=["season", "week", "team"], how="inner", validate="one_to_one")

    p3 = prepare_p3_team_pool(one(a.stack6_root, "stack6_2025_casebook.csv"), sched)
    test = make_features(p3[["season", "week", "team", "opponent"]], team_hist, def_hist)
    test = test.merge(p3[["season", "week", "team", "p3_team_rb_pool", "p3_player_rows"]], on=["season", "week", "team"], how="left", validate="one_to_one")
    actual25 = team_hist.loc[team_hist.season.eq(2025), ["season", "week", "team", "team_rb_carries"]].rename(columns={"team_rb_carries": "actual_team_rb_carries"})
    test = test.merge(actual25, on=["season", "week", "team"], how="left", validate="one_to_one")

    if int(train.asof_leakage_safe.min()) != 1 or int(test.asof_leakage_safe.min()) != 1:
        raise RuntimeError("STACK6F strict-prior leakage audit failed")
    if len(FEATURES) != 12:
        raise RuntimeError(f"STACK6F frozen feature contract changed: {len(FEATURES)}")
    if num(test.actual_team_rb_carries).isna().any():
        raise RuntimeError("STACK6F missing true 2025 team RB carry target")

    model = ridge()
    fit = train.loc[num(train.actual_team_rb_carries).notna()].copy()
    model.fit(fit[FEATURES], num(fit.actual_team_rb_carries))
    raw = model.predict(test[FEATURES])
    test["pred_history_pool"] = np.clip(raw, 0.0, None)
    test["pred_p3_history_50"] = 0.50 * num(test.p3_team_rb_pool) + 0.50 * num(test.pred_history_pool)

    coverage = pd.DataFrame([{
        "training_team_games_2024": int(len(fit)),
        "evaluation_team_games_2025": int(len(test)),
        "evaluation_w6_18_team_games": int(test.week.ge(START_WEEK).sum()),
        "feature_count": int(len(FEATURES)),
        "train_leakage_pass_rate": float(train.asof_leakage_safe.mean()),
        "test_leakage_pass_rate": float(test.asof_leakage_safe.mean()),
        "team_history_coverage_w6_18": float(test.loc[test.week.ge(START_WEEK), "team_history_games"].gt(0).mean()),
        "opp_history_coverage_w6_18": float(test.loc[test.week.ge(START_WEEK), "opp_history_games"].gt(0).mean()),
        "sportsbook_used": 0,
        "target_game_injury_used": 0,
        "target_game_participation_used": 0,
    }])

    scores = score_scopes(test)
    gates, selected = retention(scores)
    if selected == "HISTORY_POOL":
        disposition = "STACK6F_RETAIN_HISTORY_POOL"
    elif selected == "P3_HISTORY_50":
        disposition = "STACK6F_RETAIN_P3_HISTORY_50"
    else:
        disposition = "STACK6F_NO_RETAINABLE_TEAM_POOL_MODEL"
    disp = pd.DataFrame([{
        "selected_arm": selected,
        "passing_arm_count": int(gates.gate_pass.sum()),
        "disposition": disposition,
        "ridge_alpha": RIDGE_ALPHA,
        "fit_season": 2024,
        "evaluation_season": 2025,
        "feature_count": len(FEATURES),
        "fixed_blend_weight_p3": 0.50,
        "fixed_blend_weight_history": 0.50,
        "hyperparameter_search": 0,
        "feature_search": 0,
        "weight_search": 0,
        "threshold_search": 0,
        "population_search": 0,
        "sportsbook_used": 0,
        "production_change": 0,
        "player_recomposition_authorized": int(selected != "NONE"),
    }])

    coverage.to_csv(a.out_dir / "stack6f_coverage.csv", index=False)
    scores.to_csv(a.out_dir / "stack6f_score_table.csv", index=False)
    gates.to_csv(a.out_dir / "stack6f_retention_gates.csv", index=False)
    disp.to_csv(a.out_dir / "stack6f_disposition.csv", index=False)
    test.to_csv(a.out_dir / "stack6f_2025_team_casebook.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv(a.out_dir / "stack6f_features.csv", index=False)

    print("=== STACK6F coverage ===")
    print(coverage.to_string(index=False))
    print("=== STACK6F team-pool scores ===")
    print(scores.to_string(index=False))
    print("=== STACK6F frozen retention gates ===")
    print(gates.to_string(index=False))
    print("=== STACK6F disposition ===")
    print(disp.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
