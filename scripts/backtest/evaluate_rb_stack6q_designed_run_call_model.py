#!/usr/bin/env python3
"""RB STACK6Q: frozen pregame designed-run-call model qualification."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

STATES = ("lead", "neutral", "trail")
START_WEEK = 6
ALPHA = 0.75
PSEUDO = 24.0
EXPECTED_2025 = 544
EXPECTED_W6 = 388
TEAM_MAP = {"JAX": "JAC", "LAR": "LA", "STL": "LA", "OAK": "LV", "SD": "LAC"}


def num(v):
    return pd.to_numeric(v, errors="coerce")


def canon(v):
    s = str(v).strip().upper()
    return TEAM_MAP.get(s, s)


def lower(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    z.columns = [str(c).strip().lower() for c in z.columns]
    return z


def one(root: Path, name: str) -> pd.DataFrame:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(hits)}")
    return lower(pd.read_csv(hits[0], low_memory=False))


def ratio(g: pd.DataFrame, ncol: str, dcol: str, n: int | None = None) -> float:
    if n is not None:
        g = g.tail(n)
    if g.empty:
        return np.nan
    den = num(g.get(dcol, 0)).fillna(0).sum()
    if den <= 0:
        return np.nan
    return float(num(g.get(ncol, 0)).fillna(0).sum() / den)


def load_pbp() -> pd.DataFrame:
    import nflreadpy as nfl

    p = lower(nfl.load_pbp(seasons=[2023, 2024, 2025]).to_pandas())
    if "season_type" in p.columns:
        reg = p.loc[p.season_type.astype(str).str.upper().eq("REG")].copy()
        if not reg.empty:
            p = reg
    required = {"season", "week", "posteam", "defteam", "rush_attempt", "qb_dropback"}
    missing = required - set(p.columns)
    if missing:
        raise RuntimeError(f"STACK6Q PBP missing columns: {sorted(missing)}")

    for c in ["rush_attempt", "qb_dropback", "qb_scramble", "qb_kneel", "down", "game_seconds_remaining"]:
        p[c] = num(p[c]) if c in p.columns else 0.0
    p["team"] = p.posteam.map(canon)
    p["defense"] = p.defteam.map(canon)
    p["off_play"] = (p.rush_attempt.eq(1) | p.qb_dropback.eq(1)).astype(int)
    p = p.loc[p.off_play.eq(1) & p.team.ne("") & p.defense.ne("")].copy()

    if "score_differential" in p.columns:
        diff = num(p.score_differential)
    elif {"posteam_score", "defteam_score"}.issubset(p.columns):
        diff = num(p.posteam_score) - num(p.defteam_score)
    else:
        raise RuntimeError("STACK6Q PBP missing score differential")
    p["score_diff"] = diff.fillna(0.0)
    p["state"] = np.select(
        [p.score_diff.gt(3.0), p.score_diff.lt(-3.0)], ["lead", "trail"], default="neutral"
    )

    p["scramble"] = num(p.qb_scramble).fillna(0).eq(1).astype(int)
    p["kneel"] = num(p.qb_kneel).fillna(0).eq(1).astype(int)
    p["designed"] = (
        p.rush_attempt.eq(1) & p.scramble.eq(0) & p.kneel.eq(0)
    ).astype(int)

    passer_col = "passer_player_name" if "passer_player_name" in p.columns else None
    rusher_col = "rusher_player_name" if "rusher_player_name" in p.columns else None
    if passer_col is None or rusher_col is None:
        raise RuntimeError("STACK6Q requires passer_player_name and rusher_player_name")
    qb_names = set(p[passer_col].dropna().astype(str))
    p["qb_designed"] = (
        p.designed.eq(1) & p[rusher_col].notna() & p[rusher_col].astype(str).isin(qb_names)
    ).astype(int)
    p["neutral_early_play"] = (
        p.state.eq("neutral")
        & num(p.down).isin([1, 2])
        & (num(p.game_seconds_remaining).isna() | num(p.game_seconds_remaining).gt(900))
    ).astype(int)
    p["neutral_early_designed"] = (p.neutral_early_play.eq(1) & p.designed.eq(1)).astype(int)
    return p


def build_games(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (season, week, team, defense), g in p.groupby(["season", "week", "team", "defense"], dropna=False):
        rec = {
            "season": int(season), "week": int(week), "team": str(team), "opponent": str(defense),
            "off_plays": float(len(g)), "designed": float(g.designed.sum()),
            "qb_designed": float(g.qb_designed.sum()),
            "neutral_early_plays": float(g.neutral_early_play.sum()),
            "neutral_early_designed": float(g.neutral_early_designed.sum()),
            "rush_attempts": float(g.rush_attempt.sum()),
        }
        for s in STATES:
            q = g.loc[g.state.eq(s)]
            rec[f"{s}_plays"] = float(len(q))
            rec[f"{s}_designed"] = float(q.designed.sum())
            rec[f"{s}_scramble"] = float(q.scramble.sum())
            rec[f"{s}_kneel"] = float(q.kneel.sum())
        rows.append(rec)
    off = pd.DataFrame(rows).sort_values(["season", "week", "team"]).reset_index(drop=True)
    if off.empty or off.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("STACK6Q offensive game table invalid")

    drows = []
    for (season, week, defense), g in p.groupby(["season", "week", "defense"], dropna=False):
        rec = {
            "season": int(season), "week": int(week), "defense": str(defense),
            "off_plays_allowed": float(len(g)), "designed_allowed": float(g.designed.sum()),
            "qb_designed_allowed": float(g.qb_designed.sum()),
        }
        for s in STATES:
            q = g.loc[g.state.eq(s)]
            rec[f"{s}_plays_allowed"] = float(len(q))
            rec[f"{s}_designed_allowed"] = float(q.designed.sum())
        drows.append(rec)
    deff = pd.DataFrame(drows).sort_values(["season", "week", "defense"]).reset_index(drop=True)
    if deff.empty or deff.duplicated(["season", "week", "defense"]).any():
        raise RuntimeError("STACK6Q defensive game table invalid")
    return off, deff


def prior_mask(df: pd.DataFrame, season: int, week: int) -> pd.Series:
    s = num(df.season)
    w = num(df.week)
    return s.lt(season) | (s.eq(season) & w.lt(week))


def league_state_nuisance(off: pd.DataFrame, season: int, week: int, state: str, kind: str) -> float:
    g = off.loc[prior_mask(off, season, week)]
    den = num(g.get(f"{state}_plays", 0)).fillna(0).sum()
    val = num(g.get(f"{state}_{kind}", 0)).fillna(0).sum()
    return float(val / den) if den > 0 else 0.0


def build_state_row(target: pd.Series, off: pd.DataFrame, deff: pd.DataFrame, state: str) -> dict:
    season, week = int(target.season), int(target.week)
    team, opp = canon(target.team), canon(target.opponent)
    tg = off.loc[off.team.eq(team) & prior_mask(off, season, week)].sort_values(["season", "week"])
    dg = deff.loc[deff.defense.eq(opp) & prior_mask(deff, season, week)].sort_values(["season", "week"])

    rec = {
        "season": season, "week": week, "team": team, "opponent": opp,
        "team_state_designed_rate_p1": ratio(tg, f"{state}_designed", f"{state}_plays", 1),
        "team_state_designed_rate_p3": ratio(tg, f"{state}_designed", f"{state}_plays", 3),
        "team_state_designed_rate_p5": ratio(tg, f"{state}_designed", f"{state}_plays", 5),
        "team_state_plays_p5": float(num(tg.tail(5).get(f"{state}_plays", 0)).fillna(0).sum()) if len(tg) else 0.0,
        "team_overall_designed_rate_p3": ratio(tg, "designed", "off_plays", 3),
        "team_overall_designed_rate_p5": ratio(tg, "designed", "off_plays", 5),
        "team_neutral_early_designed_rate_p3": ratio(tg, "neutral_early_designed", "neutral_early_plays", 3),
        "opp_state_designed_allowed_p1": ratio(dg, f"{state}_designed_allowed", f"{state}_plays_allowed", 1),
        "opp_state_designed_allowed_p3": ratio(dg, f"{state}_designed_allowed", f"{state}_plays_allowed", 3),
        "opp_state_designed_allowed_p5": ratio(dg, f"{state}_designed_allowed", f"{state}_plays_allowed", 5),
        "opp_overall_designed_allowed_p3": ratio(dg, "designed_allowed", "off_plays_allowed", 3),
        "opp_overall_designed_allowed_p5": ratio(dg, "designed_allowed", "off_plays_allowed", 5),
        "team_qb_designed_share_p3": ratio(tg, "qb_designed", "off_plays", 3),
        "opp_qb_designed_share_allowed_p3": ratio(dg, "qb_designed_allowed", "off_plays_allowed", 3),
        "pred_mean_margin": float(num(pd.Series([target.pred_mean_margin])).iloc[0]),
        "pred_final_margin": float(num(pd.Series([target.pred_final_margin])).iloc[0]),
        "margin_blend": float(num(pd.Series([target.margin_blend])).iloc[0]),
        "margin_abs": float(num(pd.Series([target.margin_abs])).iloc[0]),
        "home": float(num(pd.Series([target.home])).iloc[0]),
        "pred_state_share": float(num(pd.Series([target[f"pred_{state}_play_share"]])).iloc[0]),
    }

    prior_league_scr = league_state_nuisance(off, season, week, state, "scramble")
    prior_league_kneel = league_state_nuisance(off, season, week, state, "kneel")
    recent = tg.tail(5)
    state_plays = num(recent.get(f"{state}_plays", 0)).fillna(0).sum() if len(recent) else 0.0
    scr = num(recent.get(f"{state}_scramble", 0)).fillna(0).sum() if len(recent) else 0.0
    kneel = num(recent.get(f"{state}_kneel", 0)).fillna(0).sum() if len(recent) else 0.0
    rec["prior_scramble_rate"] = float((scr + PSEUDO * prior_league_scr) / (state_plays + PSEUDO))
    rec["prior_kneel_rate"] = float((kneel + PSEUDO * prior_league_kneel) / (state_plays + PSEUDO))

    actual = off.loc[
        off.season.eq(season) & off.week.eq(week) & off.team.eq(team)
    ]
    if len(actual) == 1:
        plays = float(actual[f"{state}_plays"].iloc[0])
        designed = float(actual[f"{state}_designed"].iloc[0])
        rec["target_state_plays"] = plays
        rec["target_designed_rate"] = designed / plays if plays > 0 else np.nan
    else:
        rec["target_state_plays"] = np.nan
        rec["target_designed_rate"] = np.nan
    return rec


FEATURES = [
    "team_state_designed_rate_p1", "team_state_designed_rate_p3", "team_state_designed_rate_p5",
    "team_state_plays_p5", "team_overall_designed_rate_p3", "team_overall_designed_rate_p5",
    "team_neutral_early_designed_rate_p3", "opp_state_designed_allowed_p1",
    "opp_state_designed_allowed_p3", "opp_state_designed_allowed_p5",
    "opp_overall_designed_allowed_p3", "opp_overall_designed_allowed_p5",
    "team_qb_designed_share_p3", "opp_qb_designed_share_allowed_p3",
    "pred_mean_margin", "pred_final_margin", "margin_blend", "margin_abs", "home", "pred_state_share",
]


def learner() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=10.0)),
    ])


def metric(y, pred) -> dict:
    y, pred = num(y), num(pred)
    ok = y.notna() & pred.notna()
    y, pred = y.loc[ok], pred.loc[ok]
    e = pred - y
    return {
        "n": int(len(y)),
        "mae": float(e.abs().mean()) if len(e) else np.nan,
        "rmse": float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan,
        "bias": float(e.mean()) if len(e) else np.nan,
        "corr": float(pred.corr(y)) if len(y) >= 3 and pred.nunique() > 1 and y.nunique() > 1 else np.nan,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m94c-root", type=Path, required=True)
    ap.add_argument("--stack6h-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    hold = one(a.m94c_root, "m94c_2024_holdout_trace.csv")
    eval25 = one(a.m94c_root, "m94c_2025_team_trace.csv")
    h = one(a.stack6h_root, "stack6h_team_trace.csv")
    for d in [hold, eval25, h]:
        d["season"] = num(d.season).astype(int)
        d["week"] = num(d.week).astype(int)
        d["team"] = d.team.map(canon)
        if "opponent" in d.columns:
            d["opponent"] = d.opponent.map(canon)

    need_env = [
        "opponent", "pred_mean_margin", "pred_final_margin", "margin_blend", "margin_abs", "home",
        "pred_lead_play_share", "pred_neutral_play_share", "pred_trail_play_share",
        "pred_off_plays", "baseline_team_rush_att", "candidate_team_rush_att", "actual_team_rush_att",
        "actual_rush_att_pbp",
    ]
    missing = [c for c in need_env if c not in eval25.columns or c not in hold.columns]
    if missing:
        raise RuntimeError(f"STACK6Q M94C traces missing required fields: {missing}")

    pbp = load_pbp()
    off, deff = build_games(pbp)

    models = {}
    train_meta = []
    state_eval_frames = {}
    feature_coverage_rows = []
    for state in STATES:
        tr = pd.DataFrame([build_state_row(r, off, deff, state) for _, r in hold.iterrows()])
        te = pd.DataFrame([build_state_row(r, off, deff, state) for _, r in eval25.iterrows()])
        elig = tr.target_state_plays.fillna(0).gt(0) & tr.target_designed_rate.notna()
        if int(elig.sum()) < 50:
            raise RuntimeError(f"STACK6Q insufficient {state} training rows: {int(elig.sum())}")
        model = learner()
        model.fit(tr.loc[elig, FEATURES], tr.loc[elig, "target_designed_rate"])
        te[f"pred_{state}_designed_rate"] = np.clip(model.predict(te[FEATURES]), 0.0, 0.80)
        models[state] = model
        state_eval_frames[state] = te
        train_meta.append({
            "state": state, "training_rows": int(elig.sum()),
            "training_week_min": int(tr.loc[elig, "week"].min()),
            "training_week_max": int(tr.loc[elig, "week"].max()),
            "feature_count": len(FEATURES),
        })
        for f in FEATURES:
            feature_coverage_rows.append({
                "state": state, "feature": f,
                "train_nonnull_rate": float(tr.loc[elig, f].notna().mean()),
                "eval_nonnull_rate": float(te[f].notna().mean()),
            })

    t = eval25.copy()
    keys = ["season", "week", "team", "opponent"]
    for state in STATES:
        z = state_eval_frames[state][keys + [
            f"pred_{state}_designed_rate", "prior_scramble_rate", "prior_kneel_rate"
        ]].rename(columns={
            "prior_scramble_rate": f"prior_{state}_scramble_rate",
            "prior_kneel_rate": f"prior_{state}_kneel_rate",
        })
        t = t.merge(z, on=keys, how="left", validate="one_to_one")

    effective = pd.Series(0.0, index=t.index, dtype=float)
    for state in STATES:
        total_state = (
            num(t[f"pred_{state}_designed_rate"]).fillna(0)
            + num(t[f"prior_{state}_scramble_rate"]).fillna(0)
            + num(t[f"prior_{state}_kneel_rate"]).fillna(0)
        )
        t[f"stack6q_{state}_total_rush_rate"] = total_state
        effective += num(t[f"pred_{state}_play_share"]).fillna(0) * total_state
    t["stack6q_effective_rush_rate"] = effective
    t["stack6q_structured_team_rush"] = num(t.pred_off_plays) * effective
    t["stack6q_team_rush_att"] = (
        (1 - ALPHA) * num(t.baseline_team_rush_att)
        + ALPHA * t.stack6q_structured_team_rush
    ).clip(8.0, 50.0)

    hcols = ["season", "week", "team", "t_hat", "pool_over_5", "pool_under_5"]
    missing_h = [c for c in hcols if c not in h.columns]
    if missing_h:
        raise RuntimeError(f"STACK6Q STACK6H trace missing {missing_h}")
    t = t.merge(h[hcols], on=["season", "week", "team"], how="inner", validate="one_to_one")

    # Source / identity checks.
    off25 = off.loc[off.season.eq(2025), ["season", "week", "team", "rush_attempts"]].copy()
    t = t.merge(off25, on=["season", "week", "team"], how="left", validate="one_to_one")
    m94c_vs_h = float((num(t.candidate_team_rush_att) - num(t.t_hat)).abs().max())
    pbp_vs_m94c = float((num(t.rush_attempts) - num(t.actual_rush_att_pbp)).abs().max())
    w = t.loc[t.week.ge(START_WEEK)].copy()

    def scores(frame: pd.DataFrame, pop: str) -> pd.DataFrame:
        rows = []
        for arm, col in [("M94C", "candidate_team_rush_att"), ("STACK6Q", "stack6q_team_rush_att")]:
            rows.append({"population": pop, "arm": arm, **metric(frame.actual_team_rush_att, frame[col])})
        z = pd.DataFrame(rows)
        b = z.loc[z.arm.eq("M94C")].iloc[0]
        q = z.loc[z.arm.eq("STACK6Q")].iloc[0]
        z["mae_gain_vs_m94c"] = float(b.mae) - z.mae
        z["rmse_gain_vs_m94c"] = float(b.rmse) - z.rmse
        z["corr_gain_vs_m94c"] = z.corr - float(b["corr"])
        return z

    overall = scores(w, "ALL_W6_18")
    over5 = scores(w.loc[w.pool_over_5.eq(1)], "POOL_OVER_5")
    under5 = scores(w.loc[w.pool_under_5.eq(1)], "POOL_UNDER_5")
    late = scores(w.loc[w.week.ge(13)], "W13_18")
    scoring = pd.concat([overall, over5, under5, late], ignore_index=True)

    def row(pop, arm="STACK6Q"):
        return scoring.loc[scoring.population.eq(pop) & scoring.arm.eq(arm)].iloc[0]

    q_all = row("ALL_W6_18")
    b_all = row("ALL_W6_18", "M94C")
    gates = pd.DataFrame([
        {"gate": "mae_gain_ge_0_20", "value": float(q_all.mae_gain_vs_m94c), "pass": int(float(q_all.mae_gain_vs_m94c) >= 0.20)},
        {"gate": "rmse_gain_gt_0", "value": float(q_all.rmse_gain_vs_m94c), "pass": int(float(q_all.rmse_gain_vs_m94c) > 0)},
        {"gate": "corr_gain_ge_0_02", "value": float(q_all.corr_gain_vs_m94c), "pass": int(float(q_all.corr_gain_vs_m94c) >= 0.02)},
        {"gate": "abs_bias_worsening_le_0_25", "value": float(abs(q_all.bias) - abs(b_all.bias)), "pass": int(float(abs(q_all.bias) - abs(b_all.bias)) <= 0.25)},
        {"gate": "pool_over5_mae_gain_gt_0", "value": float(row("POOL_OVER_5").mae_gain_vs_m94c), "pass": int(float(row("POOL_OVER_5").mae_gain_vs_m94c) > 0)},
        {"gate": "pool_under5_mae_gain_gt_0", "value": float(row("POOL_UNDER_5").mae_gain_vs_m94c), "pass": int(float(row("POOL_UNDER_5").mae_gain_vs_m94c) > 0)},
        {"gate": "w13_18_mae_gain_gt_0", "value": float(row("W13_18").mae_gain_vs_m94c), "pass": int(float(row("W13_18").mae_gain_vs_m94c) > 0)},
    ])

    coverage = pd.DataFrame(feature_coverage_rows)
    strict_prior_flag = 1
    integrity_pass = int(
        len(t) == EXPECTED_2025
        and len(w) == EXPECTED_W6
        and m94c_vs_h <= 1e-9
        and pbp_vs_m94c <= 1e-9
        and strict_prior_flag == 1
        and len(FEATURES) == 20
    )
    all_gates = int(gates["pass"].eq(1).all())
    if not integrity_pass:
        disposition = "STACK6Q_INTEGRITY_FAILURE_DO_NOT_INTERPRET"
    elif all_gates:
        disposition = "STACK6Q_DESIGNED_RUN_MODEL_QUALIFIED"
    else:
        disposition = "STACK6Q_DESIGNED_RUN_MODEL_NOT_QUALIFIED"

    integrity = pd.DataFrame([{
        "m94c_2024_holdout_rows": len(hold),
        "m94c_2025_rows": len(eval25),
        "stack6h_rows": len(h),
        "joined_2025_rows": len(t),
        "w6_18_rows": len(w),
        "m94c_vs_stack6h_t_hat_max_abs_diff": m94c_vs_h,
        "pbp_vs_m94c_actual_rush_max_abs_diff": pbp_vs_m94c,
        "strict_prior_construction": strict_prior_flag,
        "frozen_feature_count": len(FEATURES),
        "fixed_ridge_alpha": 10.0,
        "fixed_m94c_blend_alpha": ALPHA,
        "feature_search": 0,
        "hyperparameter_search": 0,
        "model_family_search": 0,
        "threshold_search": 0,
        "sportsbook_inputs": 0,
        "target_week_participation_or_injury_inputs": 0,
        "target_game_pbp_used_as_label_or_grading_only": 1,
        "integrity_pass": integrity_pass,
    }])
    disposition_df = pd.DataFrame([{
        "scientific_gates_passed": int(gates["pass"].sum()),
        "scientific_gate_count": len(gates),
        "all_scientific_gates_pass": all_gates,
        "disposition": disposition,
        "production_change": 0,
        "p3_recomposition_authorized": int(integrity_pass and all_gates),
    }])

    pd.DataFrame(train_meta).to_csv(a.out_dir / "stack6q_training_meta.csv", index=False)
    coverage.to_csv(a.out_dir / "stack6q_feature_coverage.csv", index=False)
    t.to_csv(a.out_dir / "stack6q_2025_team_trace.csv", index=False)
    scoring.to_csv(a.out_dir / "stack6q_scores.csv", index=False)
    gates.to_csv(a.out_dir / "stack6q_gates.csv", index=False)
    integrity.to_csv(a.out_dir / "stack6q_integrity.csv", index=False)
    disposition_df.to_csv(a.out_dir / "stack6q_disposition.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv(a.out_dir / "stack6q_features.csv", index=False)

    print("=== STACK6Q training meta ===")
    print(pd.DataFrame(train_meta).to_string(index=False))
    print("=== STACK6Q integrity ===")
    print(integrity.to_string(index=False))
    print("=== STACK6Q scores ===")
    print(scoring.to_string(index=False))
    print("=== STACK6Q gates ===")
    print(gates.to_string(index=False))
    print("=== STACK6Q disposition ===")
    print(disposition_df.to_string(index=False))
    print(f"STACK6Q_DISPOSITION={disposition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
