#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ALPHA = 20.0
LAST_N = 8
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 5621
TOL = 1e-6
KEYS = ["season", "week", "team", "player_clean_key"]

PENALTY_FEATURES = [
    "team_first_down_penalty_rate_l8",
    "oppdef_first_down_penalty_allowed_rate_l8",
]
FOURTH_FEATURES = [
    "team_fourth_go_rate_l8",
    "team_fourth_conversion_rate_l8",
    "team_fourth_pass_share_l8",
    "oppdef_fourth_go_allowed_rate_l8",
    "oppdef_fourth_conversion_allowed_rate_l8",
    "oppdef_fourth_pass_allowed_share_l8",
]
FAMILY_FEATURES = {
    "PENALTY_DRIVE_EXTENSION": PENALTY_FEATURES,
    "FOURTH_DOWN_AGGRESSION": FOURTH_FEATURES,
}


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def canon_keys(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["season"] = num(x["season"])
    x["week"] = num(x["week"])
    x["team"] = x["team"].fillna("").astype(str).str.upper().str.strip()
    x["player_clean_key"] = x["player_clean_key"].fillna("").astype(str).str.strip()
    return x


def load_target_context(m89_root: Path) -> pd.DataFrame:
    cols = KEYS + ["opponent"]
    x = pd.read_csv(one(m89_root, "m89_corrected_qb_common_trace.csv"), usecols=cols, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    x = canon_keys(x)
    x["opponent"] = x["opponent"].fillna("").astype(str).str.upper().str.strip()
    x = x.loc[x["season"].eq(2024)].copy()
    if len(x) != 444 or x.duplicated(KEYS).any() or x["opponent"].eq("").any():
        raise RuntimeError(f"M89 2024 opponent-context integrity failure rows={len(x)}")
    return x


def load_chain(chain_root: Path) -> pd.DataFrame:
    cols = KEYS + [
        "actual_pass_yards", "actual_attempts", "pred_attempts", "pred_ypa",
        "football_synthesis", "pred_D", "pred_C", "pred_S", "actual_D",
    ]
    x = pd.read_csv(one(chain_root, "qb_opportunity_chain_casebook.csv"), usecols=cols, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    x = canon_keys(x)
    for c in ["actual_pass_yards", "actual_attempts", "pred_attempts", "pred_ypa", "football_synthesis", "pred_d", "pred_c", "pred_s", "actual_d"]:
        x[c] = num(x[c])
    x = x.loc[x["season"].eq(2024)].copy()
    if len(x) != 444 or x.duplicated(KEYS).any():
        raise RuntimeError(f"chain 2024 integrity failure rows={len(x)}")
    return x


def load_source_games(source_root: Path) -> pd.DataFrame:
    g = pd.read_csv(one(source_root, "pbp_team_game_candidate_counts.csv"), low_memory=False)
    g.columns = [str(c).strip().lower() for c in g.columns]
    required = {
        "season", "week", "game_id", "team", "opponent", "offensive_pbp_rows",
        "first_down_penalties", "fourth_down_decisions", "fourth_down_go_attempts",
        "fourth_down_conversions", "fourth_down_go_passes", "fourth_down_go_runs",
    }
    missing = sorted(required - set(g.columns))
    if missing:
        raise RuntimeError(f"V1B source-game artifact missing {missing}")
    g["season"] = num(g["season"])
    g["week"] = num(g["week"])
    g["team"] = g["team"].fillna("").astype(str).str.upper().str.strip()
    g["opponent"] = g["opponent"].fillna("").astype(str).str.upper().str.strip()
    for c in sorted(required - {"game_id", "team", "opponent"}):
        if c not in {"season", "week"}:
            g[c] = num(g[c]).fillna(0.0)
    if g.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate V1B source team-games")
    return g.sort_values(["season", "week", "team"]).reset_index(drop=True)


def strictly_before(g: pd.DataFrame, season: int, week: int) -> pd.Series:
    return (g["season"] < season) | ((g["season"] == season) & (g["week"] < week))


def safe_rate(numer: float, denom: float, fallback: float) -> float:
    if np.isfinite(denom) and denom > 0:
        return float(numer / denom)
    if np.isfinite(fallback):
        return float(fallback)
    raise RuntimeError("rate denominator zero and strict-prior league fallback unavailable")


def league_rates(g: pd.DataFrame, season: int, week: int) -> dict[str, float]:
    h = g.loc[strictly_before(g, season, week)].copy()
    if h.empty:
        raise RuntimeError(f"no strict-prior league source rows for {season} W{week}")
    return {
        "penalty": safe_rate(h.first_down_penalties.sum(), h.offensive_pbp_rows.sum(), np.nan),
        "go": safe_rate(h.fourth_down_go_attempts.sum(), h.fourth_down_decisions.sum(), np.nan),
        "conv": safe_rate(h.fourth_down_conversions.sum(), h.fourth_down_go_attempts.sum(), np.nan),
        "pass_share": safe_rate(h.fourth_down_go_passes.sum(), h.fourth_down_go_attempts.sum(), np.nan),
    }


def last_n_offense(g: pd.DataFrame, season: int, week: int, team: str) -> pd.DataFrame:
    h = g.loc[strictly_before(g, season, week) & g["team"].eq(team)].copy()
    return h.sort_values(["season", "week"]).tail(LAST_N)


def last_n_defense(g: pd.DataFrame, season: int, week: int, defense: str) -> pd.DataFrame:
    h = g.loc[strictly_before(g, season, week) & g["opponent"].eq(defense)].copy()
    return h.sort_values(["season", "week"]).tail(LAST_N)


def feature_row(g: pd.DataFrame, season: int, week: int, team: str, opponent: str) -> dict:
    lg = league_rates(g, season, week)
    off = last_n_offense(g, season, week, team)
    deff = last_n_defense(g, season, week, opponent)
    if off.empty or deff.empty:
        raise RuntimeError(f"missing last-{LAST_N} source history {season} W{week} {team} vs {opponent}")

    return {
        "team_first_down_penalty_rate_l8": safe_rate(off.first_down_penalties.sum(), off.offensive_pbp_rows.sum(), lg["penalty"]),
        "oppdef_first_down_penalty_allowed_rate_l8": safe_rate(deff.first_down_penalties.sum(), deff.offensive_pbp_rows.sum(), lg["penalty"]),
        "team_fourth_go_rate_l8": safe_rate(off.fourth_down_go_attempts.sum(), off.fourth_down_decisions.sum(), lg["go"]),
        "team_fourth_conversion_rate_l8": safe_rate(off.fourth_down_conversions.sum(), off.fourth_down_go_attempts.sum(), lg["conv"]),
        "team_fourth_pass_share_l8": safe_rate(off.fourth_down_go_passes.sum(), off.fourth_down_go_attempts.sum(), lg["pass_share"]),
        "oppdef_fourth_go_allowed_rate_l8": safe_rate(deff.fourth_down_go_attempts.sum(), deff.fourth_down_decisions.sum(), lg["go"]),
        "oppdef_fourth_conversion_allowed_rate_l8": safe_rate(deff.fourth_down_conversions.sum(), deff.fourth_down_go_attempts.sum(), lg["conv"]),
        "oppdef_fourth_pass_allowed_share_l8": safe_rate(deff.fourth_down_go_passes.sum(), deff.fourth_down_go_attempts.sum(), lg["pass_share"]),
        "off_history_n": int(len(off)),
        "def_history_n": int(len(deff)),
        "off_history_max_season": int(off.season.max()),
        "off_history_max_week": int(off.loc[off.season.eq(off.season.max()), "week"].max()),
        "def_history_max_season": int(deff.season.max()),
        "def_history_max_week": int(deff.loc[deff.season.eq(deff.season.max()), "week"].max()),
    }


def build_2024_features(targets: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in targets.itertuples(index=False):
        f = feature_row(games, int(r.season), int(r.week), str(r.team), str(r.opponent))
        rows.append({
            "season": int(r.season), "week": int(r.week), "team": str(r.team),
            "player_clean_key": str(r.player_clean_key), "opponent": str(r.opponent), **f,
        })
    out = pd.DataFrame(rows)
    if len(out) != 444 or out.duplicated(KEYS).any():
        raise RuntimeError(f"feature table integrity failure rows={len(out)}")
    all_features = PENALTY_FEATURES + FOURTH_FEATURES
    if out[all_features].isna().any().any() or not np.isfinite(out[all_features].to_numpy(float)).all():
        raise RuntimeError("nonfinite D2 feature values")
    return out


def load_wr_2024(shared_root: Path) -> pd.DataFrame:
    s = pd.read_csv(one(shared_root, "qb_wr_shared_pass_volume_secondary_2024_2025.csv"), low_memory=False)
    s.columns = [str(c).strip().lower() for c in s.columns]
    s = canon_keys(s)
    s = s.loc[s.season.eq(2024), KEYS + ["wr_reception_mass_residual"]].copy()
    s["wr_reception_mass_residual"] = num(s["wr_reception_mass_residual"])
    if len(s) != 444 or s.duplicated(KEYS).any() or s.wr_reception_mass_residual.isna().any():
        raise RuntimeError(f"2024 WR shared cohort integrity failure rows={len(s)}")
    return s


def metrics(actual, pred, miss_levels=()):
    a = num(actual).to_numpy(float); p = num(pred).to_numpy(float)
    if len(a) == 0 or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise RuntimeError("invalid metric arrays")
    e = p - a
    out = {
        "n": int(len(a)), "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e)),
        "corr": float(np.corrcoef(a, p)[0, 1]) if len(a) >= 2 else np.nan,
        "p90_abs_error": float(np.quantile(np.abs(e), 0.90)),
    }
    for level in miss_levels:
        out[f"miss_{int(level)}_plus_rate"] = float(np.mean(np.abs(e) >= level))
    return out


def corr_metrics(x, y):
    z = pd.DataFrame({"x": num(x), "y": num(y)}).dropna()
    return {
        "n": int(len(z)),
        "pearson": float(z.x.corr(z.y, method="pearson")),
        "spearman": float(z.x.corr(z.y, method="spearman")),
        "same_sign": float((np.sign(z.x) == np.sign(z.y)).mean()),
    }


def bootstrap_gain(actual, baseline, candidate):
    a = np.asarray(actual, float); b = np.asarray(baseline, float); c = np.asarray(candidate, float)
    gains = np.abs(b-a) - np.abs(c-a)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(gains); vals = np.empty(BOOTSTRAP_N)
    for start in range(0, BOOTSTRAP_N, 1000):
        m = min(1000, BOOTSTRAP_N-start)
        idx = rng.integers(0, n, size=(m,n))
        vals[start:start+m] = gains[idx].mean(axis=1)
    return {
        "observed_mae_gain": float(gains.mean()),
        "p_gain_gt_0": float(np.mean(vals > 0)),
        "p05": float(np.quantile(vals,.05)), "p50": float(np.quantile(vals,.50)),
        "p95": float(np.quantile(vals,.95)), "bootstrap_n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED,
    }


def model_snapshot(scaler: StandardScaler, model: Ridge, features: list[str]) -> dict:
    return {
        "features": features, "alpha": ALPHA, "fit_intercept": False,
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "ridge_coef": [float(v) for v in model.coef_],
        "ridge_intercept": float(model.intercept_),
    }


def fit_score_family(base: pd.DataFrame, features: list[str], name: str) -> tuple[dict, pd.DataFrame, StandardScaler, Ridge]:
    fit = base.loc[base.week.between(1,9)].copy()
    hold = base.loc[base.week.between(10,18)].copy()
    scaler = StandardScaler()
    Xfit = scaler.fit_transform(fit[features].to_numpy(float))
    yfit = (fit.actual_d - fit.pred_d).to_numpy(float)
    model = Ridge(alpha=ALPHA, fit_intercept=False)
    model.fit(Xfit, yfit)
    corr = model.predict(scaler.transform(hold[features].to_numpy(float)))
    hold = hold.copy()
    hold["predicted_d_correction"] = corr
    hold["candidate_d"] = hold.pred_d + corr
    hold["candidate_attempts"] = hold.candidate_d * hold.pred_c * hold.pred_s
    hold["candidate_pass_yards"] = hold.football_synthesis + (hold.candidate_attempts-hold.pred_attempts)*hold.pred_ypa

    d0 = metrics(hold.actual_d, hold.pred_d); d1 = metrics(hold.actual_d, hold.candidate_d)
    a0 = metrics(hold.actual_attempts, hold.pred_attempts, (8,10)); a1 = metrics(hold.actual_attempts, hold.candidate_attempts, (8,10))
    y0 = metrics(hold.actual_pass_yards, hold.football_synthesis, (75,100)); y1 = metrics(hold.actual_pass_yards, hold.candidate_pass_yards, (75,100))
    info_d = corr_metrics(hold.predicted_d_correction, hold.actual_d-hold.pred_d)
    info_wr = corr_metrics(hold.predicted_d_correction, hold.wr_reception_mass_residual)
    boot = bootstrap_gain(hold.actual_pass_yards, hold.football_synthesis, hold.candidate_pass_yards)

    gates = {
        "team_pass_opportunity_mae_gain_ge_0_15": d0["mae"]-d1["mae"] >= .15,
        "team_pass_opportunity_rmse_nonworse": d1["rmse"] <= d0["rmse"] + 1e-12,
        "qb_attempt_mae_gain_ge_0_10": a0["mae"]-a1["mae"] >= .10,
        "qb_pass_yard_mae_gain_ge_0_25": y0["mae"]-y1["mae"] >= .25,
        "qb_pass_yard_rmse_nonworse": y1["rmse"] <= y0["rmse"] + 1e-12,
        "qb_pass_yard_corr_nonworse": y1["corr"] >= y0["corr"] - 1e-12,
        "qb_pass_yard_p90_nonworse": y1["p90_abs_error"] <= y0["p90_abs_error"] + 1e-12,
        "qb_pass_yard_100_plus_nonworse": y1["miss_100_plus_rate"] <= y0["miss_100_plus_rate"] + 1e-12,
        "qb_attempt_10_plus_nonworse": a1["miss_10_plus_rate"] <= a0["miss_10_plus_rate"] + 1e-12,
        "d_correction_vs_actual_d_residual_spearman_ge_0_10": info_d["spearman"] >= .10,
        "d_correction_vs_wr_reception_residual_spearman_ge_0_10": info_wr["spearman"] >= .10,
        "bootstrap_p_gain_ge_0_70": boot["p_gain_gt_0"] >= .70,
    }
    result = {
        "family": name, "features": features,
        "metrics": {
            "team_pass_opportunity": {"baseline": d0, "candidate": d1, "mae_gain": d0["mae"]-d1["mae"]},
            "qb_attempts": {"baseline": a0, "candidate": a1, "mae_gain": a0["mae"]-a1["mae"]},
            "qb_pass_yards": {"baseline": y0, "candidate": y1, "mae_gain": y0["mae"]-y1["mae"]},
        },
        "information_transfer": {"vs_actual_d_residual": info_d, "vs_2024_wr_reception_residual": info_wr},
        "correction": {
            "mean": float(np.mean(corr)), "mean_abs": float(np.mean(np.abs(corr))),
            "p90_abs": float(np.quantile(np.abs(corr),.90)), "min": float(np.min(corr)), "max": float(np.max(corr)),
        },
        "bootstrap": boot, "model": model_snapshot(scaler, model, features), "gates": gates,
        "independent_survivor": bool(all(gates.values())),
    }
    return result, hold, scaler, model


def fit_all_2024(base: pd.DataFrame, features: list[str]) -> dict:
    scaler = StandardScaler(); X = scaler.fit_transform(base[features].to_numpy(float))
    y = (base.actual_d-base.pred_d).to_numpy(float)
    model = Ridge(alpha=ALPHA, fit_intercept=False); model.fit(X,y)
    snap = model_snapshot(scaler,model,features)
    snap.update({
        "training_scope":"ALL_2024_M89_ALIGNED_ROWS", "training_rows":int(len(base)),
        "target":"actual_D_minus_pred_D", "last_n_games":LAST_N,
        "2025_target_outcomes_used":False,
    })
    return snap


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--v1b-root",type=Path,required=True)
    ap.add_argument("--chain-root",type=Path,required=True)
    ap.add_argument("--shared-root",type=Path,required=True)
    ap.add_argument("--m89-root",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    ctx=load_target_context(a.m89_root); chain=load_chain(a.chain_root); games=load_source_games(a.v1b_root)
    feats=build_2024_features(ctx,games); wr=load_wr_2024(a.shared_root)
    base=chain.merge(ctx,on=KEYS,how="inner",validate="one_to_one").merge(feats,on=KEYS+["opponent"],how="inner",validate="one_to_one").merge(wr,on=KEYS,how="inner",validate="one_to_one")
    if len(base)!=444: raise RuntimeError(f"D2 aligned 2024 row drift {len(base)}")
    identity_gap=float((base.pred_d*base.pred_c*base.pred_s-base.pred_attempts).abs().max())

    strictly_prior_ok = bool(((base.off_history_max_season < base.season) | ((base.off_history_max_season == base.season) & (base.off_history_max_week < base.week))).all() and ((base.def_history_max_season < base.season) | ((base.def_history_max_season == base.season) & (base.def_history_max_week < base.week))).all())
    integrity={
        "rows_2024":int(len(base)), "fit_rows_w1_9":int(base.week.between(1,9).sum()), "holdout_rows_w10_18":int(base.week.between(10,18).sum()),
        "sportsbook_result_features_used":False, "target_2025_outcomes_used":False, "last_n_games":LAST_N,
        "penalty_feature_count":len(PENALTY_FEATURES), "fourth_feature_count":len(FOURTH_FEATURES),
        "ridge_alpha":ALPHA, "fit_intercept":False, "production_changed":False,
        "baseline_attempt_identity_max_abs_gap":identity_gap, "strictly_prior_history_verified":strictly_prior_ok,
    }
    integrity_gates={
        "immutable_parent_contracts":True, "m89_opponent_identity_only":True, "zero_sportsbook_result_features":True,
        "zero_2025_target_outcome_use":True, "exact_last8":LAST_N==8, "exact_2_and_6_features":len(PENALTY_FEATURES)==2 and len(FOURTH_FEATURES)==6,
        "ridge_alpha20_no_intercept":ALPHA==20.0, "no_production_change":True,
        "baseline_attempt_identity_reconciles":identity_gap<=TOL, "candidate_only_changes_d":True,
        "fit_holdout_unique_nonempty":base.week.between(1,9).any() and base.week.between(10,18).any() and not base.duplicated(KEYS).any(),
        "all_feature_histories_strictly_prior":strictly_prior_ok,
    }
    if not all(integrity_gates.values()):
        raise RuntimeError(f"D2 integrity gate failure: {integrity_gates}")

    family_results={}; holdouts={}
    for fam,cols in FAMILY_FEATURES.items():
        res,hold,_,_=fit_score_family(base,cols,fam); family_results[fam]=res; holdouts[fam]=hold

    survivors=[f for f,r in family_results.items() if r["independent_survivor"]]
    combined=None; combined_hold=None
    if len(survivors)==2:
        combined_features=PENALTY_FEATURES+FOURTH_FEATURES
        combined,combined_hold,_,_=fit_score_family(base,combined_features,"PENALTY_PLUS_FOURTH_DOWN")

    candidate_pool=[r for r in family_results.values() if r["independent_survivor"]]
    if combined is not None and combined["independent_survivor"]:
        candidate_pool.append(combined)

    selected=None
    if candidate_pool:
        selected=sorted(candidate_pool,key=lambda r:(
            r["metrics"]["qb_pass_yards"]["candidate"]["mae"],
            r["metrics"]["team_pass_opportunity"]["candidate"]["mae"],
            r["metrics"]["qb_attempts"]["candidate"]["mae"],
            -r["information_transfer"]["vs_2024_wr_reception_residual"]["spearman"],
            r["metrics"]["qb_pass_yards"]["candidate"]["rmse"],
        ))[0]["family"]

    disposition="D2_NO_INDEPENDENT_SURVIVOR"
    if selected is not None: disposition=f"D2_{selected}_DEVELOPMENT_SURVIVOR_READY_FOR_2025_CONFIRMATION"

    result={
        "migration":"QB_TEAM_PASS_OPPORTUNITY_PBP_D2", "disposition":disposition,
        "production_actionable":False, "confirmation_authorized":selected is not None,
        "integrity":integrity, "integrity_gates":integrity_gates,
        "independent_survivors":survivors, "selected_confirmation_candidate":selected,
        "family_results":family_results, "combined_result":combined,
    }

    a.out_dir.mkdir(parents=True,exist_ok=True)
    base[KEYS+["opponent"]+PENALTY_FEATURES+FOURTH_FEATURES].to_csv(a.out_dir/"d2_2024_strict_prior_features.csv",index=False)
    for fam,hold in holdouts.items():
        hold.to_csv(a.out_dir/f"d2_{fam.lower()}_holdout_predictions.csv",index=False)
    if combined_hold is not None: combined_hold.to_csv(a.out_dir/"d2_penalty_plus_fourth_down_holdout_predictions.csv",index=False)
    if selected is not None:
        selected_features=(PENALTY_FEATURES if selected=="PENALTY_DRIVE_EXTENSION" else FOURTH_FEATURES if selected=="FOURTH_DOWN_AGGRESSION" else PENALTY_FEATURES+FOURTH_FEATURES)
        frozen=fit_all_2024(base,selected_features); frozen["candidate_name"]=selected
        (a.out_dir/"d2_frozen_2025_confirmation_model.json").write_text(json.dumps(frozen,indent=2,sort_keys=True),encoding="utf-8")
    (a.out_dir/"d2_result.json").write_text(json.dumps(result,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0

if __name__=="__main__": raise SystemExit(main())
