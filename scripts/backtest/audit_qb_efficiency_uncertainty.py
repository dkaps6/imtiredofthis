#!/usr/bin/env python3
"""Migration 71 — QB efficiency uncertainty / tail-risk audit.

M70 found that catastrophic YPA misses were usually real game-level efficiency
shifts that the point model underreacted to, while no single physical mechanism
replicated strongly enough to justify another directional YPA correction.

M71 therefore asks a different, precommitted question: can legitimate pregame
football information predict *when* QB efficiency is unusually uncertain, even
if it cannot predict the direction of the surprise?

Scientific boundary
-------------------
- Immutable `qb_frontier_canonical_v1` is the historical projection source.
- M71 does NOT rebuild M59-M70 and does NOT change production projections.
- 2024 is training/discovery; 2025 is the untouched evaluation season for M71.
- 2023 is history context only and is never target-scored.
- No sportsbook/player-prop or game-market field is a feature.
- All PBP-derived features are built strictly from games before the target week.
- Target-game PBP is used only to resolve the canonical passer identity and to
  score actual outcomes; no target-game statistic becomes a predictor.
- No feature-subset search, hyperparameter sweep, model zoo, or post-hoc gate
  tuning is allowed.
- All M71 outputs are diagnostic; `production_actionable` is always False.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts._opponent_map import canon_team

HISTORY_WINDOW = 8
MIN_PRIOR_QB_GAMES = 3
EXTREME_YPA_ERROR = 1.50
EXTREME_REAL_SHIFT = 1.50

# Frozen M71 gates.
MIN_FEATURE_COVERAGE = 0.75
MIN_RISK_CORR = 0.20
MIN_YPA_Q4_Q1_RATIO = 1.25
MIN_EXTREME_AUC = 0.65
MIN_EXTREME_Q4_CAPTURE = 0.40
MIN_PASS_Q4_Q1_RATIO = 1.15
MIN_CAT100_Q4_CAPTURE = 0.35

# Cross-model support prevents a one-model-only result from earning follow-up.
SUPPORT_RISK_CORR = 0.10
SUPPORT_YPA_Q4_Q1_RATIO = 1.10
SUPPORT_PASS_Q4_Q1_RATIO = 1.05

QB_INTRINSIC = [
    "qb_ypa_mean8",
    "qb_ypa_sd8",
    "qb_ypa_mad8",
    "qb_comp_rate_mean8",
    "qb_comp_rate_sd8",
    "qb_yards_per_completion_mean8",
    "qb_yards_per_completion_sd8",
    "qb_air_per_completion_sd8",
    "qb_yac_per_completion_sd8",
    "qb_cpoe_sd8",
    "qb_explosive20_yard_share_sd8",
    "qb_abs_ypa_change_mean8",
    "qb_ypa_tail15_rate8",
    "qb_recent3_vs8_ypa_abs",
    "qb_attempts_mean8",
    "qb_attempts_sd8",
    "qb_prior_games",
]

OFFENSE_ECOSYSTEM = [
    "off_top1_target_share_mean8",
    "off_top1_target_share_sd8",
    "off_top2_target_share_sd8",
    "off_target_entropy_mean8",
    "off_target_entropy_sd8",
    "off_receiver_count_sd8",
    "off_explosive20_yard_share_sd8",
    "off_yac_per_completion_sd8",
    "off_ypa_sd8",
    "off_recent3_vs8_top1_abs",
    "off_prior_games",
]

OPPONENT_VOLATILITY = [
    "opp_ypa_allowed_mean8",
    "opp_ypa_allowed_sd8",
    "opp_comp_rate_allowed_sd8",
    "opp_yac_per_completion_allowed_sd8",
    "opp_explosive20_yard_share_allowed_mean8",
    "opp_explosive20_yard_share_allowed_sd8",
    "opp_recent3_vs8_ypa_allowed_abs",
    "opp_prior_games",
]

WEEK_SPECIFIC_CONTEXT = [
    "playcaller_changed_since_last_game",
    "playcaller_new_to_team",
    "playcaller_prior_games_allteams",
    "playcaller_prior_games_team",
    "opening_vs_playcaller_first15_abs",
    "opening_vs_playcaller_q1_abs",
    "opp_coverage_man_rate",
    "opp_coverage_zone_rate",
    "opp_pressure_rate_generated",
    "opp_def_pass_epa",
    "opp_success_rate_def",
    "opponent_force_pass",
    "opp_explosive_play_rate_allowed",
]

STRUCTURAL_CONTROL = [
    "model_pass_prediction_sd",
    "model_pass_prediction_range",
    "model_attempt_prediction_sd",
    "model_attempt_prediction_range",
    "model_dbr_prediction_sd",
    "model_dbr_prediction_range",
    "raw_vs_state_pass_abs",
    "raw_vs_gamescript_pass_abs",
]

FAMILIES = {
    "qb_intrinsic_volatility": QB_INTRINSIC,
    "offense_ecosystem_volatility": OFFENSE_ECOSYSTEM,
    "opponent_pass_volatility": OPPONENT_VOLATILITY,
    "week_specific_context": WEEK_SPECIFIC_CONTEXT,
    "combined_new_volatility": QB_INTRINSIC + OFFENSE_ECOSYSTEM + OPPONENT_VOLATILITY + WEEK_SPECIFIC_CONTEXT,
    "structural_uncertainty_control": STRUCTURAL_CONTROL,
    "combined_new_plus_structural_control": QB_INTRINSIC + OFFENSE_ECOSYSTEM + OPPONENT_VOLATILITY + WEEK_SPECIFIC_CONTEXT + STRUCTURAL_CONTROL,
}

NEW_ONLY_FAMILIES = {
    "qb_intrinsic_volatility",
    "offense_ecosystem_volatility",
    "opponent_pass_volatility",
    "week_specific_context",
    "combined_new_volatility",
}


def num(v):
    return pd.to_numeric(v, errors="coerce")


def lower(df):
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def canon(v):
    t = canon_team(v)
    return "WAS" if t == "WSH" else t


def to_pd(o):
    if isinstance(o, pd.DataFrame):
        return o.copy()
    if hasattr(o, "to_pandas"):
        return o.to_pandas()
    return pd.DataFrame(o)


def safe_div(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan


def safe_corr(a, b):
    z = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    if len(z) < 3 or z.a.nunique() < 2 or z.b.nunique() < 2:
        return np.nan
    return float(z.a.corr(z.b))


def regular_only(x):
    q = x.copy()
    c = "season_type" if "season_type" in q.columns else "game_type" if "game_type" in q.columns else None
    if c:
        s = q[c].astype(str).str.upper()
        keep = s.isin(["REG", "REGULAR", "RS", ""])
        if keep.any():
            q = q[keep].copy()
    return q


def ensure(x, cols):
    for c in cols:
        if c not in x.columns:
            x[c] = np.nan
    return x


def load_pbp(seasons):
    import nflreadpy as nfl

    frames = []
    manifest = []
    for season in seasons:
        try:
            q = lower(to_pd(nfl.load_pbp(seasons=[int(season)])))
            q = regular_only(q)
            if len(q):
                frames.append(q)
            manifest.append({
                "season": int(season),
                "family": "pbp_history",
                "status": "recovered" if len(q) else "empty",
                "rows": int(len(q)),
            })
        except Exception as exc:
            manifest.append({
                "season": int(season),
                "family": "pbp_history",
                "status": f"failed:{type(exc).__name__}",
                "rows": 0,
            })
    if not frames:
        raise RuntimeError("M71 requires historical PBP")

    x = pd.concat(frames, ignore_index=True, sort=False)
    x = ensure(x, [
        "season", "week", "game_id", "posteam", "defteam",
        "passer_player_id", "passer_player_name", "receiver_player_id", "receiver_player_name",
        "pass_attempt", "sack", "two_point_attempt", "complete_pass", "passing_yards",
        "air_yards", "yards_after_catch", "cpoe", "interception",
    ])
    raw_pass = num(x.pass_attempt).fillna(0).eq(1)
    sack = num(x.sack).fillna(0).eq(1)
    two_point = num(x.two_point_attempt).fillna(0).eq(1)
    x["official_pass_attempt"] = (raw_pass & ~sack & ~two_point).astype(int)
    manifest.append({
        "season": ",".join(map(str, sorted(set(map(int, seasons))))),
        "family": "official_attempt_normalization",
        "status": "sacks_and_two_point_attempts_excluded",
        "rows": int(x.official_pass_attempt.sum()),
    })
    return x, pd.DataFrame(manifest)


def _receiver_key(g):
    rid = g.receiver_player_id.astype(str).replace({"nan": "", "None": "", "<NA>": ""})
    rname = g.receiver_player_name.astype(str).replace({"nan": "", "None": "", "<NA>": ""})
    return rid.where(rid.ne(""), rname)


def target_distribution_metrics(g):
    keys = _receiver_key(g)
    keys = keys[keys.ne("")]
    if keys.empty:
        return np.nan, np.nan, np.nan, np.nan
    counts = keys.value_counts().astype(float)
    total = float(counts.sum())
    shares = counts / total
    top1 = float(shares.iloc[0])
    top2 = float(shares.iloc[:2].sum())
    n = int(len(shares))
    if n <= 1:
        entropy = 0.0
    else:
        entropy_raw = -float(np.sum(shares * np.log(shares)))
        entropy = safe_div(entropy_raw, math.log(n))
    return top1, top2, entropy, float(n)


def build_game_tables(pbp):
    x = pbp[pbp.official_pass_attempt.eq(1)].copy()
    x["season"] = num(x.season)
    x["week"] = num(x.week)
    x["team"] = x.posteam.map(canon)
    x["opponent"] = x.defteam.map(canon)
    x["passer_id"] = x.passer_player_id.astype(str).replace({"nan": "", "None": "", "<NA>": ""})
    x = x[x.team.ne("") & x.opponent.ne("") & x.passer_id.ne("")].copy()
    if x.empty:
        raise RuntimeError("M71 PBP produced no official passer attempts")

    x["_complete"] = num(x.complete_pass).fillna(0).eq(1)
    x["_pass_yards"] = num(x.passing_yards).fillna(0.0)
    x["_air"] = num(x.air_yards)
    x["_yac"] = num(x.yards_after_catch)
    x["_completed_air"] = np.where(x._complete, x._air.fillna(0.0), 0.0)
    x["_completed_yac"] = np.where(x._complete, x._yac.fillna(0.0), 0.0)
    x["_exp20_yards"] = np.where(x._complete & x._pass_yards.ge(20), x._pass_yards, 0.0)
    x["_cpoe"] = num(x.cpoe)

    passer_rows = []
    passer_keys = ["season", "week", "game_id", "team", "opponent", "passer_id"]
    for key, g in x.groupby(passer_keys, sort=True, dropna=False):
        season, week, game_id, team, opponent, passer_id = key
        attempts = float(len(g))
        completions = float(g._complete.sum())
        yards = float(g._pass_yards.sum())
        air = float(g._completed_air.sum())
        yac = float(g._completed_yac.sum())
        exp20_yards = float(g._exp20_yards.sum())
        passer_rows.append({
            "season": int(season), "week": int(week), "game_id": str(game_id),
            "team": canon(team), "opponent": canon(opponent), "passer_id": str(passer_id),
            "attempts": attempts,
            "ypa": safe_div(yards, attempts),
            "completion_rate": safe_div(completions, attempts),
            "yards_per_completion": safe_div(yards, completions),
            "air_per_completion": safe_div(air, completions),
            "yac_per_completion": safe_div(yac, completions),
            "cpoe": float(g._cpoe.mean()) if g._cpoe.notna().any() else np.nan,
            "explosive20_yard_share": safe_div(exp20_yards, yards) if yards > 0 else np.nan,
        })
    passer_games = pd.DataFrame(passer_rows)

    offense_rows = []
    offense_keys = ["season", "week", "game_id", "team", "opponent"]
    for key, g in x.groupby(offense_keys, sort=True, dropna=False):
        season, week, game_id, team, opponent = key
        attempts = float(len(g))
        completions = float(g._complete.sum())
        yards = float(g._pass_yards.sum())
        yac = float(g._completed_yac.sum())
        exp20_yards = float(g._exp20_yards.sum())
        top1, top2, entropy, receiver_count = target_distribution_metrics(g)
        offense_rows.append({
            "season": int(season), "week": int(week), "game_id": str(game_id),
            "team": canon(team), "opponent": canon(opponent),
            "ypa": safe_div(yards, attempts),
            "yac_per_completion": safe_div(yac, completions),
            "explosive20_yard_share": safe_div(exp20_yards, yards) if yards > 0 else np.nan,
            "top1_target_share": top1,
            "top2_target_share": top2,
            "target_entropy": entropy,
            "receiver_count": receiver_count,
        })
    offense_games = pd.DataFrame(offense_rows)

    defense_rows = []
    defense_keys = ["season", "week", "game_id", "opponent", "team"]
    # Here `opponent` is the defense and `team` is the offense.
    for key, g in x.groupby(defense_keys, sort=True, dropna=False):
        season, week, game_id, defense, offense = key
        attempts = float(len(g))
        completions = float(g._complete.sum())
        yards = float(g._pass_yards.sum())
        yac = float(g._completed_yac.sum())
        exp20_yards = float(g._exp20_yards.sum())
        defense_rows.append({
            "season": int(season), "week": int(week), "game_id": str(game_id),
            "defense": canon(defense), "offense": canon(offense),
            "ypa_allowed": safe_div(yards, attempts),
            "comp_rate_allowed": safe_div(completions, attempts),
            "yac_per_completion_allowed": safe_div(yac, completions),
            "explosive20_yard_share_allowed": safe_div(exp20_yards, yards) if yards > 0 else np.nan,
        })
    defense_games = pd.DataFrame(defense_rows)

    for q in [passer_games, offense_games, defense_games]:
        q.sort_values(["season", "week", "game_id"], inplace=True)
        q.reset_index(drop=True, inplace=True)
    return passer_games, offense_games, defense_games


def prior_mask(df, season, week):
    return (df.season < int(season)) | ((df.season == int(season)) & (df.week < int(week)))


def finite_array(s):
    a = num(s).to_numpy(dtype=float)
    return a[np.isfinite(a)]


def arr_mean(a):
    return float(np.mean(a)) if len(a) else np.nan


def arr_sd(a):
    return float(np.std(a, ddof=0)) if len(a) >= 2 else np.nan


def arr_mad(a):
    if not len(a):
        return np.nan
    med = float(np.median(a))
    return float(np.median(np.abs(a - med)))


def recent3_vs8_abs(a):
    if len(a) < 3:
        return np.nan
    return abs(float(np.mean(a[-3:])) - float(np.mean(a)))


def match_canonical_passers(base, passer_games):
    rows = []
    for _, r in base.iterrows():
        q = passer_games[
            passer_games.season.eq(int(r.season))
            & passer_games.week.eq(int(r.week))
            & passer_games.team.eq(canon(r.team))
        ].copy()
        if "game_id" in r and pd.notna(r.game_id):
            exact = q[q.game_id.astype(str).eq(str(r.game_id))]
            if len(exact):
                q = exact
        if q.empty:
            rows.append({"passer_id": "", "identity_match_status": "missing", "identity_attempt_diff": np.nan})
            continue
        q["_diff"] = (num(q.attempts) - float(r.actual_attempts)).abs()
        hit = q.sort_values(["_diff", "attempts"], ascending=[True, False]).iloc[0]
        diff = float(hit._diff)
        rows.append({
            "passer_id": str(hit.passer_id),
            "identity_match_status": "exact" if diff == 0 else "within2" if diff <= 2 else "mismatch",
            "identity_attempt_diff": diff,
        })
    return pd.DataFrame(rows)


def build_history_features(base, passer_games, offense_games, defense_games):
    ident = match_canonical_passers(base, passer_games)
    match_cov = float(ident.identity_match_status.isin(["exact", "within2"]).mean())
    if match_cov < 0.98:
        raise RuntimeError(f"M71 canonical passer identity coverage too low: {match_cov:.3f}")

    rows = []
    for i, r in base.reset_index(drop=True).iterrows():
        pid = str(ident.iloc[i].passer_id)
        season, week = int(r.season), int(r.week)
        team, opponent = canon(r.team), canon(r.opponent)

        qb = passer_games[
            passer_games.passer_id.astype(str).eq(pid) & prior_mask(passer_games, season, week)
        ].tail(HISTORY_WINDOW)
        off = offense_games[
            offense_games.team.eq(team) & prior_mask(offense_games, season, week)
        ].tail(HISTORY_WINDOW)
        opp = defense_games[
            defense_games.defense.eq(opponent) & prior_mask(defense_games, season, week)
        ].tail(HISTORY_WINDOW)

        qypa = finite_array(qb.ypa) if len(qb) else np.array([])
        qcomp = finite_array(qb.completion_rate) if len(qb) else np.array([])
        qypc = finite_array(qb.yards_per_completion) if len(qb) else np.array([])
        qair = finite_array(qb.air_per_completion) if len(qb) else np.array([])
        qyac = finite_array(qb.yac_per_completion) if len(qb) else np.array([])
        qcpoe = finite_array(qb.cpoe) if len(qb) else np.array([])
        qexp = finite_array(qb.explosive20_yard_share) if len(qb) else np.array([])
        qatt = finite_array(qb.attempts) if len(qb) else np.array([])

        o_top1 = finite_array(off.top1_target_share) if len(off) else np.array([])
        o_top2 = finite_array(off.top2_target_share) if len(off) else np.array([])
        o_entropy = finite_array(off.target_entropy) if len(off) else np.array([])
        o_rc = finite_array(off.receiver_count) if len(off) else np.array([])
        o_exp = finite_array(off.explosive20_yard_share) if len(off) else np.array([])
        o_yac = finite_array(off.yac_per_completion) if len(off) else np.array([])
        o_ypa = finite_array(off.ypa) if len(off) else np.array([])

        d_ypa = finite_array(opp.ypa_allowed) if len(opp) else np.array([])
        d_comp = finite_array(opp.comp_rate_allowed) if len(opp) else np.array([])
        d_yac = finite_array(opp.yac_per_completion_allowed) if len(opp) else np.array([])
        d_exp = finite_array(opp.explosive20_yard_share_allowed) if len(opp) else np.array([])

        rec = {
            "passer_id": pid,
            "identity_match_status": ident.iloc[i].identity_match_status,
            "identity_attempt_diff": ident.iloc[i].identity_attempt_diff,
            "qb_prior_games": int(len(qb)),
            "qb_ypa_mean8": arr_mean(qypa),
            "qb_ypa_sd8": arr_sd(qypa),
            "qb_ypa_mad8": arr_mad(qypa),
            "qb_comp_rate_mean8": arr_mean(qcomp),
            "qb_comp_rate_sd8": arr_sd(qcomp),
            "qb_yards_per_completion_mean8": arr_mean(qypc),
            "qb_yards_per_completion_sd8": arr_sd(qypc),
            "qb_air_per_completion_sd8": arr_sd(qair),
            "qb_yac_per_completion_sd8": arr_sd(qyac),
            "qb_cpoe_sd8": arr_sd(qcpoe),
            "qb_explosive20_yard_share_sd8": arr_sd(qexp),
            "qb_abs_ypa_change_mean8": arr_mean(np.abs(np.diff(qypa))) if len(qypa) >= 2 else np.nan,
            "qb_ypa_tail15_rate8": float(np.mean(np.abs(qypa - np.mean(qypa)) >= EXTREME_REAL_SHIFT)) if len(qypa) >= 3 else np.nan,
            "qb_recent3_vs8_ypa_abs": recent3_vs8_abs(qypa),
            "qb_attempts_mean8": arr_mean(qatt),
            "qb_attempts_sd8": arr_sd(qatt),
            "off_prior_games": int(len(off)),
            "off_top1_target_share_mean8": arr_mean(o_top1),
            "off_top1_target_share_sd8": arr_sd(o_top1),
            "off_top2_target_share_sd8": arr_sd(o_top2),
            "off_target_entropy_mean8": arr_mean(o_entropy),
            "off_target_entropy_sd8": arr_sd(o_entropy),
            "off_receiver_count_sd8": arr_sd(o_rc),
            "off_explosive20_yard_share_sd8": arr_sd(o_exp),
            "off_yac_per_completion_sd8": arr_sd(o_yac),
            "off_ypa_sd8": arr_sd(o_ypa),
            "off_recent3_vs8_top1_abs": recent3_vs8_abs(o_top1),
            "opp_prior_games": int(len(opp)),
            "opp_ypa_allowed_mean8": arr_mean(d_ypa),
            "opp_ypa_allowed_sd8": arr_sd(d_ypa),
            "opp_comp_rate_allowed_sd8": arr_sd(d_comp),
            "opp_yac_per_completion_allowed_sd8": arr_sd(d_yac),
            "opp_explosive20_yard_share_allowed_mean8": arr_mean(d_exp),
            "opp_explosive20_yard_share_allowed_sd8": arr_sd(d_exp),
            "opp_recent3_vs8_ypa_allowed_abs": recent3_vs8_abs(d_ypa),
        }
        rows.append(rec)
    return pd.DataFrame(rows)


def row_sd(r, cols):
    vals = np.array([pd.to_numeric(r.get(c, np.nan), errors="coerce") for c in cols], dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.std(vals, ddof=0)) if len(vals) >= 2 else np.nan


def row_range(r, cols):
    vals = np.array([pd.to_numeric(r.get(c, np.nan), errors="coerce") for c in cols], dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.max(vals) - np.min(vals)) if len(vals) >= 2 else np.nan


def add_canonical_pregame_features(base):
    x = base.copy()
    for c in [
        "playcaller_changed_since_last_game", "playcaller_new_to_team",
        "playcaller_prior_games_allteams", "playcaller_prior_games_team",
        "opening_first15_dbr_mean8", "opening_q1_dbr_mean8",
        "playcaller_opening_first15_dbr_mean8", "playcaller_opening_q1_dbr_mean8",
        "opp_coverage_man_rate", "opp_coverage_zone_rate", "opp_pressure_rate_generated",
        "opp_def_pass_epa", "opp_success_rate_def", "opponent_force_pass",
        "opp_explosive_play_rate_allowed",
    ]:
        if c not in x.columns:
            x[c] = np.nan
        x[c] = num(x[c])

    x["opening_vs_playcaller_first15_abs"] = (
        num(x.opening_first15_dbr_mean8) - num(x.playcaller_opening_first15_dbr_mean8)
    ).abs()
    x["opening_vs_playcaller_q1_abs"] = (
        num(x.opening_q1_dbr_mean8) - num(x.playcaller_opening_q1_dbr_mean8)
    ).abs()

    pass_cols = [
        "pred_pass_yards", "m64_pass_raw_reference", "m64_pass_generative_neutral",
        "m64_pass_generative_gamescript", "m65_pass_state_ridge",
    ]
    att_cols = [
        "pred_attempts", "attempts_raw", "m64_attempts_generative_neutral",
        "m64_attempts_generative_gamescript", "m65_attempts_state_ridge",
    ]
    dbr_cols = [
        "m64_pred_dropback_rate_neutral", "m64_pred_dropback_rate_gamescript", "m65_pred_dropback_rate",
    ]
    for c in pass_cols + att_cols + dbr_cols:
        if c not in x.columns:
            x[c] = np.nan
        x[c] = num(x[c])

    x["model_pass_prediction_sd"] = [row_sd(r, pass_cols) for _, r in x.iterrows()]
    x["model_pass_prediction_range"] = [row_range(r, pass_cols) for _, r in x.iterrows()]
    x["model_attempt_prediction_sd"] = [row_sd(r, att_cols) for _, r in x.iterrows()]
    x["model_attempt_prediction_range"] = [row_range(r, att_cols) for _, r in x.iterrows()]
    x["model_dbr_prediction_sd"] = [row_sd(r, dbr_cols) for _, r in x.iterrows()]
    x["model_dbr_prediction_range"] = [row_range(r, dbr_cols) for _, r in x.iterrows()]
    x["raw_vs_state_pass_abs"] = (num(x.m64_pass_raw_reference) - num(x.m65_pass_state_ridge)).abs()
    x["raw_vs_gamescript_pass_abs"] = (num(x.m64_pass_raw_reference) - num(x.m64_pass_generative_gamescript)).abs()
    return x


def feature_coverage(df, features):
    if not features:
        return 0.0
    return float(df[features].notna().mean(axis=0).median())


def quartile_metrics(test, risk, family, model_name):
    z = test.copy().reset_index(drop=True)
    z["risk_score"] = np.clip(num(pd.Series(risk, index=z.index)), 0, None)
    z = z[np.isfinite(z.risk_score) & np.isfinite(z.ypa_abs_error)].copy()
    if len(z) < 20:
        return None, pd.DataFrame()

    ranked = z.risk_score.rank(method="first")
    z["risk_quartile"] = pd.qcut(ranked, 4, labels=[1, 2, 3, 4]).astype(int)
    qrows = []
    for q, g in z.groupby("risk_quartile"):
        qrows.append({
            "family": family,
            "model": model_name,
            "risk_quartile": int(q),
            "n": int(len(g)),
            "mean_risk_score": float(g.risk_score.mean()),
            "mean_ypa_abs_error": float(g.ypa_abs_error.mean()),
            "passing_yards_mae": float(g.abs_pass_error.mean()),
            "extreme_ypa_error_rate": float(g.extreme_ypa_error.mean()),
            "extreme_real_shift_rate": float(g.extreme_real_shift.mean()),
            "cat100_rate": float(g.cat100_bool.mean()),
        })
    qdf = pd.DataFrame(qrows).sort_values("risk_quartile")
    means = qdf.mean_ypa_abs_error.to_numpy(dtype=float)
    monotonic = bool(len(means) == 4 and np.all(np.diff(means) >= 0))
    q1 = qdf[qdf.risk_quartile.eq(1)].iloc[0]
    q4 = qdf[qdf.risk_quartile.eq(4)].iloc[0]

    try:
        auc_extreme = float(roc_auc_score(z.extreme_ypa_error.astype(int), z.risk_score)) if z.extreme_ypa_error.nunique() > 1 else np.nan
    except Exception:
        auc_extreme = np.nan
    try:
        auc_real_shift = float(roc_auc_score(z.extreme_real_shift.astype(int), z.risk_score)) if z.extreme_real_shift.nunique() > 1 else np.nan
    except Exception:
        auc_real_shift = np.nan

    extreme_n = int(z.extreme_ypa_error.sum())
    extreme_q4 = int(z[z.risk_quartile.eq(4)].extreme_ypa_error.sum())
    cat_n = int(z.cat100_bool.sum())
    cat_q4 = int(z[z.risk_quartile.eq(4)].cat100_bool.sum())

    metrics = {
        "n_test": int(len(z)),
        "risk_corr": safe_corr(z.risk_score, z.ypa_abs_error),
        "ypa_q4_q1_ratio": safe_div(float(q4.mean_ypa_abs_error), float(q1.mean_ypa_abs_error)),
        "quartile_monotonic": monotonic,
        "auc_extreme_ypa_error": auc_extreme,
        "auc_extreme_real_shift": auc_real_shift,
        "extreme_ypa_q4_capture": safe_div(extreme_q4, extreme_n),
        "pass_q4_q1_mae_ratio": safe_div(float(q4.passing_yards_mae), float(q1.passing_yards_mae)),
        "cat100_q4_capture": safe_div(cat_q4, cat_n),
        "extreme_ypa_events": extreme_n,
        "cat100_events": cat_n,
    }
    return metrics, qdf


def fit_risk_model(train, test, features, model_name):
    train = train.copy()
    test = test.copy()
    usable = [c for c in features if c in train.columns and train[c].notna().any()]
    if not usable:
        raise RuntimeError(f"M71 family {model_name} has no train-available features")

    Xtr = train[usable].apply(num)
    Xte = test[usable].apply(num)
    ytr = num(train.ypa_abs_error)

    if model_name == "ridge50":
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=50.0)),
        ])
    elif model_name == "hgb_fixed":
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                loss="absolute_error",
                max_iter=150,
                learning_rate=0.04,
                max_depth=2,
                min_samples_leaf=15,
                l2_regularization=5.0,
                random_state=71,
            )),
        ])
    else:
        raise ValueError(model_name)
    model.fit(Xtr, ytr)
    return np.clip(model.predict(Xte), 0, None), usable


def full_gate(m, coverage):
    vals = [
        coverage >= MIN_FEATURE_COVERAGE,
        np.isfinite(m["risk_corr"]) and m["risk_corr"] >= MIN_RISK_CORR,
        np.isfinite(m["ypa_q4_q1_ratio"]) and m["ypa_q4_q1_ratio"] >= MIN_YPA_Q4_Q1_RATIO,
        bool(m["quartile_monotonic"]),
        np.isfinite(m["auc_extreme_ypa_error"]) and m["auc_extreme_ypa_error"] >= MIN_EXTREME_AUC,
        np.isfinite(m["extreme_ypa_q4_capture"]) and m["extreme_ypa_q4_capture"] >= MIN_EXTREME_Q4_CAPTURE,
        np.isfinite(m["pass_q4_q1_mae_ratio"]) and m["pass_q4_q1_mae_ratio"] >= MIN_PASS_Q4_Q1_RATIO,
        np.isfinite(m["cat100_q4_capture"]) and m["cat100_q4_capture"] >= MIN_CAT100_Q4_CAPTURE,
    ]
    return bool(all(vals))


def support_gate(m, coverage):
    return bool(
        coverage >= MIN_FEATURE_COVERAGE
        and np.isfinite(m["risk_corr"]) and m["risk_corr"] >= SUPPORT_RISK_CORR
        and np.isfinite(m["ypa_q4_q1_ratio"]) and m["ypa_q4_q1_ratio"] >= SUPPORT_YPA_Q4_Q1_RATIO
        and np.isfinite(m["pass_q4_q1_mae_ratio"]) and m["pass_q4_q1_mae_ratio"] >= SUPPORT_PASS_Q4_Q1_RATIO
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--history-seasons", default="2023,2024,2025")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    seasons = [int(v) for v in a.history_seasons.split(",") if v.strip()]
    base = lower(pd.read_csv(a.canonical, low_memory=False))
    if len(base) != 643 or set(num(base.season).dropna().astype(int).unique()) != {2024, 2025}:
        raise RuntimeError("M71 canonical v1 invariant failed")

    # Explicit football-only feature boundary: no market fields may enter any family.
    all_features = sorted(set(sum(FAMILIES.values(), [])))
    prohibited = [c for c in all_features if c.startswith("market_") or c.startswith("actual_") or "sportsbook" in c or "prop" in c]
    if prohibited:
        raise RuntimeError(f"M71 prohibited feature(s): {prohibited}")

    base["team"] = base.team.map(canon)
    base["opponent"] = base.opponent.map(canon)
    for c in ["actual_ypa", "pred_ypa", "actual_pass_yards", "pred_pass_yards", "actual_attempts", "abs_pass_error"]:
        base[c] = num(base[c])
    base["cat100_bool"] = base.cat100.astype(str).str.lower().isin(["true", "1", "yes"])

    pbp, manifest = load_pbp(seasons)
    passer_games, offense_games, defense_games = build_game_tables(pbp)
    hist = build_history_features(base, passer_games, offense_games, defense_games)
    x = pd.concat([base.reset_index(drop=True), hist.reset_index(drop=True)], axis=1)
    x = add_canonical_pregame_features(x)

    x["ypa_abs_error"] = (num(x.actual_ypa) - num(x.pred_ypa)).abs()
    x["extreme_ypa_error"] = x.ypa_abs_error.ge(EXTREME_YPA_ERROR)
    x["actual_abs_shift_vs_qb8"] = (num(x.actual_ypa) - num(x.qb_ypa_mean8)).abs()
    x["extreme_real_shift"] = x.actual_abs_shift_vs_qb8.ge(EXTREME_REAL_SHIFT)
    x["eligible_qb_history"] = num(x.qb_prior_games).ge(MIN_PRIOR_QB_GAMES) & num(x.qb_ypa_mean8).notna()

    eligible = x[x.eligible_qb_history].copy()
    train = eligible[num(eligible.season).eq(2024)].copy()
    test = eligible[num(eligible.season).eq(2025)].copy()
    if len(train) < 150 or len(test) < 150:
        raise RuntimeError(f"M71 insufficient train/test rows: {len(train)}/{len(test)}")

    results = []
    quartiles = []
    coverage_rows = []

    # Frozen simple comparator: prior-8 QB YPA standard deviation as risk score.
    baseline_features = ["qb_ypa_sd8"]
    baseline_cov = feature_coverage(test, baseline_features)
    bm, bq = quartile_metrics(test, num(test.qb_ypa_sd8), "qb_intrinsic_volatility", "qb_ypa_sd8_baseline")
    if bm is not None:
        bm.update({
            "family": "qb_intrinsic_volatility",
            "model": "qb_ypa_sd8_baseline",
            "feature_coverage": baseline_cov,
            "used_feature_count": 1,
            "full_gate": full_gate(bm, baseline_cov),
            "support_gate": support_gate(bm, baseline_cov),
        })
        results.append(bm)
        quartiles.append(bq)

    for family, features in FAMILIES.items():
        coverage = feature_coverage(test, features)
        coverage_rows.append({
            "family": family,
            "feature_count": len(features),
            "median_2025_feature_coverage": coverage,
            "min_2025_feature_coverage": float(test[features].notna().mean(axis=0).min()),
        })
        for model_name in ["ridge50", "hgb_fixed"]:
            risk, used = fit_risk_model(train, test, features, model_name)
            m, q = quartile_metrics(test, risk, family, model_name)
            if m is None:
                continue
            m.update({
                "family": family,
                "model": model_name,
                "feature_coverage": coverage,
                "used_feature_count": len(used),
                "full_gate": full_gate(m, coverage),
                "support_gate": support_gate(m, coverage),
            })
            results.append(m)
            quartiles.append(q)

    result_df = pd.DataFrame(results)
    if result_df.empty:
        raise RuntimeError("M71 produced no model results")

    family_rows = []
    for family in FAMILIES:
        g = result_df[(result_df.family.eq(family)) & (result_df.model.isin(["ridge50", "hgb_fixed"]))]
        strong = g[g.full_gate.astype(bool)]
        supported = False
        if len(strong):
            for _, srow in strong.iterrows():
                other = g[g.model.ne(srow.model)]
                if len(other) and bool(other.iloc[0].support_gate):
                    supported = True
                    break
        family_rows.append({
            "family": family,
            "full_gate_model_count": int(g.full_gate.astype(bool).sum()) if len(g) else 0,
            "cross_model_supported_followup": bool(supported),
        })
    family_df = pd.DataFrame(family_rows)

    baseline_pass = False
    b = result_df[result_df.model.eq("qb_ypa_sd8_baseline")]
    if len(b):
        baseline_pass = bool(b.iloc[0].full_gate)

    new_supported = family_df[
        family_df.family.isin(NEW_ONLY_FAMILIES) & family_df.cross_model_supported_followup.astype(bool)
    ].family.tolist()
    structural_supported = family_df[
        family_df.family.isin(["structural_uncertainty_control", "combined_new_plus_structural_control"])
        & family_df.cross_model_supported_followup.astype(bool)
    ].family.tolist()

    if new_supported or baseline_pass:
        interpretation = "m71_efficiency_uncertainty_predictable_followup"
    elif structural_supported:
        interpretation = "m71_existing_model_uncertainty_signal_only"
    else:
        interpretation = "m71_efficiency_uncertainty_not_predictable_with_current_pregame_information"

    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    x.to_csv(out / "m71_game_features_and_targets.csv", index=False)
    result_df.to_csv(out / "m71_model_results.csv", index=False)
    family_df.to_csv(out / "m71_family_gate_summary.csv", index=False)
    pd.concat(quartiles, ignore_index=True).to_csv(out / "m71_risk_quartiles.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(out / "m71_feature_coverage.csv", index=False)
    manifest.to_csv(out / "m71_source_manifest.csv", index=False)
    pd.DataFrame([{
        "canonical_rows": int(len(base)),
        "eligible_rows": int(len(eligible)),
        "train_2024_rows": int(len(train)),
        "test_2025_rows": int(len(test)),
        "identity_exact_or_within2_share": float(hist.identity_match_status.isin(["exact", "within2"]).mean()),
        "simple_qb_ypa_sd8_full_gate": baseline_pass,
        "new_supported_families": "|".join(new_supported),
        "structural_supported_families": "|".join(structural_supported),
        "m71_interpretation": interpretation,
        "production_actionable": False,
    }]).to_csv(out / "m71_precommitted_interpretation.csv", index=False)

    print("=== M71 PRECOMMITTED INTERPRETATION ===")
    print(pd.read_csv(out / "m71_precommitted_interpretation.csv").to_string(index=False))
    print("=== M71 MODEL RESULTS ===")
    cols = [
        "family", "model", "n_test", "feature_coverage", "risk_corr", "ypa_q4_q1_ratio",
        "quartile_monotonic", "auc_extreme_ypa_error", "extreme_ypa_q4_capture",
        "pass_q4_q1_mae_ratio", "cat100_q4_capture", "full_gate", "support_gate",
    ]
    print(result_df[cols].to_string(index=False))
    print("=== M71 FAMILY GATES ===")
    print(family_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
