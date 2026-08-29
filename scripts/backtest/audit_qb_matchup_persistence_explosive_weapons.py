#!/usr/bin/env python3
"""Migration 72 — QB matchup persistence + explosive receiving-weapon bridge.

M72 follows M71's negative uncertainty result. It tests two specific, predeclared
matchup hypotheses that were not previously evaluated as a QB passing-yards bridge:

1) defense residual persistence:
   does a defense repeatedly make opposing QBs over/underperform each QB's own
   pregame baseline, especially when prior opposing QBs resemble today's QB?

2) explosive receiving-weapon bridge:
   do the offense's actual receiving weapons (WR/TE/RB collectively), summarized
   from strictly prior targets/explosive catches/YAC/air production, interact with
   a defense's strictly prior explosive/YAC/air vulnerability in a way that
   predicts the canonical QB residual?

Scientific boundary:
- immutable qb_frontier_canonical_v1 is the projection/target source;
- M72 does not rebuild M59-M71 and does not change production logic;
- 2024 trains fixed models; 2025 is untouched M72 evaluation;
- 2023 is history context only;
- no sportsbook/player-prop or game-market field is a feature;
- all PBP features use games strictly before target week;
- no individual-CB matchup is claimed: current historical/live DB matchup data is
  not reliable enough for this migration. Coverage aggregates remain controls only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import scripts.backtest.audit_qb_efficiency_uncertainty as m71

HISTORY_WINDOW = 8
RECENT_WEAPON_WINDOW = 4
MIN_QB_PRIOR = 3
MIN_DEF_PRIOR = 3

MIN_COVERAGE = 0.75
MIN_PASS_RESID_CORR = 0.20
MIN_PASS_MAE_GAIN = 1.00
MIN_PASS_CORR_GAIN = 0.02
MIN_COMPONENT_CORR = 0.15
SUPPORT_PASS_RESID_CORR = 0.10
SUPPORT_PASS_MAE_GAIN = 0.25
SUPPORT_COMPONENT_CORR = 0.05

DEFENSE_PERSISTENCE = [
    "def_qb_resid_mean8", "def_qb_resid_recent3", "def_qb_resid_sd8",
    "def_qb_positive_resid_rate8", "def_prior_qb_games",
]
SIMILAR_QB_PERSISTENCE = DEFENSE_PERSISTENCE + [
    "def_similar_qb_resid8", "def_similarity_weight_sum8",
    "def_similarity_max_weight8", "def_similarity_effective_n8",
]
EXPLOSIVE_WEAPON_BRIDGE = [
    "weapon_exp20_per_target", "weapon_exp40_per_target",
    "weapon_yards_per_target", "weapon_yac_per_reception",
    "weapon_air_per_target", "weapon_top1_exp20_rate",
    "weapon_top2_exp20_rate", "weapon_recent_target_concentration",
    "def_exp20_per_attempt_allowed8", "def_exp40_per_attempt_allowed8",
    "def_yac_per_completion_allowed8", "def_air_per_attempt_allowed8",
    "bridge_exp20", "bridge_exp40", "bridge_yac", "bridge_air",
    "bridge_top_weapon_exp20", "weapon_prior_team_games",
]
EXISTING_DEFENSE_CONTROL = [
    "opp_coverage_man_rate", "opp_coverage_zone_rate",
    "opp_pressure_rate_generated", "opp_def_pass_epa",
    "opp_success_rate_def", "opponent_force_pass",
    "opp_explosive_play_rate_allowed",
]
FAMILIES = {
    "defense_residual_persistence": DEFENSE_PERSISTENCE,
    "similar_qb_defense_persistence": SIMILAR_QB_PERSISTENCE,
    "explosive_weapon_matchup": EXPLOSIVE_WEAPON_BRIDGE,
    "combined_new_matchup": SIMILAR_QB_PERSISTENCE + EXPLOSIVE_WEAPON_BRIDGE,
    "existing_defense_control": EXISTING_DEFENSE_CONTROL,
    "existing_plus_new_matchup": EXISTING_DEFENSE_CONTROL + SIMILAR_QB_PERSISTENCE + EXPLOSIVE_WEAPON_BRIDGE,
}
NEW_FAMILIES = {
    "defense_residual_persistence",
    "similar_qb_defense_persistence",
    "explosive_weapon_matchup",
    "combined_new_matchup",
    "existing_plus_new_matchup",
}


def num(v):
    return pd.to_numeric(v, errors="coerce")


def safe_corr(a, b):
    return m71.safe_corr(a, b)


def safe_div(a, b):
    return m71.safe_div(a, b)


def prior_mask(df, season, week):
    return m71.prior_mask(df, season, week)


def receiver_key(x):
    rid = x.receiver_player_id.astype(str).replace({"nan": "", "None": "", "<NA>": ""})
    name = x.receiver_player_name.astype(str).replace({"nan": "", "None": "", "<NA>": ""})
    return rid.where(rid.ne(""), name)


def build_receiver_games(pbp):
    x = pbp[pbp.official_pass_attempt.eq(1)].copy()
    x["season"] = num(x.season)
    x["week"] = num(x.week)
    x["team"] = x.posteam.map(m71.canon)
    x["opponent"] = x.defteam.map(m71.canon)
    x["receiver_id"] = receiver_key(x)
    x = x[x.team.ne("") & x.opponent.ne("") & x.receiver_id.ne("")].copy()

    x["_complete"] = num(x.complete_pass).fillna(0).eq(1)
    x["_yards"] = num(x.passing_yards).fillna(0.0)
    x["_air"] = num(x.air_yards)
    x["_yac"] = num(x.yards_after_catch)
    x["_exp20"] = (x._complete & x._yards.ge(20)).astype(int)
    x["_exp40"] = (x._complete & x._yards.ge(40)).astype(int)

    rows = []
    keys = ["season", "week", "game_id", "team", "opponent", "receiver_id"]
    for key, g in x.groupby(keys, sort=True, dropna=False):
        season, week, game_id, team, opponent, receiver_id = key
        targets = float(len(g))
        recs = float(g._complete.sum())
        yards = float(g._yards.sum())
        air = float(g._air.fillna(0.0).sum())
        yac = float(np.where(g._complete, g._yac.fillna(0.0), 0.0).sum())
        rows.append({
            "season": int(season), "week": int(week), "game_id": str(game_id),
            "team": m71.canon(team), "opponent": m71.canon(opponent),
            "receiver_id": str(receiver_id), "targets": targets, "receptions": recs,
            "yards": yards, "air_yards": air, "yac": yac,
            "exp20": float(g._exp20.sum()), "exp40": float(g._exp40.sum()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out.sort_values(["season", "week", "game_id", "team"], inplace=True)
        out.reset_index(drop=True, inplace=True)
    return out


def build_defense_explosive_games(pbp):
    x = pbp[pbp.official_pass_attempt.eq(1)].copy()
    x["season"] = num(x.season)
    x["week"] = num(x.week)
    x["offense"] = x.posteam.map(m71.canon)
    x["defense"] = x.defteam.map(m71.canon)
    x = x[x.offense.ne("") & x.defense.ne("")].copy()
    x["_complete"] = num(x.complete_pass).fillna(0).eq(1)
    x["_yards"] = num(x.passing_yards).fillna(0.0)
    x["_air"] = num(x.air_yards)
    x["_yac"] = num(x.yards_after_catch)

    rows = []
    keys = ["season", "week", "game_id", "defense", "offense"]
    for key, g in x.groupby(keys, sort=True, dropna=False):
        season, week, game_id, defense, offense = key
        att = float(len(g))
        comp = float(g._complete.sum())
        rows.append({
            "season": int(season), "week": int(week), "game_id": str(game_id),
            "defense": m71.canon(defense), "offense": m71.canon(offense),
            "exp20_per_attempt_allowed": safe_div(float((g._complete & g._yards.ge(20)).sum()), att),
            "exp40_per_attempt_allowed": safe_div(float((g._complete & g._yards.ge(40)).sum()), att),
            "yac_per_completion_allowed": safe_div(float(np.where(g._complete, g._yac.fillna(0.0), 0.0).sum()), comp),
            "air_per_attempt_allowed": safe_div(float(g._air.fillna(0.0).sum()), att),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out.sort_values(["season", "week", "game_id"], inplace=True)
        out.reset_index(drop=True, inplace=True)
    return out


def qb_traits_from_prior(qb):
    def wmean(col, weight="attempts"):
        if not len(qb) or col not in qb:
            return np.nan
        z = pd.DataFrame({"v": num(qb[col]), "w": num(qb[weight])}).dropna()
        if not len(z) or z.w.sum() <= 0:
            return np.nan
        return float(np.average(z.v, weights=z.w))
    return {
        "ypa": wmean("ypa"),
        "completion_rate": wmean("completion_rate"),
        "yards_per_completion": wmean("yards_per_completion"),
        "air_per_completion": wmean("air_per_completion"),
        "yac_per_completion": wmean("yac_per_completion"),
        "explosive20_yard_share": wmean("explosive20_yard_share"),
        "prior_games": int(len(qb)),
    }


SIM_SCALES = {
    "ypa": 1.5,
    "completion_rate": 0.08,
    "yards_per_completion": 3.0,
    "air_per_completion": 2.5,
    "yac_per_completion": 2.0,
    "explosive20_yard_share": 0.20,
}


def trait_similarity(a, b):
    vals = []
    for c, scale in SIM_SCALES.items():
        av, bv = a.get(c, np.nan), b.get(c, np.nan)
        if np.isfinite(av) and np.isfinite(bv):
            vals.append(((av - bv) / scale) ** 2)
    if len(vals) < 3:
        return np.nan
    return float(np.exp(-0.5 * np.mean(vals)))


def build_historical_qb_residual_games(passer_games):
    rows = []
    chrono = passer_games.sort_values(["season", "week", "game_id"]).copy()
    for _, r in chrono.iterrows():
        prior = chrono[
            chrono.passer_id.astype(str).eq(str(r.passer_id))
            & prior_mask(chrono, int(r.season), int(r.week))
        ].tail(HISTORY_WINDOW)
        traits = qb_traits_from_prior(prior)
        baseline = traits["ypa"]
        rec = r.to_dict()
        for k, v in traits.items():
            rec[f"pregame_{k}"] = v
        rec["qb_ypa_residual_vs_prior8"] = float(r.ypa - baseline) if np.isfinite(baseline) and np.isfinite(r.ypa) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def weapon_features(receiver_games, season, week, team):
    prior_team = receiver_games[
        receiver_games.team.eq(team) & prior_mask(receiver_games, season, week)
    ].copy()
    if not len(prior_team):
        return {"weapon_prior_team_games": 0}

    recent_game_ids = (
        prior_team[["season", "week", "game_id"]]
        .drop_duplicates()
        .sort_values(["season", "week"])
        .tail(RECENT_WEAPON_WINDOW)
        .game_id.astype(str).tolist()
    )
    recent = prior_team[prior_team.game_id.astype(str).isin(recent_game_ids)].copy()
    team_targets = float(recent.targets.sum())
    shares = recent.groupby("receiver_id").targets.sum().sort_values(ascending=False)
    shares = shares / team_targets if team_targets > 0 else shares * np.nan

    player_rows = []
    for rid, share in shares.items():
        hist = receiver_games[
            receiver_games.receiver_id.astype(str).eq(str(rid))
            & prior_mask(receiver_games, season, week)
        ].tail(HISTORY_WINDOW)
        targets = float(hist.targets.sum())
        recs = float(hist.receptions.sum())
        if targets <= 0:
            continue
        player_rows.append({
            "receiver_id": str(rid), "share": float(share),
            "exp20_rate": safe_div(float(hist.exp20.sum()), targets),
            "exp40_rate": safe_div(float(hist.exp40.sum()), targets),
            "ypt": safe_div(float(hist.yards.sum()), targets),
            "yac_per_rec": safe_div(float(hist.yac.sum()), recs),
            "air_per_target": safe_div(float(hist.air_yards.sum()), targets),
        })
    p = pd.DataFrame(player_rows)
    if not len(p):
        return {"weapon_prior_team_games": int(len(recent_game_ids))}

    def weighted(col):
        z = p[["share", col]].dropna()
        if not len(z) or z["share"].sum() <= 0:
            return np.nan
        return float(np.average(z[col], weights=z["share"]))

    p = p.sort_values("share", ascending=False).reset_index(drop=True)
    return {
        "weapon_exp20_per_target": weighted("exp20_rate"),
        "weapon_exp40_per_target": weighted("exp40_rate"),
        "weapon_yards_per_target": weighted("ypt"),
        "weapon_yac_per_reception": weighted("yac_per_rec"),
        "weapon_air_per_target": weighted("air_per_target"),
        "weapon_top1_exp20_rate": float(p.iloc[0].exp20_rate) if len(p) >= 1 else np.nan,
        "weapon_top2_exp20_rate": float(p.iloc[1].exp20_rate) if len(p) >= 2 else np.nan,
        "weapon_recent_target_concentration": float((shares.iloc[:2].sum())) if len(shares) else np.nan,
        "weapon_prior_team_games": int(len(recent_game_ids)),
    }


def defense_explosive_features(def_games, season, week, defense):
    q = def_games[
        def_games.defense.eq(defense) & prior_mask(def_games, season, week)
    ].tail(HISTORY_WINDOW)
    if not len(q):
        return {}
    return {
        "def_exp20_per_attempt_allowed8": float(num(q.exp20_per_attempt_allowed).mean()),
        "def_exp40_per_attempt_allowed8": float(num(q.exp40_per_attempt_allowed).mean()),
        "def_yac_per_completion_allowed8": float(num(q.yac_per_completion_allowed).mean()),
        "def_air_per_attempt_allowed8": float(num(q.air_per_attempt_allowed).mean()),
    }


def defense_residual_features(hist_qb, passer_games, target_pid, season, week, defense):
    prior_target_qb = passer_games[
        passer_games.passer_id.astype(str).eq(str(target_pid))
        & prior_mask(passer_games, season, week)
    ].tail(HISTORY_WINDOW)
    target_traits = qb_traits_from_prior(prior_target_qb)

    q = hist_qb[
        hist_qb.opponent.eq(defense)
        & prior_mask(hist_qb, season, week)
        & num(hist_qb.pregame_prior_games).ge(MIN_QB_PRIOR)
    ].tail(HISTORY_WINDOW).copy()
    q = q[np.isfinite(num(q.qb_ypa_residual_vs_prior8))].copy()
    if len(q) < MIN_DEF_PRIOR:
        return {
            "def_qb_resid_mean8": np.nan,
            "def_qb_resid_recent3": np.nan,
            "def_qb_resid_sd8": np.nan,
            "def_qb_positive_resid_rate8": np.nan,
            "def_prior_qb_games": int(len(q)),
            "def_similar_qb_resid8": np.nan,
            "def_similarity_weight_sum8": np.nan,
            "def_similarity_max_weight8": np.nan,
            "def_similarity_effective_n8": np.nan,
        }

    residuals = num(q.qb_ypa_residual_vs_prior8).to_numpy(dtype=float)
    recent3 = residuals[-3:] if len(residuals) >= 3 else residuals

    weights = []
    for _, g in q.iterrows():
        traits = {
            "ypa": g.get("pregame_ypa", np.nan),
            "completion_rate": g.get("pregame_completion_rate", np.nan),
            "yards_per_completion": g.get("pregame_yards_per_completion", np.nan),
            "air_per_completion": g.get("pregame_air_per_completion", np.nan),
            "yac_per_completion": g.get("pregame_yac_per_completion", np.nan),
            "explosive20_yard_share": g.get("pregame_explosive20_yard_share", np.nan),
        }
        weights.append(trait_similarity(target_traits, traits))
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(w) & np.isfinite(residuals)

    if ok.any() and w[ok].sum() > 0:
        similar = float(np.average(residuals[ok], weights=w[ok]))
        weight_sum = float(w[ok].sum())
        max_weight = float(w[ok].max())
        eff_n = float((w[ok].sum() ** 2) / np.square(w[ok]).sum()) if np.square(w[ok]).sum() > 0 else np.nan
    else:
        similar = weight_sum = max_weight = eff_n = np.nan

    return {
        "def_qb_resid_mean8": float(np.mean(residuals)),
        "def_qb_resid_recent3": float(np.mean(recent3)),
        "def_qb_resid_sd8": float(np.std(residuals, ddof=0)) if len(residuals) >= 2 else np.nan,
        "def_qb_positive_resid_rate8": float(np.mean(residuals > 0)),
        "def_prior_qb_games": int(len(residuals)),
        "def_similar_qb_resid8": similar,
        "def_similarity_weight_sum8": weight_sum,
        "def_similarity_max_weight8": max_weight,
        "def_similarity_effective_n8": eff_n,
    }


def build_features(base, pbp):
    passer_games, _, _ = m71.build_game_tables(pbp)
    ident = m71.match_canonical_passers(base, passer_games)
    if float(ident.identity_match_status.isin(["exact", "within2"]).mean()) < 0.98:
        raise RuntimeError("M72 canonical QB identity coverage below 98%")

    receivers = build_receiver_games(pbp)
    def_games = build_defense_explosive_games(pbp)
    hist_qb = build_historical_qb_residual_games(passer_games)

    out_rows = []
    for i, r in base.reset_index(drop=True).iterrows():
        rec = r.to_dict()
        pid = str(ident.iloc[i].passer_id)
        season, week = int(r.season), int(r.week)
        team, defense = m71.canon(r.team), m71.canon(r.opponent)

        rec["passer_id"] = pid
        rec["identity_match_status"] = ident.iloc[i].identity_match_status
        rec.update(defense_residual_features(hist_qb, passer_games, pid, season, week, defense))
        rec.update(weapon_features(receivers, season, week, team))
        rec.update(defense_explosive_features(def_games, season, week, defense))

        rec["bridge_exp20"] = rec.get("weapon_exp20_per_target", np.nan) * rec.get("def_exp20_per_attempt_allowed8", np.nan)
        rec["bridge_exp40"] = rec.get("weapon_exp40_per_target", np.nan) * rec.get("def_exp40_per_attempt_allowed8", np.nan)
        rec["bridge_yac"] = rec.get("weapon_yac_per_reception", np.nan) * rec.get("def_yac_per_completion_allowed8", np.nan)
        rec["bridge_air"] = rec.get("weapon_air_per_target", np.nan) * rec.get("def_air_per_attempt_allowed8", np.nan)
        rec["bridge_top_weapon_exp20"] = rec.get("weapon_top1_exp20_rate", np.nan) * rec.get("def_exp20_per_attempt_allowed8", np.nan)

        for c in EXISTING_DEFENSE_CONTROL:
            if c not in rec:
                rec[c] = np.nan

        rec["ypa_residual"] = float(r.actual_ypa - r.pred_ypa)
        rec["attempt_residual"] = float(r.actual_attempts - r.pred_attempts)
        rec["pass_residual"] = float(r.actual_pass_yards - r.pred_pass_yards)
        out_rows.append(rec)

    return pd.DataFrame(out_rows), passer_games, receivers, def_games


def feature_coverage(df, features):
    return float(df[features].notna().mean(axis=0).median()) if features else 0.0


def make_model(kind):
    if kind == "ridge":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=50.0)),
        ])
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=150, learning_rate=0.04,
            max_depth=2, min_samples_leaf=15, l2_regularization=5.0,
            random_state=72,
        )),
    ])


def fit_target(train, test, features, target, kind):
    xtr = train[features].copy()
    xte = test[features].copy()
    ytr = num(train[target])
    ok = ytr.notna()
    model = make_model(kind)
    model.fit(xtr.loc[ok], ytr.loc[ok])
    pred = np.asarray(model.predict(xte), dtype=float)

    lo, hi = np.nanquantile(ytr.loc[ok], [0.05, 0.95])
    return np.clip(pred, lo, hi)


def score_candidate(test, pass_corr_pred, ypa_corr_pred, att_corr_pred):
    actual_pass = num(test.actual_pass_yards).to_numpy(dtype=float)
    base_pass = num(test.pred_pass_yards).to_numpy(dtype=float)
    actual_ypa_resid = num(test.ypa_residual).to_numpy(dtype=float)
    actual_att_resid = num(test.attempt_residual).to_numpy(dtype=float)
    actual_pass_resid = num(test.pass_residual).to_numpy(dtype=float)

    corrected = base_pass + pass_corr_pred
    base_err = actual_pass - base_pass
    corr_err = actual_pass - corrected

    base_mae = float(np.nanmean(np.abs(base_err)))
    corrected_mae = float(np.nanmean(np.abs(corr_err)))
    base_corr = safe_corr(base_pass, actual_pass)
    corrected_corr = safe_corr(corrected, actual_pass)
    base_cat100 = int(np.nansum(np.abs(base_err) >= 100.0))
    corrected_cat100 = int(np.nansum(np.abs(corr_err) >= 100.0))

    return {
        "pass_residual_corr": safe_corr(pass_corr_pred, actual_pass_resid),
        "ypa_residual_corr": safe_corr(ypa_corr_pred, actual_ypa_resid),
        "attempt_residual_corr": safe_corr(att_corr_pred, actual_att_resid),
        "base_pass_mae": base_mae,
        "corrected_pass_mae": corrected_mae,
        "pass_mae_gain": base_mae - corrected_mae,
        "base_pass_corr": base_corr,
        "corrected_pass_corr": corrected_corr,
        "pass_corr_gain": corrected_corr - base_corr if np.isfinite(base_corr) and np.isfinite(corrected_corr) else np.nan,
        "base_cat100": base_cat100,
        "corrected_cat100": corrected_cat100,
        "cat100_delta": corrected_cat100 - base_cat100,
    }


def full_gate(row):
    component = max(
        row.get("ypa_residual_corr", np.nan) if np.isfinite(row.get("ypa_residual_corr", np.nan)) else -999,
        row.get("attempt_residual_corr", np.nan) if np.isfinite(row.get("attempt_residual_corr", np.nan)) else -999,
    )
    return bool(
        row["coverage"] >= MIN_COVERAGE
        and row["pass_residual_corr"] >= MIN_PASS_RESID_CORR
        and row["pass_mae_gain"] >= MIN_PASS_MAE_GAIN
        and row["pass_corr_gain"] >= MIN_PASS_CORR_GAIN
        and row["cat100_delta"] <= 0
        and component >= MIN_COMPONENT_CORR
    )


def support_gate(row):
    component = max(
        row.get("ypa_residual_corr", np.nan) if np.isfinite(row.get("ypa_residual_corr", np.nan)) else -999,
        row.get("attempt_residual_corr", np.nan) if np.isfinite(row.get("attempt_residual_corr", np.nan)) else -999,
    )
    return bool(
        row["pass_residual_corr"] >= SUPPORT_PASS_RESID_CORR
        and row["pass_mae_gain"] >= SUPPORT_PASS_MAE_GAIN
        and row["cat100_delta"] <= 0
        and component >= SUPPORT_COMPONENT_CORR
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--history-seasons", default="2023,2024,2025")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    base = m71.lower(pd.read_csv(a.canonical, low_memory=False))
    if len(base) != 643:
        raise RuntimeError(f"M72 canonical invariant expected 643 rows, got {len(base)}")
    base["team"] = base.team.map(m71.canon)
    base["opponent"] = base.opponent.map(m71.canon)

    seasons = [int(v) for v in a.history_seasons.split(",") if v.strip()]
    pbp, manifest = m71.load_pbp(seasons)
    features, passer_games, receiver_games, def_games = build_features(base, pbp)

    train = features[features.season.eq(2024)].copy().reset_index(drop=True)
    test = features[features.season.eq(2025)].copy().reset_index(drop=True)
    if not len(train) or not len(test):
        raise RuntimeError("M72 requires both 2024 train and 2025 evaluation rows")

    rows = []
    prediction_rows = []
    coverage_rows = []
    for family, cols in FAMILIES.items():
        available = [c for c in cols if c in features.columns]
        cov = feature_coverage(test, available)
        coverage_rows.append({
            "family": family, "feature_count": len(available),
            "median_2025_coverage": cov,
        })
        for kind in ["ridge", "hgb"]:
            pass_pred = fit_target(train, test, available, "pass_residual", kind)
            ypa_pred = fit_target(train, test, available, "ypa_residual", kind)
            att_pred = fit_target(train, test, available, "attempt_residual", kind)

            metrics = score_candidate(test, pass_pred, ypa_pred, att_pred)
            row = {
                "family": family, "model": kind, "coverage": cov,
                **metrics,
            }
            row["full_gate"] = full_gate(row)
            row["support_gate"] = support_gate(row)
            rows.append(row)

            pr = test[["season", "week", "game_id", "team", "opponent",
                       "actual_pass_yards", "pred_pass_yards",
                       "actual_ypa", "pred_ypa", "actual_attempts", "pred_attempts"]].copy()
            pr["family"] = family
            pr["model"] = kind
            pr["pred_pass_residual"] = pass_pred
            pr["pred_ypa_residual"] = ypa_pred
            pr["pred_attempt_residual"] = att_pred
            pr["corrected_pass_yards"] = num(pr.pred_pass_yards) + pass_pred
            prediction_rows.append(pr)

    results = pd.DataFrame(rows)

    supported_new = []
    for family in NEW_FAMILIES:
        q = results[results.family.eq(family)]
        if len(q) != 2:
            continue
        for _, winner in q.iterrows():
            other = q[q.model.ne(winner.model)]
            if bool(winner.full_gate) and len(other) and bool(other.iloc[0].support_gate):
                supported_new.append(family)
                break

    existing = results[results.family.eq("existing_defense_control")]
    existing_supported = any(bool(v) for v in existing.full_gate.tolist()) if len(existing) else False

    if supported_new:
        interpretation = "m72_matchup_persistence_or_explosive_weapon_signal_followup"
    elif existing_supported:
        interpretation = "m72_existing_defense_signal_only_no_new_matchup_breakthrough"
    else:
        interpretation = "m72_no_replicated_matchup_bridge_signal_return_to_attempt_opportunity_frontier"

    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    features.to_csv(out / "m72_game_features_and_targets.csv", index=False)
    results.to_csv(out / "m72_model_results.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(out / "m72_feature_coverage.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(out / "m72_2025_predictions.csv", index=False)
    season_label = ",".join(map(str, sorted(seasons)))
    per_season = manifest[manifest.family.eq("pbp_history")].copy()
    all_recovered = (
        len(per_season) == len(set(seasons))
        and per_season.status.astype(str).eq("recovered").all()
    )
    derived_status = "recovered" if all_recovered else "partial_from_recovered_pbp"
    manifest2 = pd.concat([
        manifest,
        pd.DataFrame([
            {"season": season_label, "family": "receiver_level_pbp", "status": derived_status, "rows": int(len(receiver_games))},
            {"season": season_label, "family": "defense_explosive_pbp", "status": derived_status, "rows": int(len(def_games))},
            {"season": season_label, "family": "individual_db_matchup", "status": "not_used_unreliable_live_historical_contract", "rows": 0},
        ])
    ], ignore_index=True)
    manifest2.to_csv(out / "m72_source_manifest.csv", index=False)

    pd.DataFrame([{
        "train_rows_2024": int(len(train)),
        "evaluation_rows_2025": int(len(test)),
        "supported_new_families": "|".join(sorted(set(supported_new))),
        "existing_defense_full_gate": bool(existing_supported),
        "m72_interpretation": interpretation,
        "production_actionable": False,
    }]).to_csv(out / "m72_precommitted_interpretation.csv", index=False)

    print("=== M72 INTERPRETATION ===")
    print(pd.read_csv(out / "m72_precommitted_interpretation.csv").to_string(index=False))
    print("=== M72 MODEL RESULTS ===")
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
