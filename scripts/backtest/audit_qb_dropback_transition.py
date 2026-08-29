#!/usr/bin/env python3
"""Migration 74 — QB dropback-rate transition / inducement audit.

M73 established two facts on the frozen 643-game QB frontier:
- perfect attempts retain ~11.16 passing-MAE yards of oracle headroom; and
- dropback rate is the dominant canonical-anchored opportunity mechanism in
  60/116 (51.7%) 10+ attempt misses, replicated in both 2024 and 2025.

M74 is deliberately narrower than M65/M67/M68. It does not rebuild state
occupancy or fit another generic additive QB residual. It predicts the
*multiplicative full-game dropback-rate shift* away from M73's strictly-prior
pregame DBR expectation, then maps that shift back onto the frozen canonical Raw
attempt and passing-yard points.

New information tested:
1) team DBR-transition persistence (prior actual/expected DBR ratios),
2) opponent-adjusted defensive DBR inducement (how much a defense made previous
   offenses deviate from their own pregame DBR expectations), and
3) already-verified opening-script / playcaller pregame tendencies from M68,
   residualized against the current DBR expectation.

Scientific boundary:
- immutable qb_frontier_canonical_v1 is the point-projection source
- 2024 trains fixed models; 2025 is untouched M74 evaluation
- 2022/2023 may provide strictly-prior history only
- no sportsbook player-prop OR game-market field is a feature
- no feature-subset search, hyperparameter sweep, threshold retuning, or model zoo
- M74 is diagnostic; production_actionable is always False
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtest import audit_qb_efficiency_uncertainty as m71
from scripts.backtest import audit_qb_opportunity_recoverability as m73

HISTORY_WINDOW = 8
RECENT_WINDOW = 3
DBR_MIN = 0.20
DBR_MAX = 0.90
HIGH_DBR_SHIFT = 0.10
LOW_DBR_SHIFT = -0.10

# Frozen M74 gates.
MIN_COVERAGE = 0.75
MIN_TRANSITION_CORR = 0.20
MIN_DBR_MAE_GAIN = 0.0075
MIN_DBR_CORR_GAIN = 0.10
MIN_ATTEMPT_MAE_GAIN = 0.30
MIN_ATTEMPT_CORR_GAIN = 0.03
MIN_ATTEMPT_TAIL_REDUCTION = 0.05
MIN_PASS_MAE_GAIN = 1.00
MIN_PASS_CORR_GAIN = 0.02

# Cross-model support. A one-model-only result does not earn follow-up.
SUPPORT_TRANSITION_CORR = 0.10
SUPPORT_DBR_MAE_GAIN = 0.0025
SUPPORT_ATTEMPT_MAE_GAIN = 0.10
SUPPORT_PASS_MAE_GAIN = 0.25

TEAM_TRANSITION = [
    "team_dbr_logratio_last1",
    "team_dbr_logratio_mean3",
    "team_dbr_logratio_mean8",
    "team_dbr_logratio_sd8",
    "team_dbr_logratio_positive_rate8",
    "team_dbr_logratio_mean3_minus8",
]

DEFENSE_INDUCEMENT = [
    "def_dbr_logratio_last1",
    "def_dbr_logratio_mean3",
    "def_dbr_logratio_mean8",
    "def_dbr_logratio_sd8",
    "def_dbr_logratio_positive_rate8",
    "def_dbr_logratio_mean3_minus8",
]

OPENING_PLAYCALLER = [
    "opening_first15_dbr_mean8",
    "opening_q1_dbr_mean8",
    "playcaller_opening_first15_dbr_mean8",
    "playcaller_opening_q1_dbr_mean8",
    "playcaller_changed_since_last_game",
    "playcaller_prior_games_allteams",
    "playcaller_prior_games_team",
    "playcaller_new_to_team",
    "opening_first15_vs_current_dbr",
    "opening_q1_vs_current_dbr",
    "playcaller_first15_vs_current_dbr",
    "playcaller_q1_vs_current_dbr",
]

STATE_CONTROL = [
    "m64_pred_dropback_rate_neutral",
    "m64_pred_dropback_rate_gamescript",
    "m65_pred_neutral_share",
    "m65_pred_trailing_share",
    "m65_pred_leading_share",
    "m65_pred_neutral_dropback_rate",
    "m65_pred_trailing_dropback_rate",
    "m65_pred_leading_dropback_rate",
    "m65_pred_dropback_rate",
    "m65_state_rate_spread",
    "m65_state_occupancy_entropy",
]

COMBINED_NEW = TEAM_TRANSITION + DEFENSE_INDUCEMENT + OPENING_PLAYCALLER
FAMILIES = {
    "team_dbr_transition": TEAM_TRANSITION,
    "defense_dbr_inducement": DEFENSE_INDUCEMENT,
    "opening_playcaller_intent": OPENING_PLAYCALLER,
    "combined_new_transition": COMBINED_NEW,
    "m65_state_control": STATE_CONTROL,
    "state_plus_new_transition": STATE_CONTROL + COMBINED_NEW,
}
NEW_STANDALONE_FAMILIES = {
    "team_dbr_transition",
    "defense_dbr_inducement",
    "opening_playcaller_intent",
    "combined_new_transition",
}


def num(x):
    return pd.to_numeric(x, errors="coerce")


def safe_corr(a, b):
    z = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    if len(z) < 3 or z.a.nunique() < 2 or z.b.nunique() < 2:
        return np.nan
    return float(z.a.corr(z.b))


def mae(a, b):
    z = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    return float((z.a - z.b).abs().mean()) if len(z) else np.nan


def rmse(a, b):
    z = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    return float(np.sqrt(np.mean(np.square(z.a - z.b)))) if len(z) else np.nan


def safe_auc(y, score):
    z = pd.DataFrame({"y": num(y), "s": num(score)}).dropna()
    if len(z) < 20 or z.y.nunique() != 2:
        return np.nan
    return float(roc_auc_score(z.y.astype(int), z.s))


def prior_mask(df, season, week):
    return (num(df.season) < int(season)) | ((num(df.season) == int(season)) & (num(df.week) < int(week)))


def finite_values(s):
    a = num(s).to_numpy(dtype=float)
    return a[np.isfinite(a)]


def summarize_prior(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {
            "last1": np.nan, "mean3": np.nan, "mean8": np.nan,
            "sd8": np.nan, "positive_rate8": np.nan, "mean3_minus8": np.nan,
        }
    a = a[-HISTORY_WINDOW:]
    mean8 = float(np.mean(a))
    mean3 = float(np.mean(a[-RECENT_WINDOW:])) if len(a) >= 1 else np.nan
    return {
        "last1": float(a[-1]),
        "mean3": mean3,
        "mean8": mean8,
        "sd8": float(np.std(a, ddof=0)) if len(a) >= 2 else np.nan,
        "positive_rate8": float(np.mean(a > 0)),
        "mean3_minus8": mean3 - mean8 if np.isfinite(mean3) else np.nan,
    }


def historical_dbr_residual_table(team_games):
    rows = []
    ordered = team_games.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
    for r in ordered.itertuples(index=False):
        season, week = int(r.season), int(r.week)
        team, defense = m71.canon(r.team), m71.canon(r.defense)
        expected = m73.component_expectation(
            team_games, season, week, team, defense, "dropback_rate"
        )
        actual = float(r.dropback_rate) if np.isfinite(r.dropback_rate) else np.nan
        if np.isfinite(actual) and np.isfinite(expected) and actual > 0 and expected > 0:
            logratio = float(np.log(actual / expected))
        else:
            logratio = np.nan
        rows.append({
            "season": season,
            "week": week,
            "game_id": str(r.game_id),
            "team": team,
            "defense": defense,
            "actual_dropback_rate": actual,
            "expected_dropback_rate": expected,
            "dbr_logratio": logratio,
        })
    return pd.DataFrame(rows)


def add_safe_canonical_columns(rec, r):
    # Exact allow-list. Market fields and all target-game outcome/forensic fields
    # in the canonical snapshot are intentionally unreachable here.
    for c in [
        "opening_first15_dbr_mean8", "opening_q1_dbr_mean8",
        "playcaller_opening_first15_dbr_mean8", "playcaller_opening_q1_dbr_mean8",
        "playcaller_changed_since_last_game", "playcaller_prior_games_allteams",
        "playcaller_prior_games_team", "playcaller_new_to_team",
        "m64_pred_dropback_rate_neutral", "m64_pred_dropback_rate_gamescript",
        "m65_pred_neutral_share", "m65_pred_trailing_share", "m65_pred_leading_share",
        "m65_pred_neutral_dropback_rate", "m65_pred_trailing_dropback_rate",
        "m65_pred_leading_dropback_rate", "m65_pred_dropback_rate",
    ]:
        rec[c] = float(r[c]) if c in r.index and pd.notna(r[c]) else np.nan


def build_features(base, pbp):
    team_games = m73.build_team_game_components(pbp)
    residuals = historical_dbr_residual_table(team_games)
    rows = []

    for _, r in base.iterrows():
        season, week = int(r.season), int(r.week)
        team, defense = m71.canon(r.team), m71.canon(r.opponent)
        actual = m73.get_target_team_game(team_games, r)
        if actual is None:
            continue

        baseline = m73.component_expectation(
            team_games, season, week, team, defense, "dropback_rate"
        )
        actual_dbr = float(actual.dropback_rate) if np.isfinite(actual.dropback_rate) else np.nan
        if not (np.isfinite(baseline) and baseline > 0 and np.isfinite(actual_dbr) and actual_dbr > 0):
            continue

        rec = r.to_dict()
        rec["team"] = team
        rec["opponent"] = defense
        rec["baseline_dropback_rate"] = baseline
        rec["actual_dropback_rate_m74"] = actual_dbr
        rec["dbr_logratio_target"] = float(np.log(actual_dbr / baseline))
        rec["dbr_residual_target"] = actual_dbr - baseline
        add_safe_canonical_columns(rec, r)

        hist = residuals[prior_mask(residuals, season, week)]
        team_hist = hist[hist.team.eq(team)].tail(HISTORY_WINDOW)
        def_hist = hist[hist.defense.eq(defense)].tail(HISTORY_WINDOW)
        ts = summarize_prior(finite_values(team_hist.dbr_logratio))
        ds = summarize_prior(finite_values(def_hist.dbr_logratio))
        for k, v in ts.items():
            rec[f"team_dbr_logratio_{k}"] = v
        for k, v in ds.items():
            rec[f"def_dbr_logratio_{k}"] = v

        rec["opening_first15_vs_current_dbr"] = (
            rec["opening_first15_dbr_mean8"] - baseline
            if np.isfinite(rec["opening_first15_dbr_mean8"])
            else np.nan
        )
        rec["opening_q1_vs_current_dbr"] = (
            rec["opening_q1_dbr_mean8"] - baseline
            if np.isfinite(rec["opening_q1_dbr_mean8"])
            else np.nan
        )
        rec["playcaller_first15_vs_current_dbr"] = (
            rec["playcaller_opening_first15_dbr_mean8"] - baseline
            if np.isfinite(rec["playcaller_opening_first15_dbr_mean8"])
            else np.nan
        )
        rec["playcaller_q1_vs_current_dbr"] = (
            rec["playcaller_opening_q1_dbr_mean8"] - baseline
            if np.isfinite(rec["playcaller_opening_q1_dbr_mean8"])
            else np.nan
        )

        state_rates = [
            rec.get("m65_pred_neutral_dropback_rate", np.nan),
            rec.get("m65_pred_trailing_dropback_rate", np.nan),
            rec.get("m65_pred_leading_dropback_rate", np.nan),
        ]
        state_rates = [v for v in state_rates if np.isfinite(v)]
        rec["m65_state_rate_spread"] = (
            float(max(state_rates) - min(state_rates)) if len(state_rates) >= 2 else np.nan
        )
        shares = np.asarray([
            rec.get("m65_pred_neutral_share", np.nan),
            rec.get("m65_pred_trailing_share", np.nan),
            rec.get("m65_pred_leading_share", np.nan),
        ], dtype=float)
        shares = shares[np.isfinite(shares) & (shares > 0)]
        if len(shares):
            shares = shares / shares.sum()
            rec["m65_state_occupancy_entropy"] = float(-np.sum(shares * np.log(shares)))
        else:
            rec["m65_state_occupancy_entropy"] = np.nan
        rows.append(rec)

    out = pd.DataFrame(rows)
    if len(out) < 0.98 * len(base):
        raise RuntimeError(f"M74 feature coverage population too low: {len(out)}/{len(base)}")
    return out, residuals, team_games


def feature_coverage(df, cols):
    if not cols:
        return 0.0
    return float(df[cols].notna().mean(axis=0).median())


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
            random_state=74,
        )),
    ])


def fit_transition(train, test, cols, kind):
    model = make_model(kind)
    y = num(train.dbr_logratio_target)
    ok = y.notna()
    model.fit(train.loc[ok, cols], y.loc[ok])
    pred = np.asarray(model.predict(test[cols]), dtype=float)
    lo, hi = np.nanquantile(y.loc[ok], [0.05, 0.95])
    return np.clip(pred, lo, hi)


def evaluate(test, pred_logratio):
    actual_dbr = num(test.actual_dropback_rate_m74).to_numpy(dtype=float)
    base_dbr = num(test.baseline_dropback_rate).to_numpy(dtype=float)
    actual_transition = num(test.dbr_logratio_target).to_numpy(dtype=float)
    corrected_dbr = np.clip(base_dbr * np.exp(pred_logratio), DBR_MIN, DBR_MAX)

    base_attempts = num(test.pred_attempts).to_numpy(dtype=float)
    actual_attempts = num(test.actual_attempts).to_numpy(dtype=float)
    ratio = np.divide(
        corrected_dbr, base_dbr,
        out=np.ones_like(corrected_dbr),
        where=np.isfinite(base_dbr) & (base_dbr > 0),
    )
    corrected_attempts = base_attempts * ratio

    base_pass = num(test.pred_pass_yards).to_numpy(dtype=float)
    actual_pass = num(test.actual_pass_yards).to_numpy(dtype=float)
    implied_ypa = np.divide(
        base_pass, base_attempts,
        out=np.full_like(base_pass, np.nan),
        where=np.isfinite(base_attempts) & (base_attempts != 0),
    )
    corrected_pass = corrected_attempts * implied_ypa

    base_attempt_err = actual_attempts - base_attempts
    corr_attempt_err = actual_attempts - corrected_attempts
    base_pass_err = actual_pass - base_pass
    corr_pass_err = actual_pass - corrected_pass

    high = (actual_dbr - base_dbr >= HIGH_DBR_SHIFT).astype(int)
    low = (actual_dbr - base_dbr <= LOW_DBR_SHIFT).astype(int)

    return {
        "transition_corr": safe_corr(pred_logratio, actual_transition),
        "high_dbr_shift_auc": safe_auc(high, pred_logratio),
        "low_dbr_shift_auc": safe_auc(low, -pred_logratio),
        "base_dbr_mae": mae(actual_dbr, base_dbr),
        "corrected_dbr_mae": mae(actual_dbr, corrected_dbr),
        "dbr_mae_gain": mae(actual_dbr, base_dbr) - mae(actual_dbr, corrected_dbr),
        "base_dbr_corr": safe_corr(actual_dbr, base_dbr),
        "corrected_dbr_corr": safe_corr(actual_dbr, corrected_dbr),
        "dbr_corr_gain": safe_corr(actual_dbr, corrected_dbr) - safe_corr(actual_dbr, base_dbr),
        "base_attempt_mae": float(np.nanmean(np.abs(base_attempt_err))),
        "corrected_attempt_mae": float(np.nanmean(np.abs(corr_attempt_err))),
        "attempt_mae_gain": float(np.nanmean(np.abs(base_attempt_err)) - np.nanmean(np.abs(corr_attempt_err))),
        "base_attempt_corr": safe_corr(actual_attempts, base_attempts),
        "corrected_attempt_corr": safe_corr(actual_attempts, corrected_attempts),
        "attempt_corr_gain": safe_corr(actual_attempts, corrected_attempts) - safe_corr(actual_attempts, base_attempts),
        "base_attempt_10plus": int(np.nansum(np.abs(base_attempt_err) >= 10.0)),
        "corrected_attempt_10plus": int(np.nansum(np.abs(corr_attempt_err) >= 10.0)),
        "base_pass_mae": float(np.nanmean(np.abs(base_pass_err))),
        "corrected_pass_mae": float(np.nanmean(np.abs(corr_pass_err))),
        "pass_mae_gain": float(np.nanmean(np.abs(base_pass_err)) - np.nanmean(np.abs(corr_pass_err))),
        "base_pass_corr": safe_corr(actual_pass, base_pass),
        "corrected_pass_corr": safe_corr(actual_pass, corrected_pass),
        "pass_corr_gain": safe_corr(actual_pass, corrected_pass) - safe_corr(actual_pass, base_pass),
        "base_pass_100plus": int(np.nansum(np.abs(base_pass_err) >= 100.0)),
        "corrected_pass_100plus": int(np.nansum(np.abs(corr_pass_err) >= 100.0)),
        "pred_logratio": pred_logratio,
        "corrected_dbr": corrected_dbr,
        "corrected_attempts": corrected_attempts,
        "corrected_pass": corrected_pass,
    }


def full_gate(r):
    tail_limit = np.floor(r["base_attempt_10plus"] * (1.0 - MIN_ATTEMPT_TAIL_REDUCTION))
    return bool(
        r["coverage"] >= MIN_COVERAGE
        and r["transition_corr"] >= MIN_TRANSITION_CORR
        and r["dbr_mae_gain"] >= MIN_DBR_MAE_GAIN
        and r["dbr_corr_gain"] >= MIN_DBR_CORR_GAIN
        and r["attempt_mae_gain"] >= MIN_ATTEMPT_MAE_GAIN
        and r["attempt_corr_gain"] >= MIN_ATTEMPT_CORR_GAIN
        and r["corrected_attempt_10plus"] <= tail_limit
        and r["pass_mae_gain"] >= MIN_PASS_MAE_GAIN
        and r["pass_corr_gain"] >= MIN_PASS_CORR_GAIN
        and r["corrected_pass_100plus"] <= r["base_pass_100plus"]
    )


def support_gate(r):
    return bool(
        r["transition_corr"] >= SUPPORT_TRANSITION_CORR
        and r["dbr_mae_gain"] >= SUPPORT_DBR_MAE_GAIN
        and r["attempt_mae_gain"] >= SUPPORT_ATTEMPT_MAE_GAIN
        and r["pass_mae_gain"] >= SUPPORT_PASS_MAE_GAIN
        and r["corrected_attempt_10plus"] <= r["base_attempt_10plus"]
        and r["corrected_pass_100plus"] <= r["base_pass_100plus"]
    )


def univariate_screen(features, cols):
    rows = []
    y = num(features.dbr_logratio_target)
    for c in cols:
        s = num(features[c])
        vals = {}
        for season in [2024, 2025]:
            m = num(features.season).eq(season) & s.notna() & y.notna()
            vals[season] = float(s[m].corr(y[m])) if int(m.sum()) >= 20 else np.nan
        m = s.notna() & y.notna() & num(features.season).isin([2024, 2025])
        combined = float(s[m].corr(y[m])) if int(m.sum()) >= 40 else np.nan
        same = (
            np.isfinite(vals[2024]) and np.isfinite(vals[2025])
            and vals[2024] * vals[2025] > 0
        )
        strong = bool(
            same and abs(vals[2024]) >= 0.10 and abs(vals[2025]) >= 0.10
            and abs(combined) >= 0.15
        )
        rows.append({
            "feature": c,
            "corr_2024": vals[2024],
            "corr_2025": vals[2025],
            "corr_combined": combined,
            "strong_replicated": strong,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--history-seasons", default="2022,2023,2024,2025")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    base = m71.lower(pd.read_csv(a.canonical, low_memory=False))
    if len(base) != 643:
        raise RuntimeError(f"M74 canonical invariant expected 643 rows, got {len(base)}")
    base["team"] = base.team.map(m71.canon)
    base["opponent"] = base.opponent.map(m71.canon)

    seasons = [int(v) for v in a.history_seasons.split(",") if v.strip()]
    pbp, manifest = m71.load_pbp(seasons)
    features, hist_residuals, team_games = build_features(base, pbp)
    train = features[num(features.season).eq(2024)].copy().reset_index(drop=True)
    test = features[num(features.season).eq(2025)].copy().reset_index(drop=True)
    if not len(train) or not len(test):
        raise RuntimeError("M74 requires 2024 train and 2025 evaluation rows")

    rows = []
    pred_rows = []
    coverage_rows = []
    for family, cols in FAMILIES.items():
        missing = [c for c in cols if c not in features.columns]
        if missing:
            raise RuntimeError(f"M74 family {family} missing frozen columns: {missing}")
        cov = feature_coverage(test, cols)
        coverage_rows.append({
            "family": family,
            "feature_count": len(cols),
            "median_2025_coverage": cov,
        })
        for kind in ["ridge", "hgb"]:
            pred = fit_transition(train, test, cols, kind)
            ev = evaluate(test, pred)
            row = {
                "family": family,
                "model": kind,
                "coverage": cov,
                **{k: v for k, v in ev.items() if not isinstance(v, np.ndarray)},
            }
            row["full_gate"] = full_gate(row)
            row["support_gate"] = support_gate(row)
            rows.append(row)

            pr = test[[
                "season", "week", "game_id", "team", "opponent",
                "actual_pass_yards", "pred_pass_yards", "actual_attempts", "pred_attempts",
                "actual_dropback_rate_m74", "baseline_dropback_rate", "dbr_logratio_target",
            ]].copy()
            pr["family"] = family
            pr["model"] = kind
            pr["pred_dbr_logratio"] = ev["pred_logratio"]
            pr["corrected_dropback_rate"] = ev["corrected_dbr"]
            pr["corrected_attempts"] = ev["corrected_attempts"]
            pr["corrected_pass_yards"] = ev["corrected_pass"]
            pred_rows.append(pr)

    results = pd.DataFrame(rows)

    # Existing+new attribution: it must clear the base gate AND beat the same-model
    # M65 state control. New information cannot inherit credit from old state info.
    results["incremental_control_gate"] = False
    results["dbr_mae_gain_vs_state_control"] = np.nan
    results["attempt_mae_gain_vs_state_control"] = np.nan
    results["pass_mae_gain_vs_state_control"] = np.nan
    for i, r in results[results.family.eq("state_plus_new_transition")].iterrows():
        c = results[
            results.family.eq("m65_state_control") & results.model.eq(r.model)
        ]
        if c.empty:
            continue
        c = c.iloc[0]
        dbr_inc = float(c.corrected_dbr_mae - r.corrected_dbr_mae)
        att_inc = float(c.corrected_attempt_mae - r.corrected_attempt_mae)
        pass_inc = float(c.corrected_pass_mae - r.corrected_pass_mae)
        tails_ok = bool(
            r.corrected_attempt_10plus <= c.corrected_attempt_10plus
            and r.corrected_pass_100plus <= c.corrected_pass_100plus
        )
        inc = bool(dbr_inc >= 0.0025 and att_inc >= 0.10 and pass_inc >= 0.50 and tails_ok)
        results.loc[i, "dbr_mae_gain_vs_state_control"] = dbr_inc
        results.loc[i, "attempt_mae_gain_vs_state_control"] = att_inc
        results.loc[i, "pass_mae_gain_vs_state_control"] = pass_inc
        results.loc[i, "incremental_control_gate"] = inc

    supported_new = []
    for family in NEW_STANDALONE_FAMILIES:
        q = results[results.family.eq(family)]
        if len(q) != 2:
            continue
        for _, winner in q.iterrows():
            other = q[q.model.ne(winner.model)]
            if bool(winner.full_gate) and len(other) and bool(other.iloc[0].support_gate):
                supported_new.append(family)
                break

    combo_supported = False
    combo = results[results.family.eq("state_plus_new_transition")]
    if len(combo) == 2:
        for _, winner in combo.iterrows():
            other = combo[combo.model.ne(winner.model)]
            if (
                bool(winner.full_gate)
                and bool(winner.incremental_control_gate)
                and len(other)
                and bool(other.iloc[0].support_gate)
            ):
                combo_supported = True
                break

    state = results[results.family.eq("m65_state_control")]
    state_supported = False
    if len(state) == 2:
        for _, winner in state.iterrows():
            other = state[state.model.ne(winner.model)]
            if bool(winner.full_gate) and len(other) and bool(other.iloc[0].support_gate):
                state_supported = True
                break

    if supported_new or combo_supported:
        verdict = "m74_dbr_transition_signal_followup"
    elif state_supported:
        verdict = "m74_existing_state_signal_only_no_new_transition_breakthrough"
    else:
        verdict = "m74_dbr_shift_not_predictable_with_current_opening_inducement_information"

    uni = univariate_screen(features, list(dict.fromkeys(COMBINED_NEW)))
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    features.to_csv(out / "m74_game_features_and_targets.csv", index=False)
    hist_residuals.to_csv(out / "m74_historical_dbr_residuals.csv", index=False)
    results.to_csv(out / "m74_model_results.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_csv(out / "m74_2025_predictions.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(out / "m74_feature_coverage.csv", index=False)
    uni.to_csv(out / "m74_univariate_transition_screen.csv", index=False)
    manifest.to_csv(out / "m74_source_manifest.csv", index=False)
    pd.DataFrame([{
        "train_rows_2024": int(len(train)),
        "evaluation_rows_2025": int(len(test)),
        "supported_new_families": "|".join(sorted(set(supported_new))),
        "state_plus_new_incremental_supported": bool(combo_supported),
        "m65_state_control_supported": bool(state_supported),
        "strong_replicated_new_features": int(uni.strong_replicated.sum()),
        "m74_interpretation": verdict,
        "production_actionable": False,
    }]).to_csv(out / "m74_precommitted_interpretation.csv", index=False)

    print("=== M74 INTERPRETATION ===")
    print(pd.read_csv(out / "m74_precommitted_interpretation.csv").to_string(index=False))
    print("=== M74 MODEL RESULTS ===")
    print(results.to_string(index=False))
    print("=== M74 REPLICATED UNIVARIATE ===")
    print(uni[uni.strong_replicated].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
