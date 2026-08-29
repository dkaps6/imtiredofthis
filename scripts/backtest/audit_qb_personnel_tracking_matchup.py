#!/usr/bin/env python3
"""Migration 75 — genuinely-new QB personnel/tracking matchup audit.

M75 is intentionally outside the M56-M74 team-level feature loop. It tests
weekly NFL Next Gen Stats receiver tracking and weekly PFR advanced defender
coverage data as new pregame information for QB efficiency.

Frozen scientific boundary:
- qb_frontier_canonical_v1 supplies immutable Raw QB points/outcomes.
- 2024 trains fixed models; 2025 is untouched evaluation.
- 2022/2023 are strictly-prior history only.
- all target-week NGS/PFR rows are outcomes and are NEVER features.
- receiver/defender features use only games strictly before the target week.
- no sportsbook player-prop or game-market feature.
- no subset search, model zoo, threshold retuning or hyperparameter sweep.
- M75 is diagnostic; production_actionable is always False.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts._opponent_map import canon_team

HIST = 8
RECENT = 3
MIN_COVERAGE = 0.70
MIN_YPA_RESID_CORR = 0.18
MIN_YPA_MAE_GAIN = 0.10
MIN_PASS_MAE_GAIN = 0.75
MIN_PASS_CORR_GAIN = 0.015
SUPPORT_CORR = 0.08
SUPPORT_PASS_GAIN = 0.20


def num(x):
    return pd.to_numeric(x, errors="coerce")


def to_pd(x):
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if hasattr(x, "to_pandas"):
        return x.to_pandas()
    return pd.DataFrame(x)


def lower(x):
    y = x.copy()
    y.columns = [str(c).strip().lower() for c in y.columns]
    return y


def canon(v):
    t = canon_team(v)
    return "WAS" if t == "WSH" else t


def first_col(df, names):
    for c in names:
        if c in df.columns:
            return c
    return None


def regular_week_rows(df):
    q = df.copy()
    if "season_type" in q:
        q = q[q.season_type.fillna("").astype(str).str.upper().isin(["REG", "REGULAR", "RS", ""])].copy()
    if "game_type" in q:
        q = q[q.game_type.fillna("").astype(str).str.upper().isin(["REG", "REGULAR", "RS", ""])].copy()
    if "week" in q:
        q = q[num(q.week).gt(0)].copy()
    return q


def safe_corr(a, b):
    z = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    if len(z) < 3 or z.a.nunique() < 2 or z.b.nunique() < 2:
        return np.nan
    return float(z.a.corr(z.b))


def wmean(v, w):
    z = pd.DataFrame({"v": num(v), "w": num(w)}).dropna()
    z = z[z.w.gt(0)]
    if z.empty or z.w.sum() <= 0:
        return np.nan
    return float(np.average(z.v, weights=z.w))


def norm_name(v):
    return re.sub(r"[^a-z]", "", str(v).lower())


def load_sources(seasons):
    import nflreadpy as nfl

    ngs_err = ""
    pfr_err = ""
    try:
        ngs = lower(to_pd(nfl.load_nextgen_stats(seasons=seasons, stat_type="receiving")))
    except Exception as e:
        ngs = pd.DataFrame()
        ngs_err = f"{type(e).__name__}:{e}"

    try:
        pfr = lower(to_pd(nfl.load_pfr_advstats(seasons=seasons, stat_type="def", summary_level="week")))
    except Exception as e:
        pfr = pd.DataFrame()
        pfr_err = f"{type(e).__name__}:{e}"

    try:
        players = lower(to_pd(nfl.load_players()))
    except Exception:
        players = pd.DataFrame()

    return ngs, pfr, players, ngs_err, pfr_err


def build_ngs_team_week(ngs):
    if ngs.empty:
        return pd.DataFrame(), {"usable": False, "reason": "ngs_empty"}
    q = regular_week_rows(ngs)
    team_col = first_col(q, ["team_abbr", "team"])
    target_col = first_col(q, ["targets"])
    rec_col = first_col(q, ["receptions"])
    sep_col = first_col(q, ["avg_separation"])
    cushion_col = first_col(q, ["avg_cushion"])
    adot_col = first_col(q, ["avg_intended_air_yards", "avg_air_distance"])
    yacoe_col = first_col(q, ["avg_yac_above_expectation"])
    eyac_col = first_col(q, ["avg_expected_yac"])
    iay_col = first_col(q, ["percent_share_of_intended_air_yards"])
    if not all([team_col, target_col, sep_col, cushion_col, adot_col, yacoe_col]):
        return pd.DataFrame(), {
            "usable": False,
            "reason": "missing_required_ngs_columns",
            "columns": "|".join(q.columns),
        }
    q["team"] = q[team_col].map(canon)
    q["targets_n"] = num(q[target_col]).fillna(0)
    q["receptions_n"] = num(q[rec_col]).fillna(0) if rec_col else 0.0
    rows = []
    for (season, week, team), g in q.groupby(["season", "week", "team"], dropna=False):
        g = g[g.targets_n.gt(0)].copy().sort_values("targets_n", ascending=False)
        if g.empty:
            continue
        total = float(g.targets_n.sum())
        rec = {
            "season": int(season), "week": int(week), "team": team,
            "ngs_targets": total,
            "ngs_sep": wmean(g[sep_col], g.targets_n),
            "ngs_cushion": wmean(g[cushion_col], g.targets_n),
            "ngs_adot": wmean(g[adot_col], g.targets_n),
            "ngs_yacoe": wmean(g[yacoe_col], g.receptions_n.where(g.receptions_n.gt(0), g.targets_n)),
            "ngs_expected_yac": wmean(g[eyac_col], g.receptions_n.where(g.receptions_n.gt(0), g.targets_n)) if eyac_col else np.nan,
            "ngs_iay_share": wmean(g[iay_col], g.targets_n) if iay_col else np.nan,
            "ngs_top1_target_share": float(g.iloc[0].targets_n / total),
            "ngs_top2_target_share": float(g.head(2).targets_n.sum() / total),
            "ngs_top1_sep": float(num(pd.Series([g.iloc[0][sep_col]])).iloc[0]),
            "ngs_top1_adot": float(num(pd.Series([g.iloc[0][adot_col]])).iloc[0]),
            "ngs_top1_yacoe": float(num(pd.Series([g.iloc[0][yacoe_col]])).iloc[0]),
            "ngs_receivers": int(len(g)),
        }
        rows.append(rec)
    return pd.DataFrame(rows), {"usable": True, "reason": "ok", "rows": len(rows)}


def attach_positions(pfr, players):
    q = pfr.copy()
    if "position" in q and q.position.notna().any():
        return q, "native_position"
    if players.empty:
        q["position"] = ""
        return q, "position_unavailable"
    pfr_id = first_col(q, ["pfr_player_id", "pfr_id"])
    player_id = first_col(players, ["pfr_id", "pfr_player_id"])
    pos = first_col(players, ["position", "position_group"])
    if pfr_id and player_id and pos:
        bridge = players[[player_id, pos]].dropna().drop_duplicates(player_id).rename(columns={player_id: "_pfr", pos: "_pos"})
        q = q.merge(bridge, left_on=pfr_id, right_on="_pfr", how="left")
        q["position"] = q["_pos"]
        return q.drop(columns=[c for c in ["_pfr", "_pos"] if c in q]), "players_pfr_bridge"
    q["position"] = ""
    return q, "position_unavailable"


def build_pfr_secondary_week(pfr, players):
    if pfr.empty:
        return pd.DataFrame(), {"usable": False, "reason": "pfr_empty"}
    q = regular_week_rows(pfr)
    team_col = first_col(q, ["team", "team_abbr"])
    tgt = first_col(q, ["targets", "tgt", "times_targeted"])
    cmpc = first_col(q, ["completions", "cmp", "completions_allowed"])
    yards = first_col(q, ["yards", "yds", "yards_allowed"])
    ypt = first_col(q, ["yards_per_target", "yds_per_target", "yds_tgt"])
    cmp_pct = first_col(q, ["completion_pct", "cmp_pct", "completion_percentage"])
    rating = first_col(q, ["passer_rating", "rating", "rat"])
    adot = first_col(q, ["average_depth_of_target", "avg_depth_of_target", "adot", "d_adot"])
    yac = first_col(q, ["yards_after_catch", "yac", "yards_after_catch_allowed"])
    if not all([team_col, tgt]) or not (yards or ypt):
        return pd.DataFrame(), {
            "usable": False,
            "reason": "missing_required_pfr_coverage_columns",
            "columns": "|".join(q.columns),
        }
    q, pos_source = attach_positions(q, players)
    q["team"] = q[team_col].map(canon)
    q["targets_n"] = num(q[tgt]).fillna(0)
    if ypt:
        q["ypt_n"] = num(q[ypt])
    else:
        q["ypt_n"] = np.divide(num(q[yards]), q.targets_n, out=np.full(len(q), np.nan), where=q.targets_n.to_numpy() > 0)
    if cmp_pct:
        q["cmp_pct_n"] = num(q[cmp_pct])
    elif cmpc:
        q["cmp_pct_n"] = np.divide(num(q[cmpc]), q.targets_n, out=np.full(len(q), np.nan), where=q.targets_n.to_numpy() > 0)
    else:
        q["cmp_pct_n"] = np.nan
    q["rating_n"] = num(q[rating]) if rating else np.nan
    q["adot_n"] = num(q[adot]) if adot else np.nan
    q["yac_n"] = num(q[yac]) if yac else np.nan
    pos = q.position.fillna("").astype(str).str.upper()
    dbmask = pos.isin(["CB", "DB", "S", "FS", "SS", "NB"])
    position_filtered = bool(dbmask.any())
    if position_filtered:
        q = q[dbmask].copy()
    rows = []
    for (season, week, team), g in q.groupby(["season", "week", "team"], dropna=False):
        g = g[g.targets_n.gt(0)].copy()
        if g.empty:
            continue
        meaningful = g[g.targets_n.ge(2)]
        rec = {
            "season": int(season), "week": int(week), "team": team,
            "db_targets": float(g.targets_n.sum()),
            "db_ypt": wmean(g.ypt_n, g.targets_n),
            "db_cmp_pct": wmean(g.cmp_pct_n, g.targets_n),
            "db_rating": wmean(g.rating_n, g.targets_n),
            "db_adot": wmean(g.adot_n, g.targets_n),
            "db_yac": wmean(g.yac_n, g.targets_n),
            "db_weak_ypt": float(num(meaningful.ypt_n).max()) if len(meaningful) else np.nan,
            "db_weak_rating": float(num(meaningful.rating_n).max()) if len(meaningful) else np.nan,
            "db_coverage_players": int(len(g)),
        }
        rows.append(rec)
    return pd.DataFrame(rows), {
        "usable": bool(len(rows)), "reason": "ok" if len(rows) else "no_team_week_rows",
        "rows": len(rows), "position_filtered": position_filtered,
        "position_source": pos_source,
        "target_col": tgt, "yards_col": yards or "", "ypt_col": ypt or "",
        "cmp_pct_col": cmp_pct or "", "rating_col": rating or "",
        "adot_col": adot or "", "yac_col": yac or "",
    }


def prior_rows(df, team, season, week):
    if df.empty:
        return df
    m = df.team.eq(team) & ((num(df.season) < season) | ((num(df.season) == season) & (num(df.week) < week)))
    return df[m].sort_values(["season", "week"]).tail(HIST)


def add_roll(rec, hist, prefix, cols):
    for c in cols:
        vals = num(hist[c]).dropna().to_numpy(dtype=float) if c in hist else np.asarray([])
        rec[f"{prefix}_{c}_last1"] = float(vals[-1]) if len(vals) else np.nan
        rec[f"{prefix}_{c}_mean3"] = float(np.mean(vals[-RECENT:])) if len(vals) else np.nan
        rec[f"{prefix}_{c}_mean8"] = float(np.mean(vals[-HIST:])) if len(vals) else np.nan
        rec[f"{prefix}_{c}_delta3_8"] = (
            rec[f"{prefix}_{c}_mean3"] - rec[f"{prefix}_{c}_mean8"] if len(vals) else np.nan
        )


def build_features(base, ngs_week, db_week):
    ngs_cols = ["ngs_sep", "ngs_cushion", "ngs_adot", "ngs_yacoe", "ngs_expected_yac", "ngs_top1_target_share", "ngs_top2_target_share", "ngs_top1_sep", "ngs_top1_adot", "ngs_top1_yacoe"]
    db_cols = ["db_ypt", "db_cmp_pct", "db_rating", "db_adot", "db_yac", "db_weak_ypt", "db_weak_rating"]
    rows = []
    for _, r in base.iterrows():
        season, week = int(r.season), int(r.week)
        team, opp = canon(r.team), canon(r.opponent)
        rec = r.to_dict()
        rec["team"] = team; rec["opponent"] = opp
        nh = prior_rows(ngs_week, team, season, week)
        dh = prior_rows(db_week, opp, season, week)
        add_roll(rec, nh, "off", ngs_cols)
        add_roll(rec, dh, "def", db_cols)
        # Frozen football-motivated interactions, not a searched interaction zoo.
        rec["x_sep_ypt"] = rec.get("off_ngs_sep_mean3", np.nan) * rec.get("def_db_ypt_mean3", np.nan)
        rec["x_yacoe_yac"] = rec.get("off_ngs_yacoe_mean3", np.nan) * rec.get("def_db_yac_mean3", np.nan)
        rec["x_adot_adot"] = rec.get("off_ngs_adot_mean3", np.nan) * rec.get("def_db_adot_mean3", np.nan)
        rec["x_top1_weak_ypt"] = rec.get("off_ngs_top1_target_share_mean3", np.nan) * rec.get("def_db_weak_ypt_mean3", np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def make_model(kind):
    if kind == "ridge":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=50.0))])
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(loss="absolute_error", max_iter=150, learning_rate=.04, max_depth=2, min_samples_leaf=15, l2_regularization=5.0, random_state=75))])


def evaluate(test, pred_resid):
    actual_att = num(test.actual_attempts).to_numpy(float)
    pred_att = num(test.pred_attempts).to_numpy(float)
    actual_pass = num(test.actual_pass_yards).to_numpy(float)
    pred_pass = num(test.pred_pass_yards).to_numpy(float)
    actual_ypa = np.divide(actual_pass, actual_att, out=np.full(len(test), np.nan), where=actual_att > 0)
    pred_ypa = np.divide(pred_pass, pred_att, out=np.full(len(test), np.nan), where=pred_att > 0)
    corrected_ypa = np.clip(pred_ypa + pred_resid, 2.0, 14.0)
    corrected_pass = pred_att * corrected_ypa
    base_err = actual_pass - pred_pass
    corr_err = actual_pass - corrected_pass
    base_ypa_mae = float(np.nanmean(np.abs(actual_ypa - pred_ypa)))
    corr_ypa_mae = float(np.nanmean(np.abs(actual_ypa - corrected_ypa)))
    return {
        "ypa_residual_corr": safe_corr(actual_ypa - pred_ypa, pred_resid),
        "base_ypa_mae": base_ypa_mae,
        "corrected_ypa_mae": corr_ypa_mae,
        "ypa_mae_gain": base_ypa_mae - corr_ypa_mae,
        "base_pass_mae": float(np.nanmean(np.abs(base_err))),
        "corrected_pass_mae": float(np.nanmean(np.abs(corr_err))),
        "pass_mae_gain": float(np.nanmean(np.abs(base_err)) - np.nanmean(np.abs(corr_err))),
        "base_pass_corr": safe_corr(actual_pass, pred_pass),
        "corrected_pass_corr": safe_corr(actual_pass, corrected_pass),
        "pass_corr_gain": safe_corr(actual_pass, corrected_pass) - safe_corr(actual_pass, pred_pass),
        "base_100plus": int(np.nansum(np.abs(base_err) >= 100)),
        "corrected_100plus": int(np.nansum(np.abs(corr_err) >= 100)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seasons", default="2022,2023,2024,2025")
    a = ap.parse_args()
    out = a.out_dir; out.mkdir(parents=True, exist_ok=True)

    base = lower(pd.read_csv(a.canonical, low_memory=False))
    if len(base) != 643:
        raise RuntimeError(f"M75 expected frozen 643-row canonical frontier, got {len(base)}")
    base["team"] = base.team.map(canon); base["opponent"] = base.opponent.map(canon)
    seasons = [int(v) for v in a.seasons.split(",") if v.strip()]

    ngs, pfr, players, ngs_err, pfr_err = load_sources(seasons)
    ngs_week, ngs_meta = build_ngs_team_week(ngs)
    db_week, db_meta = build_pfr_secondary_week(pfr, players)
    features = build_features(base, ngs_week, db_week)

    off_cols = [c for c in features if c.startswith("off_ngs_")]
    def_cols = [c for c in features if c.startswith("def_db_")]
    x_cols = ["x_sep_ypt", "x_yacoe_yac", "x_adot_adot", "x_top1_weak_ypt"]
    families = {
        "ngs_receiving_tracking": off_cols,
        "pfr_secondary_coverage": def_cols,
        "tracking_x_secondary": list(dict.fromkeys(off_cols + def_cols + x_cols)),
        "combined_personnel_tracking": list(dict.fromkeys(off_cols + def_cols + x_cols)),
    }

    train = features[num(features.season).eq(2024)].copy().reset_index(drop=True)
    test = features[num(features.season).eq(2025)].copy().reset_index(drop=True)
    train["ypa_resid"] = num(train.actual_pass_yards) / num(train.actual_attempts) - num(train.pred_pass_yards) / num(train.pred_attempts)
    test["ypa_resid"] = num(test.actual_pass_yards) / num(test.actual_attempts) - num(test.pred_pass_yards) / num(test.pred_attempts)

    rows = []
    for family, cols in families.items():
        if not cols:
            continue
        cov = float(test[cols].notna().mean().median())
        for kind in ["ridge", "hgb"]:
            model = make_model(kind)
            ok = train.ypa_resid.notna()
            model.fit(train.loc[ok, cols], train.loc[ok, "ypa_resid"])
            pred = np.asarray(model.predict(test[cols]), dtype=float)
            ev = evaluate(test, pred)
            full = bool(cov >= MIN_COVERAGE and ev["ypa_residual_corr"] >= MIN_YPA_RESID_CORR and ev["ypa_mae_gain"] >= MIN_YPA_MAE_GAIN and ev["pass_mae_gain"] >= MIN_PASS_MAE_GAIN and ev["pass_corr_gain"] >= MIN_PASS_CORR_GAIN and ev["corrected_100plus"] <= ev["base_100plus"])
            support = bool(cov >= MIN_COVERAGE and ev["ypa_residual_corr"] >= SUPPORT_CORR and ev["pass_mae_gain"] >= SUPPORT_PASS_GAIN and ev["corrected_100plus"] <= ev["base_100plus"])
            rows.append({"family": family, "model": kind, "feature_count": len(cols), "coverage": cov, **ev, "full_gate": full, "support_gate": support})
    results = pd.DataFrame(rows)

    supported = []
    for family in results.family.unique() if len(results) else []:
        q = results[results.family.eq(family)]
        if len(q) != 2:
            continue
        for _, w in q.iterrows():
            o = q[q.model.ne(w.model)]
            if bool(w.full_gate) and len(o) and bool(o.iloc[0].support_gate):
                supported.append(family); break

    source_rows = [
        {"source": "nflverse_ngs_receiving_weekly", "usable": bool(ngs_meta.get("usable")), "rows_raw": len(ngs), "rows_team_week": len(ngs_week), "error": ngs_err, "detail": str(ngs_meta)},
        {"source": "nflverse_pfr_advstats_def_weekly", "usable": bool(db_meta.get("usable")), "rows_raw": len(pfr), "rows_team_week": len(db_week), "error": pfr_err, "detail": str(db_meta)},
    ]
    source_df = pd.DataFrame(source_rows)
    if supported:
        verdict = "m75_personnel_tracking_new_information_signal"
    elif bool(ngs_meta.get("usable")) and bool(db_meta.get("usable")):
        verdict = "m75_sources_qualified_no_predictive_breakthrough"
    elif bool(ngs_meta.get("usable")):
        verdict = "m75_receiver_tracking_qualified_secondary_contract_incomplete"
    else:
        verdict = "m75_free_personnel_sources_not_qualified"

    source_df.to_csv(out / "m75_source_manifest.csv", index=False)
    ngs_week.to_csv(out / "m75_ngs_receiver_team_week.csv", index=False)
    db_week.to_csv(out / "m75_pfr_secondary_team_week.csv", index=False)
    features.to_csv(out / "m75_game_features.csv", index=False)
    results.to_csv(out / "m75_model_results.csv", index=False)
    pd.DataFrame([{
        "train_rows_2024": len(train), "evaluation_rows_2025": len(test),
        "supported_families": "|".join(sorted(set(supported))),
        "m75_interpretation": verdict, "production_actionable": False,
    }]).to_csv(out / "m75_precommitted_interpretation.csv", index=False)

    print("=== M75 SOURCES ==="); print(source_df.to_string(index=False))
    print("=== M75 RESULTS ==="); print(results.to_string(index=False))
    print("=== M75 INTERPRETATION ==="); print(pd.read_csv(out / "m75_precommitted_interpretation.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
