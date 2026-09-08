#!/usr/bin/env python3
"""RB-R13 diagnostic: strict-prior receiving-efficiency identity.

This is diagnostic-only. It does not change R9/R12, production parameters, target
pools, or sportsbook inputs. The purpose is to test whether persistent pregame RB
receiving-efficiency history predicts which games land in high yards-per-target
states after the R12 entitlement work exposed frozen YPT as a major error source.

Current-game targets/yards are labels only. Every candidate feature is built from
completed games strictly before kickoff.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.audit_rb_receiving_identity_v1 import _load_logs

PRIMARY_YPT = ["prior_ypt", "last8_ypt", "prev_season_ypt", "same_team_ypt"]
EFF_FEATURES = PRIMARY_YPT + [
    "prior_catch_rate", "last8_catch_rate", "prev_season_catch_rate", "same_team_catch_rate",
    "prior_ypr", "last8_ypr", "prev_season_ypr", "same_team_ypr",
    "prior_rec_yards_pg", "last8_rec_yards_pg", "prev_season_rec_yards_pg",
    "prior_40plus_rec_rate", "prior_60plus_rec_rate",
    "prior_high8_ypt_rate_3plus", "prior_high10_ypt_rate_3plus",
    "frozen_ypt", "state_probability",
]

# Frozen diagnostic thresholds before results.
MIN_PRIMARY_SPEARMAN = 0.08
MIN_PRIMARY_HIGH8_AUC = 0.55
MIN_POSITIVE_SEASONS = 2
MIN_STANDALONE_YPT_MAE_GAIN = 0.05
MIN_STANDALONE_NONWORSE_SEASONS = 2


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _ratio(a, b):
    a = _num(a); b = _num(b)
    return np.where(b.gt(0), a / b, np.nan)


def _series_state(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("time_key").copy()
    t = _num(g.targets).fillna(0.0)
    r = _num(g.receptions).fillna(0.0)
    y = _num(g.rec_yards).fillna(0.0)
    game_ypt = pd.Series(np.where(t.gt(0), y / t, np.nan), index=g.index)

    ct, cr, cy = t.cumsum(), r.cumsum(), y.cumsum()
    games = np.arange(1, len(g) + 1, dtype=float)
    g["after_ypt"] = _ratio(cy, ct)
    g["after_catch_rate"] = _ratio(cr, ct)
    g["after_ypr"] = _ratio(cy, cr)
    g["after_rec_yards_pg"] = cy / games
    g["after_targets"] = ct

    rt = t.rolling(8, min_periods=1).sum()
    rr = r.rolling(8, min_periods=1).sum()
    ry = y.rolling(8, min_periods=1).sum()
    rg = pd.Series(np.minimum(np.arange(1, len(g) + 1), 8), index=g.index, dtype=float)
    g["after_last8_ypt"] = _ratio(ry, rt)
    g["after_last8_catch_rate"] = _ratio(rr, rt)
    g["after_last8_ypr"] = _ratio(ry, rr)
    g["after_last8_rec_yards_pg"] = ry / rg

    g["after_40plus_rec_rate"] = y.ge(40).astype(float).cumsum() / games
    g["after_60plus_rec_rate"] = y.ge(60).astype(float).cumsum() / games
    elig = t.ge(3).astype(float).cumsum()
    h8 = (t.ge(3) & game_ypt.ge(8)).astype(float).cumsum()
    h10 = (t.ge(3) & game_ypt.ge(10)).astype(float).cumsum()
    g["after_high8_ypt_rate_3plus"] = np.where(elig.gt(0), h8 / elig, np.nan)
    g["after_high10_ypt_rate_3plus"] = np.where(elig.gt(0), h10 / elig, np.nan)
    g["eff_source_time_key"] = g.time_key
    return g


def _same_team_state(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("time_key").copy()
    t = _num(g.targets).fillna(0.0); r = _num(g.receptions).fillna(0.0); y = _num(g.rec_yards).fillna(0.0)
    ct, cr, cy = t.cumsum(), r.cumsum(), y.cumsum()
    g["same_team_after_ypt"] = _ratio(cy, ct)
    g["same_team_after_catch_rate"] = _ratio(cr, ct)
    g["same_team_after_ypr"] = _ratio(cy, cr)
    g["same_team_eff_source_time_key"] = g.time_key
    return g


def _prev_season(rb: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (key, season), g in rb.groupby(["player_clean_key", "season"], dropna=False):
        t = float(_num(g.targets).fillna(0).sum()); r = float(_num(g.receptions).fillna(0).sum()); y = float(_num(g.rec_yards).fillna(0).sum())
        rows.append({
            "player_clean_key": key, "season": int(season) + 1,
            "prev_season_ypt": y / t if t > 0 else np.nan,
            "prev_season_catch_rate": r / t if t > 0 else np.nan,
            "prev_season_ypr": y / r if r > 0 else np.nan,
            "prev_season_rec_yards_pg": y / len(g) if len(g) else np.nan,
            "prev_season_targets": t,
        })
    return pd.DataFrame(rows)


def _attach_history(q: pd.DataFrame, history_start: int) -> tuple[pd.DataFrame, dict]:
    through = int(_num(q.season).max())
    logs = _load_logs(list(range(int(history_start), through + 1)))
    rb = logs.loc[logs.position_family.isin({"RB", "FB"})].copy()
    rb["time_key"] = _num(rb.season).astype(int) * 100 + _num(rb.week).astype(int)

    states = pd.concat([_series_state(g) for _, g in rb.groupby("player_clean_key", sort=False)], ignore_index=True)
    team_states = pd.concat([_same_team_state(g) for _, g in rb.groupby(["player_clean_key", "team"], sort=False)], ignore_index=True)
    prev = _prev_season(rb)

    x = q.copy().reset_index(drop=True)
    x["time_key"] = _num(x.season).astype(int) * 100 + _num(x.week).astype(int)
    x["_qrow"] = np.arange(len(x))

    s = states[[
        "player_clean_key", "time_key", "eff_source_time_key", "after_targets",
        "after_ypt", "after_catch_rate", "after_ypr", "after_rec_yards_pg",
        "after_last8_ypt", "after_last8_catch_rate", "after_last8_ypr", "after_last8_rec_yards_pg",
        "after_40plus_rec_rate", "after_60plus_rec_rate", "after_high8_ypt_rate_3plus", "after_high10_ypt_rate_3plus",
    ]].sort_values(["time_key", "player_clean_key"])
    x = pd.merge_asof(
        x.sort_values(["time_key", "player_clean_key"]), s,
        on="time_key", by="player_clean_key", direction="backward", allow_exact_matches=False,
    )
    x = x.rename(columns={
        "after_ypt":"prior_ypt", "after_catch_rate":"prior_catch_rate", "after_ypr":"prior_ypr",
        "after_rec_yards_pg":"prior_rec_yards_pg", "after_last8_ypt":"last8_ypt",
        "after_last8_catch_rate":"last8_catch_rate", "after_last8_ypr":"last8_ypr",
        "after_last8_rec_yards_pg":"last8_rec_yards_pg", "after_40plus_rec_rate":"prior_40plus_rec_rate",
        "after_60plus_rec_rate":"prior_60plus_rec_rate", "after_high8_ypt_rate_3plus":"prior_high8_ypt_rate_3plus",
        "after_high10_ypt_rate_3plus":"prior_high10_ypt_rate_3plus", "after_targets":"prior_targets",
    })

    ts = team_states[[
        "player_clean_key", "team", "time_key", "same_team_eff_source_time_key",
        "same_team_after_ypt", "same_team_after_catch_rate", "same_team_after_ypr",
    ]].sort_values(["time_key", "player_clean_key", "team"])
    x = pd.merge_asof(
        x.sort_values(["time_key", "player_clean_key", "team"]), ts,
        on="time_key", by=["player_clean_key", "team"], direction="backward", allow_exact_matches=False,
    ).rename(columns={
        "same_team_after_ypt":"same_team_ypt", "same_team_after_catch_rate":"same_team_catch_rate",
        "same_team_after_ypr":"same_team_ypr",
    })
    x = x.merge(prev, on=["player_clean_key", "season"], how="left")
    x = x.sort_values("_qrow").drop(columns="_qrow").reset_index(drop=True)

    v1 = int((x.eff_source_time_key.notna() & (x.eff_source_time_key >= x.time_key)).sum())
    v2 = int((x.same_team_eff_source_time_key.notna() & (x.same_team_eff_source_time_key >= x.time_key)).sum())
    audit = {
        "history_start": int(history_start), "through_season": through,
        "history_rb_rows": int(len(rb)), "query_rows": int(len(x)),
        "strict_prior_player_time_violations": v1, "strict_prior_same_team_time_violations": v2,
        "sportsbook_inputs_added": 0,
    }
    return x, audit


def _auc(y: pd.Series, score: pd.Series) -> float:
    y = _num(y); score = _num(score)
    ok = y.notna() & score.notna()
    y = y.loc[ok].astype(int); score = score.loc[ok]
    n1 = int(y.sum()); n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = score.rank(method="average")
    return float((ranks.loc[y.eq(1)].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _feature_row(g: pd.DataFrame, feature: str, season_bucket: str, population: str) -> dict | None:
    a = _num(g.actual_ypt); f = _num(g[feature])
    ok = a.notna() & f.notna()
    z = g.loc[ok].copy()
    if len(z) < 50:
        return None
    a = _num(z.actual_ypt); f = _num(z[feature])
    pearson = float(a.corr(f, method="pearson")) if a.std() > 0 and f.std() > 0 else np.nan
    spearman = float(a.corr(f, method="spearman")) if a.nunique() > 1 and f.nunique() > 1 else np.nan
    y8 = a.ge(8).astype(int); y10 = a.ge(10).astype(int)
    pct = z.assign(_f=f).groupby(["season", "week"], dropna=False)["_f"].rank(pct=True, method="average")
    top = pct.gt(.80)
    e8 = y8.mean(); e10 = y10.mean()
    top8 = y8.loc[top].mean() if top.any() else np.nan
    top10 = y10.loc[top].mean() if top.any() else np.nan
    cap8 = float(y8.loc[top].sum() / y8.sum()) if y8.sum() else np.nan
    cap10 = float(y10.loc[top].sum() / y10.sum()) if y10.sum() else np.nan
    out = {
        "season_bucket":season_bucket, "population":population, "feature":feature, "n":int(len(z)),
        "pearson_actual_ypt":pearson, "spearman_actual_ypt":spearman,
        "high8_auc":_auc(y8, f), "high10_auc":_auc(y10, f),
        "high8_event_rate":float(e8), "high10_event_rate":float(e10),
        "top20_high8_rate":float(top8) if pd.notna(top8) else np.nan,
        "top20_high10_rate":float(top10) if pd.notna(top10) else np.nan,
        "top20_high8_lift":float(top8 / e8) if pd.notna(top8) and e8 > 0 else np.nan,
        "top20_high10_lift":float(top10 / e10) if pd.notna(top10) and e10 > 0 else np.nan,
        "top20_high8_capture":cap8, "top20_high10_capture":cap10,
    }
    if feature in PRIMARY_YPT or feature == "frozen_ypt":
        out["direct_ypt_mae"] = float((f - a).abs().mean())
    else:
        out["direct_ypt_mae"] = np.nan
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--history-start", type=int, default=2013)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    x = pd.read_csv(a.predictions, low_memory=False)
    req = {"season","week","team","player_clean_key","identity_bucket","actual_targets","actual_rec_yards","frozen_ypt","state_probability"}
    missing = sorted(req - set(x.columns))
    if missing:
        raise RuntimeError(f"R13 predictions missing columns: {missing}")
    if x.duplicated(["season","week","team","player_clean_key"]).any():
        raise RuntimeError("R13 requires one row per RB player-game")

    x, audit = _attach_history(x, a.history_start)
    x["actual_targets"] = _num(x.actual_targets)
    x["actual_rec_yards"] = _num(x.actual_rec_yards)
    x["actual_ypt"] = np.where(x.actual_targets.gt(0), x.actual_rec_yards / x.actual_targets, np.nan)
    x["eff_eval_population"] = np.where(x.actual_targets.ge(3), "3PLUS_TARGETS", np.where(x.actual_targets.gt(0), "1_2_TARGETS", "ZERO_TARGETS"))

    rows = []
    for season_bucket, g0 in [("COMBINED", x)] + [(str(int(s)), g) for s, g in x.groupby("season")]:
        for pop, g1 in [("ALL_RB", g0), ("TOP20_IDENTITY", g0.loc[g0.identity_bucket.eq("TOP20")]), ("REST80_IDENTITY", g0.loc[g0.identity_bucket.eq("REST80")])]:
            g = g1.loc[g1.actual_targets.ge(3)].copy()
            for f in EFF_FEATURES:
                if f in g.columns:
                    r = _feature_row(g, f, season_bucket, pop)
                    if r is not None:
                        rows.append(r)
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("R13 produced zero feature summaries")

    combined = summary.loc[(summary.season_bucket.eq("COMBINED")) & summary.population.eq("ALL_RB")].copy()
    frozen = combined.loc[combined.feature.eq("frozen_ypt")]
    if frozen.empty:
        raise RuntimeError("R13 missing frozen_ypt benchmark")
    frozen_mae = float(frozen.iloc[0].direct_ypt_mae)

    primary = combined.loc[combined.feature.isin(PRIMARY_YPT)].copy()
    primary["positive_seasons"] = 0
    primary["nonworse_mae_seasons_vs_frozen"] = 0
    for i, r in primary.iterrows():
        feat = r.feature
        pos = 0; nonworse = 0
        for season in (2023, 2024, 2025):
            q = summary.loc[(summary.season_bucket.eq(str(season))) & summary.population.eq("ALL_RB") & summary.feature.eq(feat)]
            b = summary.loc[(summary.season_bucket.eq(str(season))) & summary.population.eq("ALL_RB") & summary.feature.eq("frozen_ypt")]
            if len(q) == 1:
                pos += int(pd.notna(q.iloc[0].spearman_actual_ypt) and float(q.iloc[0].spearman_actual_ypt) > 0)
            if len(q) == 1 and len(b) == 1 and pd.notna(q.iloc[0].direct_ypt_mae) and pd.notna(b.iloc[0].direct_ypt_mae):
                nonworse += int(float(q.iloc[0].direct_ypt_mae) <= float(b.iloc[0].direct_ypt_mae) + 1e-12)
        primary.loc[i, "positive_seasons"] = pos
        primary.loc[i, "nonworse_mae_seasons_vs_frozen"] = nonworse
    primary["direct_ypt_mae_gain_vs_frozen"] = frozen_mae - _num(primary.direct_ypt_mae)

    signal_mask = (
        _num(primary.spearman_actual_ypt).ge(MIN_PRIMARY_SPEARMAN)
        & _num(primary.high8_auc).ge(MIN_PRIMARY_HIGH8_AUC)
        & _num(primary.positive_seasons).ge(MIN_POSITIVE_SEASONS)
    )
    standalone_mask = (
        _num(primary.direct_ypt_mae_gain_vs_frozen).ge(MIN_STANDALONE_YPT_MAE_GAIN)
        & _num(primary.nonworse_mae_seasons_vs_frozen).ge(MIN_STANDALONE_NONWORSE_SEASONS)
    )
    signal_supported = bool(signal_mask.any())
    standalone_supported = bool(standalone_mask.any())

    best_signal = None
    if len(primary):
        best = primary.sort_values(["spearman_actual_ypt","high8_auc"], ascending=False).iloc[0]
        best_signal = {
            "feature":str(best.feature), "spearman_actual_ypt":float(best.spearman_actual_ypt),
            "high8_auc":float(best.high8_auc), "high10_auc":float(best.high10_auc),
            "direct_ypt_mae":float(best.direct_ypt_mae), "direct_ypt_mae_gain_vs_frozen":float(best.direct_ypt_mae_gain_vs_frozen),
            "positive_seasons":int(best.positive_seasons), "nonworse_mae_seasons_vs_frozen":int(best.nonworse_mae_seasons_vs_frozen),
        }

    result = {
        "diagnostic":"RB_R13_RECEIVING_EFFICIENCY_IDENTITY_V1",
        "disposition":"RB_R13_EFFICIENCY_HISTORY_SIGNAL_SUPPORTED_DIAGNOSTIC_ONLY" if signal_supported else "RB_R13_EFFICIENCY_HISTORY_SIGNAL_NOT_SUPPORTED_DIAGNOSTIC_ONLY",
        "efficiency_history_signal_supported":signal_supported,
        "standalone_ypt_replacement_supported":standalone_supported,
        "r9_status":"RB_R9_RECEIVING_IDENTITY_SHRINKAGE_OOS_PASS_UNCHANGED",
        "r12_status":"RB_R12_STATE_GATED_MODERN_STABILITY_FAIL_DIAGNOSTIC_ONLY_UNCHANGED",
        "frozen_ypt_direct_mae_3plus_targets":frozen_mae,
        "best_primary_signal":best_signal,
        "thresholds":{
            "min_primary_spearman":MIN_PRIMARY_SPEARMAN, "min_primary_high8_auc":MIN_PRIMARY_HIGH8_AUC,
            "min_positive_seasons":MIN_POSITIVE_SEASONS, "min_standalone_ypt_mae_gain":MIN_STANDALONE_YPT_MAE_GAIN,
            "min_standalone_nonworse_seasons":MIN_STANDALONE_NONWORSE_SEASONS,
        },
        "strict_prior_audit":audit,
        "sportsbook_inputs_added":0, "production_parameters_changed":0,
        "governance_note":"2023-2025 are research-visible diagnostic seasons; this cannot promote an efficiency model. A supported signal only authorizes a separately frozen candidate/OOS design.",
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir / "rb_r13_efficiency_casebook.csv", index=False)
    summary.to_csv(a.out_dir / "rb_r13_efficiency_feature_summary.csv", index=False)
    primary.to_csv(a.out_dir / "rb_r13_primary_ypt_summary.csv", index=False)
    (a.out_dir / "rb_r13_efficiency_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("\n=== primary YPT signals ===\n", primary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
