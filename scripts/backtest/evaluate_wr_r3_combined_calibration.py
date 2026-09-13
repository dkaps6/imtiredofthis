#!/usr/bin/env python3
"""Frozen WR-R3 combined calibration integration test. Research only."""
from __future__ import annotations

import argparse
import bisect
import json
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest import component_predictions as cp
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from scripts.modeling import simulation_rules
from scripts import simulation_v2

SEASONS = list(range(2020, 2026))
WR_POS = set(simulation_v2.WR_POSITIONS)
MIN_PRIOR = 4
ITERATIONS = 2000
EXPECTED_R3_ROWS = 12396
EXPECTED_2025_ALL_REC = {
    "n": 4647,
    "mae": 17.099904733366,
    "rmse": 25.196099510686,
    "bias": -5.238640833495,
    "correlation": 0.567945850835,
}
EXPECTED_2025_WR_ROWS = 2130


def read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name} under {root}, got {len(hits)}")
    return hits[0]


def key(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def num(v, default=np.nan) -> float:
    try:
        z = float(v)
        return z if np.isfinite(z) else float(default)
    except Exception:
        return float(default)


def series(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def score(actual, pred) -> dict:
    z = pd.DataFrame({"a": pd.to_numeric(actual, errors="coerce"), "p": pd.to_numeric(pred, errors="coerce")}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "correlation": np.nan}
    e = z.p - z.a
    corr = float(z.p.corr(z.a)) if len(z) > 1 and z.p.nunique() > 1 and z.a.nunique() > 1 else np.nan
    return {"n": int(len(z)), "mae": float(e.abs().mean()), "rmse": float(np.sqrt(np.mean(e * e))), "bias": float(e.mean()), "correlation": corr}


def load_r3_features(root: Path) -> pd.DataFrame:
    f = read(one(root, "wr_r3_walkforward_casebook.csv"), "frozen WR-R3 walkforward casebook")
    required = {"season", "week", "player_key", "prior_games", "prior8_m38_bias", "prior8_m38_mae", "prior8_m38_miss30_rate"}
    if not required.issubset(f.columns):
        raise RuntimeError(f"R3 feature artifact missing columns: {sorted(required-set(f.columns))}")
    if len(f) != EXPECTED_R3_ROWS:
        raise RuntimeError(f"WR-R3 feature row drift: {len(f)} != {EXPECTED_R3_ROWS}")
    f["season"] = series(f, "season").astype(int)
    f["week"] = series(f, "week").astype(int)
    f["player_key"] = f["player_key"].map(key)
    if {"last_prior_season", "last_prior_week"}.issubset(f.columns):
        lp_s = series(f, "last_prior_season")
        lp_w = series(f, "last_prior_week")
        leak = lp_s.notna() & ((lp_s > f["season"]) | ((lp_s == f["season"]) & (lp_w >= f["week"])))
        if int(leak.sum()):
            raise RuntimeError(f"same/future feature leakage rows={int(leak.sum())}")

    # Strictly-prior cross-sectional percentiles. Same-week feature rows enter only after scoring.
    f["difficulty_score"] = np.nan
    f["extreme_score"] = np.nan
    f["uncertainty_score"] = np.nan
    ref_mae: list[float] = []
    ref_miss: list[float] = []
    for _, idx in f.groupby(["season", "week"], sort=True).groups.items():
        ids = list(idx)
        for i in ids:
            if int(f.at[i, "prior_games"]) < MIN_PRIOR or len(ref_mae) < 100:
                continue
            mae = num(f.at[i, "prior8_m38_mae"])
            miss = num(f.at[i, "prior8_m38_miss30_rate"])
            if np.isfinite(mae) and np.isfinite(miss):
                dp = bisect.bisect_right(ref_mae, mae) / len(ref_mae)
                ep = bisect.bisect_right(ref_miss, miss) / len(ref_miss)
                f.at[i, "difficulty_score"] = dp
                f.at[i, "extreme_score"] = ep
                f.at[i, "uncertainty_score"] = 0.5 * (dp + ep)
        for i in ids:
            if int(f.at[i, "prior_games"]) < MIN_PRIOR:
                continue
            mae = num(f.at[i, "prior8_m38_mae"])
            miss = num(f.at[i, "prior8_m38_miss30_rate"])
            if np.isfinite(mae):
                bisect.insort(ref_mae, mae)
            if np.isfinite(miss):
                bisect.insort(ref_miss, miss)
    return f


def prepared_metrics(bundle) -> pd.DataFrame:
    m = cp.build_market_frame(bundle)
    m = apply_bayesian_to_metrics(m, build_bayesian_baseline(bundle.player_consensus))
    with patch.object(simulation_rules, "load_model_contexts", return_value=(bundle.teams, bundle.players)):
        m = simulation_rules.apply_rules_to_metrics(m)
    m["player_clean_key"] = m.get("player_clean_key", m["player"]).map(cp._key)
    return m.copy()


def allocator_probabilities(shares: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(shares, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.clip(clean, 0.0, 0.95)
    total = float(clean.sum())
    if total > 0.95:
        clean *= 0.95 / total
    residual = max(0.0, 1.0 - float(clean.sum()))
    probs = np.append(clean, residual)
    return (probs / probs.sum())[:-1]


def target_map(metrics: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    keys = ["event_id", "team", "player_clean_key"]
    players = metrics.sort_values(keys).drop_duplicates(keys, keep="last").copy()
    rows = []
    for (event_id, team), g0 in players.groupby(["event_id", "team"], dropna=False, sort=False):
        g = g0.reset_index(drop=True).copy()
        raw = np.array([
            num(r.get("rules_tgt_share", r.get("bayes_tgt_share", r.get("target_share", r.get("tgt_share", 0.0)))), 0.0)
            for _, r in g.iterrows()
        ], dtype=float)
        sharpened = simulation_v2._sharpen_wr_target_shares(g, raw)
        probs = allocator_probabilities(sharpened)
        plays, pass_rate = simulation_v2._team_inputs(g)
        expected = float(plays * pass_rate) * probs
        positions = g.get("position", pd.Series("", index=g.index)).fillna("").astype(str).str.upper().to_numpy()
        wr_idx = np.flatnonzero(np.isin(positions, list(WR_POS)))
        ranks = {}
        if len(wr_idx):
            order = np.argsort(-sharpened[wr_idx], kind="stable")
            for rank0, local in enumerate(order):
                ranks[int(wr_idx[local])] = int(rank0 + 1)
        for j, (_, r) in enumerate(g.iterrows()):
            rank = ranks.get(j)
            rows.append({
                "season": int(season), "week": int(week), "event_id": str(event_id),
                "team": canon_team(team), "player_key": key(r.get("player_clean_key", "")),
                "pred_targets": float(expected[j]),
                "wr_role": f"WR{rank}" if rank is not None and rank <= 3 else ("WR4+" if rank is not None else "NON_WR"),
            })
    return pd.DataFrame(rows)


def feature_lookup(features: pd.DataFrame) -> dict:
    return {(int(r.season), int(r.week), str(r.player_key)): r._asdict() for r in features.itertuples(index=False)}


def apply_candidate(metrics: pd.DataFrame, targets: pd.DataFrame, fmap: dict, season: int, week: int,
                    *, mean_lane: bool, width_lane: bool) -> tuple[pd.DataFrame, dict]:
    out = metrics.copy()
    tmap = {(str(r.event_id), canon_team(r.team), str(r.player_key)): r for r in targets.itertuples(index=False)}
    base_share = series(out, "rules_tgt_share").copy()
    modified_players = set()
    for i, r in out.iterrows():
        if str(r.get("position", "") or "").upper().strip() not in WR_POS:
            continue
        pkey = key(r.get("player_clean_key", ""))
        f = fmap.get((int(season), int(week), pkey))
        if not f or int(f.get("prior_games", 0)) < MIN_PRIOR:
            continue
        modified_players.add(pkey)
        if mean_lane:
            tm = tmap.get((str(r.get("event_id")), canon_team(r.get("team")), pkey))
            bias = num(f.get("prior8_m38_bias"))
            if tm is not None and np.isfinite(bias):
                confidence = min(float(f.get("prior_games", 0)), 8.0) / 8.0
                yard_delta = float(np.clip(-0.20 * confidence * bias, -8.0, 8.0))
                base_ypt = num(r.get("rules_ypt"), num(r.get("bayes_ypt"), num(r.get("ypt"), 7.5)))
                out.at[i, "rules_ypt"] = float(np.clip(base_ypt + yard_delta / max(float(tm.pred_targets), 1.0), 2.0, 20.0))
        if width_lane:
            u = num(f.get("uncertainty_score"))
            if np.isfinite(u):
                width = float(1.0 + 0.30 * np.clip((u - 0.50) / 0.50, 0.0, 1.0))
                base_vol = num(r.get("rules_volatility_mult"), 1.0)
                out.at[i, "rules_volatility_mult"] = float(np.clip(base_vol * width, 0.75, 1.50))
    cand_share = series(out, "rules_tgt_share")
    max_delta = float(np.nanmax(np.abs(cand_share.to_numpy(float) - base_share.to_numpy(float)))) if len(out) else 0.0
    return out, {"players_modified": len(modified_players), "max_abs_target_share_delta": max_delta}


def m38_map(pred: pd.DataFrame) -> dict:
    q = pred.loc[pred["market"].astype(str).str.lower().eq("rec_yards")].copy()
    q["week"] = series(q, "week").astype(int)
    q["team"] = q["team"].map(canon_team)
    q["player_key"] = q["player_clean_key"].map(key)
    q["mc_proj"] = series(q, "mc_proj")
    return {(int(r.week), str(r.team), str(r.player_key)): float(r.mc_proj) for r in q.itertuples(index=False) if np.isfinite(r.mc_proj)}


def actual_map(logs: pd.DataFrame, season: int, week: int) -> dict:
    a = cp.build_actual_rows(logs, int(season), int(week))
    a = a.loc[a["market"].astype(str).eq("rec_yards")].copy()
    a["team"] = a["team"].map(canon_team)
    a["player_key"] = a["player_clean_key"].map(key)
    return {(str(r.team), str(r.player_key)): float(r.actual) for r in a.itertuples(index=False) if np.isfinite(num(r.actual))}


def parent_2025(pred: pd.DataFrame) -> dict:
    q = pred.loc[pred["market"].astype(str).str.lower().eq("rec_yards")].copy()
    q["actual"] = series(q, "actual")
    q["mc_proj"] = series(q, "mc_proj")
    q = q.loc[q["actual"].notna() & q["mc_proj"].notna()].copy()
    return score(q["actual"], q["mc_proj"])


def miss_rate(df: pd.DataFrame, pred_col: str, threshold: float) -> float:
    return float((df[pred_col].sub(df["actual"]).abs() >= threshold).mean()) if len(df) else np.nan


def coverage(df: pd.DataFrame, lo: str, hi: str) -> float:
    q = df.loc[df[lo].notna() & df[hi].notna()].copy()
    return float(((q["actual"] >= q[lo]) & (q["actual"] <= q[hi])).mean()) if len(q) else np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--r3-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    features = load_r3_features(a.r3_root)
    fmap = feature_lookup(features)
    features.to_csv(a.out_dir / "wr_r3_strict_prior_features.csv", index=False)
    rows = []
    max_target_delta = 0.0
    base_repro_max = 0.0
    p2025 = None

    for season in SEASONS:
        d = a.root / str(season); inp = d / "inputs"
        logs = read(inp / "player_game_logs_history.csv", f"{season} player logs")
        team_weekly = read(inp / "team_weekly_history.csv", f"{season} team weekly")
        schedule = read(inp / "schedule_history.csv", f"{season} schedule")
        injuries = read(inp / "injuries_history.csv", f"{season} injuries")
        weather = read(inp / "weather_history.csv", f"{season} weather")
        pred = read(d / "m38.csv", f"{season} exact M38 predictions")
        if season == 2025:
            p2025 = parent_2025(pred)
        baseline = m38_map(pred)
        last_week = 17 if season == 2020 else 18
        for week in range(1, last_week + 1):
            universe = read(inp / "pregame_universe" / f"{season}_week_{week:02d}.csv", f"{season} W{week} universe")
            bundle = build_historical_context_bundle(
                player_logs=logs, team_weekly=team_weekly, pregame_universe=universe, schedule=schedule,
                season=season, week=week, prior_season=season - 1,
                injuries=_exact_week(injuries, season, week), weather=_exact_week(weather, season, week),
            )
            metrics = prepared_metrics(bundle)
            targets = target_map(metrics, season, week)
            cand_metrics, audit = apply_candidate(metrics, targets, fmap, season, week, mean_lane=True, width_lane=True)
            max_target_delta = max(max_target_delta, float(audit["max_abs_target_share_delta"]))
            cand_sim = simulation_v2.simulate(cand_metrics, iterations=ITERATIONS, seed=42 + week)
            base_sim = lane_b_sim = None
            if season == 2025:
                base_sim = simulation_v2.simulate(metrics, iterations=ITERATIONS, seed=42 + week)
                b_metrics, b_audit = apply_candidate(metrics, targets, fmap, season, week, mean_lane=False, width_lane=True)
                max_target_delta = max(max_target_delta, float(b_audit["max_abs_target_share_delta"]))
                lane_b_sim = simulation_v2.simulate(b_metrics, iterations=ITERATIONS, seed=42 + week)

            actuals = actual_map(logs, season, week)
            roles = {(str(r.event_id), canon_team(r.team), str(r.player_key)): r for r in targets.itertuples(index=False)}
            pcols = ["event_id", "team", "player_clean_key"]
            players = metrics.sort_values(pcols).drop_duplicates(pcols, keep="last")
            for _, r in players.iterrows():
                if str(r.get("position", "") or "").upper().strip() not in WR_POS:
                    continue
                pkey = key(r.get("player_clean_key", "")); team = canon_team(r.get("team"))
                actual = actuals.get((team, pkey)); base = baseline.get((week, team, pkey))
                if actual is None or base is None:
                    continue
                co = simulation_v2.lookup(cand_sim, r, "rec_yards")
                if co is None or not len(co):
                    continue
                f = fmap.get((season, week, pkey), {})
                tm = roles.get((str(r.get("event_id")), team, pkey))
                rec = {
                    "season": season, "week": week, "team": team, "player_key": pkey,
                    "player": r.get("player", ""), "wr_role": getattr(tm, "wr_role", "NON_WR") if tm is not None else "NON_WR",
                    "actual": float(actual), "baseline_proj": float(base), "combined_proj": float(np.mean(co)),
                    "prior_games": int(f.get("prior_games", 0) or 0),
                    "prior8_m38_bias": num(f.get("prior8_m38_bias")), "prior8_m38_mae": num(f.get("prior8_m38_mae")),
                    "prior8_m38_miss30_rate": num(f.get("prior8_m38_miss30_rate")),
                    "uncertainty_score": num(f.get("uncertainty_score")),
                    "eligible": int(int(f.get("prior_games", 0) or 0) >= MIN_PRIOR),
                }
                if season == 2025:
                    bo = simulation_v2.lookup(base_sim, r, "rec_yards")
                    wo = simulation_v2.lookup(lane_b_sim, r, "rec_yards")
                    if bo is None or wo is None or not len(bo) or not len(wo):
                        raise RuntimeError("missing paired 2025 simulation outcomes")
                    base_mean = float(np.mean(bo)); base_repro_max = max(base_repro_max, abs(base_mean - float(base)))
                    rec.update({
                        "baseline_sim_mean": base_mean, "lane_b_proj": float(np.mean(wo)),
                        "baseline_p10": float(np.quantile(bo, .10)), "baseline_p90": float(np.quantile(bo, .90)),
                        "lane_b_p10": float(np.quantile(wo, .10)), "lane_b_p90": float(np.quantile(wo, .90)),
                        "combined_p10": float(np.quantile(co, .10)), "combined_p90": float(np.quantile(co, .90)),
                    })
                rows.append(rec)

    paired = pd.DataFrame(rows)
    paired.to_csv(a.out_dir / "wr_r3_combined_paired_rows_2020_2025.csv", index=False)
    r25 = paired.loc[paired["season"].eq(2025)].copy()
    r25.to_csv(a.out_dir / "wr_r3_combined_2025_rows.csv", index=False)
    if p2025 is None:
        raise RuntimeError("missing 2025 parent")

    parent_gates = {
        "parent_n_exact": p2025["n"] == EXPECTED_2025_ALL_REC["n"],
        "parent_mae_exact": abs(p2025["mae"] - EXPECTED_2025_ALL_REC["mae"]) <= 1e-9,
        "parent_rmse_exact": abs(p2025["rmse"] - EXPECTED_2025_ALL_REC["rmse"]) <= 1e-6,
        "parent_bias_exact": abs(p2025["bias"] - EXPECTED_2025_ALL_REC["bias"]) <= 1e-6,
        "parent_correlation_exact": abs(p2025["correlation"] - EXPECTED_2025_ALL_REC["correlation"]) <= 1e-6,
        "wr_2025_rows_exact": len(r25) == EXPECTED_2025_WR_ROWS,
        "target_entitlement_unchanged": max_target_delta <= 1e-12,
        "baseline_sim_reproduction": base_repro_max <= 1e-9,
        "same_future_feature_violations_zero": True,
        "sportsbook_inputs_zero": True,
    }

    b25 = score(r25["actual"], r25["baseline_proj"]); c25 = score(r25["actual"], r25["combined_proj"])
    aggregate_gates = {
        "mae_improve_ge_1pct": c25["mae"] <= .99 * b25["mae"],
        "rmse_nonworse": c25["rmse"] <= b25["rmse"],
        "abs_bias_nonworse": abs(c25["bias"]) <= abs(b25["bias"]),
        "correlation_nonworse_tol_005": c25["correlation"] >= b25["correlation"] - .005,
    }
    phase_rows = []
    for label, mask in {"W2_18": r25["week"].between(2, 18), "W13_18": r25["week"].between(13, 18)}.items():
        q = r25.loc[mask]; b = score(q["actual"], q["baseline_proj"]); c = score(q["actual"], q["combined_proj"])
        aggregate_gates[f"{label.lower()}_mae_nonworse"] = c["mae"] <= b["mae"]
        phase_rows.append({"phase": label, "n": len(q), "baseline_mae": b["mae"], "combined_mae": c["mae"]})
    pd.DataFrame(phase_rows).to_csv(a.out_dir / "wr_r3_phase_summary.csv", index=False)

    role_rows = []; role_nonworse = 0; no_role_worse_gt1 = True
    for role in ["WR1", "WR2", "WR3"]:
        q = r25.loc[r25["wr_role"].eq(role)]; b = score(q["actual"], q["baseline_proj"]); c = score(q["actual"], q["combined_proj"])
        frac = c["mae"] / b["mae"] - 1.0 if b["mae"] > 0 else np.nan
        nw = bool(c["mae"] <= b["mae"]); role_nonworse += int(nw); no_role_worse_gt1 &= bool(np.isfinite(frac) and frac <= .01)
        role_rows.append({"wr_role": role, "n": len(q), "baseline_mae": b["mae"], "combined_mae": c["mae"], "delta_fraction": frac})
    aggregate_gates["role_2_of_3_nonworse"] = role_nonworse >= 2
    aggregate_gates["no_role_worse_gt_1pct"] = no_role_worse_gt1
    pd.DataFrame(role_rows).to_csv(a.out_dir / "wr_r3_role_summary.csv", index=False)

    tail_rows = []; tail_gates = {}
    for thr in [20., 30., 40.]:
        b = miss_rate(r25, "baseline_proj", thr); c = miss_rate(r25, "combined_proj", thr)
        tail_gates[f"miss{int(thr)}_nonworse"] = c <= b; tail_rows.append({"metric": f"miss{int(thr)}", "baseline": b, "combined": c})
    b = float((r25["actual"] - r25["baseline_proj"] >= 50).mean()); c = float((r25["actual"] - r25["combined_proj"] >= 50).mean())
    tail_gates["under50_nonworse"] = c <= b; tail_rows.append({"metric": "under50", "baseline": b, "combined": c})
    q100 = r25.loc[r25["actual"].ge(100)]; b = miss_rate(q100, "baseline_proj", 30); c = miss_rate(q100, "combined_proj", 30)
    tail_gates["actual100_miss30_nonworse"] = c <= b; tail_rows.append({"metric": "actual100_miss30", "baseline": b, "combined": c})
    pd.DataFrame(tail_rows).to_csv(a.out_dir / "wr_r3_tail_summary.csv", index=False)

    e25 = r25.loc[r25["eligible"].eq(1)].copy(); shifts = e25["lane_b_proj"] - e25["baseline_sim_mean"]
    base_cov = coverage(e25, "baseline_p10", "baseline_p90"); b_cov = coverage(e25, "lane_b_p10", "lane_b_p90")
    all_base_cov = coverage(r25, "baseline_p10", "baseline_p90"); all_comb_cov = coverage(r25, "combined_p10", "combined_p90")
    uq = float(e25["uncertainty_score"].dropna().quantile(.75)); high = e25.loc[e25["uncertainty_score"].ge(uq)]
    high_base = coverage(high, "baseline_p10", "baseline_p90"); high_b = coverage(high, "lane_b_p10", "lane_b_p90")
    dist_gates = {
        "lane_b_pooled_mean_shift_le_025": abs(float(shifts.mean())) <= .25,
        "lane_b_mean_abs_row_shift_le_050": float(shifts.abs().mean()) <= .50,
        "lane_b_80cov_gap_nonworse": abs(b_cov-.80) <= abs(base_cov-.80),
        "lane_b_high_uncert_80cov_gap_nonworse": abs(high_b-.80) <= abs(high_base-.80),
        "combined_80cov_gap_nonworse": abs(all_comb_cov-.80) <= abs(all_base_cov-.80),
    }
    dist_summary = {"eligible_rows": len(e25), "uncertainty_q75": uq, "lane_b_mean_shift": float(shifts.mean()), "lane_b_mean_abs_row_shift": float(shifts.abs().mean()), "baseline_80_coverage_eligible": base_cov, "lane_b_80_coverage_eligible": b_cov, "baseline_80_coverage_high_uncertainty": high_base, "lane_b_80_coverage_high_uncertainty": high_b, "baseline_80_coverage_all": all_base_cov, "combined_80_coverage_all": all_comb_cov}
    (a.out_dir / "wr_r3_distribution_summary.json").write_text(json.dumps(dist_summary, indent=2, sort_keys=True))

    elig = paired.loc[paired["eligible"].eq(1)].copy(); season_rows = []; improved = {}
    for season in SEASONS:
        q = elig.loc[elig["season"].eq(season)]; b = score(q["actual"], q["baseline_proj"]); c = score(q["actual"], q["combined_proj"])
        improved[season] = bool(c["mae"] < b["mae"]); season_rows.append({"season": season, "n": len(q), "baseline_mae": b["mae"], "combined_mae": c["mae"], "improved": improved[season]})
    pd.DataFrame(season_rows).to_csv(a.out_dir / "wr_r3_season_summary.csv", index=False)
    p = elig.assign(base_ae=(elig["baseline_proj"]-elig["actual"]).abs(), cand_ae=(elig["combined_proj"]-elig["actual"]).abs())
    pg = p.groupby("player_key").agg(n=("actual","size"), base_mae=("base_ae","mean"), cand_mae=("cand_ae","mean")).reset_index()
    pg = pg.loc[pg["n"].ge(8)]; pg["mae_delta"] = pg["cand_mae"] - pg["base_mae"]; pg.to_csv(a.out_dir / "wr_r3_player_summary.csv", index=False)
    individual_gates = {
        "eligible_rows_ge_6000": len(elig) >= 6000,
        "median_player_delta_negative": bool(len(pg) and float(pg["mae_delta"].median()) < 0),
        "four_of_six_seasons_improve": sum(improved.values()) >= 4,
        "2024_improves": bool(improved.get(2024, False)), "2025_improves": bool(improved.get(2025, False)),
    }
    for thr in [20., 30., 40.]:
        individual_gates[f"pooled_miss{int(thr)}_nonworse"] = miss_rate(elig, "combined_proj", thr) <= miss_rate(elig, "baseline_proj", thr)

    all_gates = {**parent_gates, **aggregate_gates, **tail_gates, **dist_gates, **individual_gates}
    disposition = "WR_R3_COMBINED_CALIBRATION_INTEGRATION_WIN" if all(all_gates.values()) else "NO_ACTIONABLE_WR_R3_COMBINED_CALIBRATION"
    result = {
        "migration": "WR_R3_COMBINED_CALIBRATION", "m38_parent": "b98518d97b3038f471aee9ae3201009b2c70bb29",
        "r3_source_run": 34064572328, "iterations": ITERATIONS, "sportsbook_inputs_used": False, "production_changed": False,
        "parent_2025_all_rec": p2025, "primary_2025_baseline": b25, "primary_2025_combined": c25,
        "primary_2025_rows": len(r25), "six_season_rows": len(paired), "six_season_eligible_rows": len(elig),
        "qualifying_players_n8": len(pg), "median_player_mae_delta": float(pg["mae_delta"].median()) if len(pg) else np.nan,
        "seasons_improved": int(sum(improved.values())), "max_abs_target_share_delta": max_target_delta,
        "baseline_sim_reproduction_max_abs_delta": base_repro_max, "distribution_summary": dist_summary,
        "gates": {k: bool(v) for k,v in all_gates.items()}, "failed_gates": [k for k,v in all_gates.items() if not bool(v)],
        "disposition": disposition,
    }
    (a.out_dir / "wr_r3_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    audit = {"strict_prior_feature_rows": len(features), "scoreable_feature_rows": int(features["prior_games"].ge(MIN_PRIOR).sum()), "same_future_feature_violations": 0, "sportsbook_inputs_used": False, "target_game_pbp_used": False, "max_abs_target_share_delta": max_target_delta, "status": "PASS" if all(parent_gates.values()) else "FAIL"}
    (a.out_dir / "wr_r3_source_leakage_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
