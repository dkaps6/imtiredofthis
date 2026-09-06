#!/usr/bin/env python3
"""WR-ND5 frozen snap/depth target-entitlement diagnostic.

Diagnostic only. Execute with PYTHONPATH pointed at the exact M38 checkout so
canonical football components come from b98518d97b3038f471aee9ae3201009b2c70bb29.
No model is fit and no production projection is changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest import component_predictions as cp
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from scripts.modeling import simulation_rules
from scripts import simulation_v2

WR_POS = set(simulation_v2.WR_POSITIONS)
EXPECTED_M38_REC_N = 4647
EXPECTED_M38_REC_MAE = 17.099904733366
EXPECTED_WR_ROWS = 2130
EXPECTED_TARGET_MAE = 2.076010432545868
TARGET_MAE_TOL = 0.01

SIGNALS = {
    "SNAP_ACCEL_1V4": {"column": "snap_accel_1v4", "mode": "quartile"},
    "SNAP_LEVEL_PRIOR1": {"column": "snap_level_prior1", "mode": "quartile"},
    "DEPTH_TOP2_STATE": {"column": "depth_top2_state", "mode": "binary"},
    "DEPTH_RANK_PROMOTION": {"column": "depth_rank_promotion", "mode": "positive_vs_nonpositive"},
}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _optional(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
    x = pd.read_csv(path)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _to_pandas(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


def _lower(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _num(value, default=np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _series(frame: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _prepared_metrics(bundle) -> pd.DataFrame:
    m = cp.build_market_frame(bundle)
    m = apply_bayesian_to_metrics(m, build_bayesian_baseline(bundle.player_consensus))
    with patch.object(simulation_rules, "load_model_contexts", return_value=(bundle.teams, bundle.players)):
        m = simulation_rules.apply_rules_to_metrics(m)
    keys = ["event_id", "team", "player_clean_key"]
    return m.sort_values(keys).drop_duplicates(keys, keep="last").copy()


def _allocator_probabilities(shares: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(shares, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.clip(clean, 0.0, 0.95)
    total = float(clean.sum())
    if total > 0.95:
        clean *= 0.95 / total
    residual = max(0.0, 1.0 - float(clean.sum()))
    probs = np.append(clean, residual)
    probs /= probs.sum()
    return probs[:-1]


def _m38_target_rows(metrics: pd.DataFrame, week: int) -> pd.DataFrame:
    rows: list[dict] = []
    for (event_id, team), g0 in metrics.groupby(["event_id", "team"], dropna=False):
        g = g0.reset_index(drop=True).copy()
        raw = np.array([
            _num(r.get("rules_tgt_share", r.get("bayes_tgt_share", r.get("target_share", r.get("tgt_share", 0.0)))), 0.0)
            for _, r in g.iterrows()
        ], dtype=float)
        sharpened = simulation_v2._sharpen_wr_target_shares(g, raw)
        probs = _allocator_probabilities(sharpened)
        plays, pass_rate = simulation_v2._team_inputs(g)
        team_expected_targets = float(plays * pass_rate)
        positions = g.get("position", pd.Series("", index=g.index)).fillna("").astype(str).str.upper().to_numpy()
        wr_idx = np.flatnonzero(np.isin(positions, list(WR_POS)))
        rank_by_idx: dict[int, int] = {}
        if len(wr_idx):
            order = np.argsort(-sharpened[wr_idx], kind="stable")
            for rank0, local_idx in enumerate(order):
                rank_by_idx[int(wr_idx[local_idx])] = int(rank0 + 1)
        for j, (_, r) in enumerate(g.iterrows()):
            pos = str(r.get("position", "") or "").upper().strip()
            if pos not in WR_POS:
                continue
            rank = rank_by_idx.get(j)
            rows.append({
                "season": int(r.get("season", 2025)),
                "week": int(week),
                "event_id": str(event_id),
                "team": canon_team(r.get("team")),
                "opponent": canon_team(r.get("opponent")),
                "player": r.get("player", ""),
                "player_clean_key": str(r.get("player_clean_key", "")),
                "position": pos,
                "m38_wr_rank": rank,
                "m38_wr_role": f"WR{rank}" if rank is not None and rank <= 3 else "WR4+",
                "pred_targets": team_expected_targets * float(probs[j]),
            })
    return pd.DataFrame(rows)


def _parent_m38_check(predictions: pd.DataFrame) -> dict:
    p = predictions.copy()
    g = p.loc[p["market"].astype(str).eq("rec_yards")].copy()
    g["actual"] = _series(g, "actual")
    g["mc_proj"] = _series(g, "mc_proj")
    g = g.loc[g["actual"].notna() & g["mc_proj"].notna()].copy()
    err = g["mc_proj"] - g["actual"]
    out = {
        "n": int(len(g)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(err.mean()),
        "correlation": float(g["mc_proj"].corr(g["actual"])),
    }
    if out["n"] != EXPECTED_M38_REC_N or abs(out["mae"] - EXPECTED_M38_REC_MAE) > 1e-9:
        raise RuntimeError(f"exact M38 parent drift: {out}")
    return out


def _actual_target_rows(logs: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    actual = cp.build_actual_rows(logs, int(season), int(week))
    at = actual.loc[actual["market"].eq("receptions"), ["team", "player_clean_key", "actual_opportunities"]].rename(
        columns={"actual_opportunities": "actual_targets"}
    )
    ay = actual.loc[actual["market"].eq("rec_yards"), ["team", "player_clean_key", "actual"]].rename(
        columns={"actual": "actual_rec_yards"}
    )
    return at.merge(ay, on=["team", "player_clean_key"], how="left", validate="one_to_one")


def _player_id_rows(logs: pd.DataFrame) -> pd.DataFrame:
    x = logs.copy()
    x["season"] = _series(x, "season")
    x["week"] = _series(x, "week")
    x["team"] = x["team"].fillna("").map(canon_team)
    x["player_clean_key"] = x.get("player_clean_key", x.get("player", "")).map(cp._key)
    if "player_id" not in x.columns:
        raise RuntimeError("player logs missing canonical player_id for ND5 source joins")
    return x[["season", "week", "team", "player_clean_key", "player_id"]].drop_duplicates(
        ["season", "week", "team", "player_clean_key"], keep="last"
    )


def _player_map(players: pd.DataFrame) -> pd.DataFrame:
    p = _lower(players)
    gsis = "gsis_id" if "gsis_id" in p.columns else "player_id"
    pfr = "pfr_id" if "pfr_id" in p.columns else "pfr_player_id"
    if gsis not in p.columns or pfr not in p.columns:
        raise RuntimeError(f"player map missing GSIS/PFR columns: {list(p.columns)}")
    out = p[[gsis, pfr]].rename(columns={gsis: "player_id", pfr: "pfr_player_id"}).copy()
    out["player_id"] = out["player_id"].astype("string").str.strip()
    out["pfr_player_id"] = out["pfr_player_id"].astype("string").str.strip()
    return out.dropna().loc[lambda d: d["player_id"].ne("") & d["pfr_player_id"].ne("")].drop_duplicates("player_id", keep="last")


def _normalize_snaps(snaps: pd.DataFrame, pmap: pd.DataFrame) -> pd.DataFrame:
    s = _lower(snaps)
    required = {"season", "week", "team", "pfr_player_id", "offense_pct"}
    if not required.issubset(s.columns):
        raise RuntimeError(f"snap counts missing columns: {sorted(required - set(s.columns))}")
    s["season"] = _series(s, "season")
    s["week"] = _series(s, "week")
    s["team"] = s["team"].fillna("").map(canon_team)
    s["pfr_player_id"] = s["pfr_player_id"].astype("string").str.strip()
    s["offense_pct"] = _series(s, "offense_pct")
    if "game_type" in s.columns:
        reg = s["game_type"].fillna("").astype(str).str.upper().eq("REG")
        if reg.any():
            s = s.loc[reg].copy()
    inv = pmap[["player_id", "pfr_player_id"]].drop_duplicates("pfr_player_id", keep="last")
    return s.merge(inv, on="pfr_player_id", how="left", validate="many_to_one")


def _attach_snap_signals(casebook: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, r in casebook.iterrows():
        prior = snaps.loc[
            snaps["player_id"].astype(str).eq(str(r["player_id"]))
            & snaps["team"].eq(r["team"])
            & (
                snaps["season"].lt(float(r["season"]))
                | (snaps["season"].eq(float(r["season"])) & snaps["week"].lt(float(r["week"])))
            )
        ].sort_values(["season", "week"])
        pct = prior["offense_pct"].dropna()
        prior1 = float(pct.iloc[-1]) if len(pct) >= 1 else np.nan
        prior4 = float(pct.tail(4).mean()) if len(pct) >= 4 else np.nan
        rows.append({
            "snap_level_prior1": prior1,
            "snap_prior4_mean": prior4,
            "snap_accel_1v4": float(prior1 - prior4) if np.isfinite(prior1) and np.isfinite(prior4) else np.nan,
        })
    return pd.concat([casebook.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _game_dates(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    raw = nfl.load_schedules(int(season))
    s = _lower(_to_pandas(raw))
    if "game_type" in s.columns:
        reg = s["game_type"].fillna("").astype(str).str.upper().eq("REG")
        if reg.any():
            s = s.loc[reg].copy()
    date_col = next((c for c in ["gameday", "game_date", "date"] if c in s.columns), None)
    away_col = "away_team" if "away_team" in s.columns else "away"
    home_col = "home_team" if "home_team" in s.columns else "home"
    if date_col is None or away_col not in s.columns or home_col not in s.columns or "week" not in s.columns:
        raise RuntimeError(f"nflverse schedule missing ND5 date/team fields: {list(s.columns)}")
    s["game_date"] = pd.to_datetime(s[date_col], errors="coerce").dt.date
    s["week"] = _series(s, "week")
    rows = []
    for _, r in s.loc[s["game_date"].notna() & s["week"].notna()].iterrows():
        for team in [r[away_col], r[home_col]]:
            rows.append({"season": int(season), "week": int(r["week"]), "team": canon_team(team), "game_date": r["game_date"]})
    out = pd.DataFrame(rows).drop_duplicates(["season", "week", "team"])
    # Previous game date is strictly earlier than the current target game.
    out = out.sort_values(["team", "game_date"]).reset_index(drop=True)
    out["previous_game_date"] = out.groupby("team")["game_date"].shift(1)
    return out


def _normalize_depth(depth: pd.DataFrame) -> pd.DataFrame:
    d = _lower(depth)
    required = {"dt", "team", "gsis_id", "pos_rank"}
    if not required.issubset(d.columns):
        raise RuntimeError(f"depth chart missing ND5 fields: {sorted(required - set(d.columns))}")
    d["team"] = d["team"].fillna("").map(canon_team)
    d["gsis_id"] = d["gsis_id"].astype("string").str.strip()
    d["dt_parsed"] = pd.to_datetime(d["dt"], errors="coerce", utc=True)
    d["snapshot_date"] = d["dt_parsed"].dt.date
    d["pos_rank"] = _series(d, "pos_rank")
    d["pos_slot"] = _series(d, "pos_slot") if "pos_slot" in d.columns else np.nan
    return d.loc[d["dt_parsed"].notna() & d["team"].ne("")].copy()


def _depth_rank_at(depth: pd.DataFrame, team: str, player_id: str, cutoff_date) -> tuple[float, float, object]:
    if pd.isna(cutoff_date):
        return np.nan, np.nan, pd.NaT
    eligible = depth.loc[depth["team"].eq(team) & depth["snapshot_date"].lt(cutoff_date)].copy()
    if eligible.empty:
        return np.nan, np.nan, pd.NaT
    latest_dt = eligible["dt_parsed"].max()
    snap = eligible.loc[eligible["dt_parsed"].eq(latest_dt)]
    hit = snap.loc[snap["gsis_id"].astype(str).eq(str(player_id))]
    if hit.empty:
        return np.nan, np.nan, latest_dt
    rank = float(hit["pos_rank"].dropna().min()) if hit["pos_rank"].notna().any() else np.nan
    slot = float(hit["pos_slot"].dropna().min()) if hit["pos_slot"].notna().any() else np.nan
    return rank, slot, latest_dt


def _attach_depth_signals(casebook: pd.DataFrame, depth: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    x = casebook.merge(dates, on=["season", "week", "team"], how="left", validate="many_to_one")
    if x["game_date"].isna().any():
        bad = x.loc[x["game_date"].isna(), ["season", "week", "team"]].drop_duplicates().head(20)
        raise RuntimeError(f"ND5 missing target game dates:\n{bad.to_string(index=False)}")
    rows = []
    for _, r in x.iterrows():
        cur_rank, cur_slot, cur_dt = _depth_rank_at(depth, r["team"], r["player_id"], r["game_date"])
        prev_rank, prev_slot, prev_dt = _depth_rank_at(depth, r["team"], r["player_id"], r["previous_game_date"])
        top2 = 1.0 if np.isfinite(cur_rank) and cur_rank <= 2 else 0.0 if np.isfinite(cur_rank) and cur_rank >= 3 else np.nan
        promotion = float(prev_rank - cur_rank) if np.isfinite(prev_rank) and np.isfinite(cur_rank) else np.nan
        rows.append({
            "depth_current_rank": cur_rank,
            "depth_current_slot": cur_slot,
            "depth_previous_rank": prev_rank,
            "depth_previous_slot": prev_slot,
            "depth_top2_state": top2,
            "depth_rank_promotion": promotion,
            "depth_current_snapshot_dt": cur_dt,
            "depth_previous_snapshot_dt": prev_dt,
        })
    out = pd.concat([x.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    cur_date = pd.to_datetime(out["depth_current_snapshot_dt"], errors="coerce", utc=True).dt.date
    prev_date = pd.to_datetime(out["depth_previous_snapshot_dt"], errors="coerce", utc=True).dt.date
    out["current_depth_timestamp_violation"] = cur_date.notna() & cur_date.ge(out["game_date"])
    out["previous_depth_timestamp_violation"] = prev_date.notna() & out["previous_game_date"].notna() & prev_date.ge(out["previous_game_date"])
    if int(out["current_depth_timestamp_violation"].sum()) or int(out["previous_depth_timestamp_violation"].sum()):
        raise RuntimeError("ND5 depth timestamp leakage detected")
    return out


def _thresholds(frame: pd.DataFrame, col: str, mode: str) -> dict:
    v = pd.to_numeric(frame[col], errors="coerce").dropna()
    if v.empty:
        return {"mode": mode, "low": np.nan, "high": np.nan}
    if mode == "quartile":
        return {"mode": mode, "low": float(v.quantile(0.25)), "high": float(v.quantile(0.75))}
    return {"mode": mode, "low": 0.0, "high": 0.0}


def _masks(frame: pd.DataFrame, col: str, threshold: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
    v = pd.to_numeric(frame[col], errors="coerce")
    valid = v.notna()
    mode = threshold["mode"]
    if mode == "quartile":
        if not np.isfinite(threshold["low"]) or not np.isfinite(threshold["high"]) or threshold["high"] <= threshold["low"]:
            return valid & False, valid & False, valid
        return valid & v.ge(float(threshold["high"])), valid & v.le(float(threshold["low"])), valid
    if mode == "binary":
        return valid & v.eq(1.0), valid & v.eq(0.0), valid
    if mode == "positive_vs_nonpositive":
        return valid & v.gt(0.0), valid & v.le(0.0), valid
    raise RuntimeError(f"unknown ND5 threshold mode: {mode}")


def _gap(frame: pd.DataFrame, col: str, threshold: dict, outcome: str = "allocation_residual") -> float:
    high, low, _ = _masks(frame, col, threshold)
    if int(high.sum()) == 0 or int(low.sum()) == 0:
        return np.nan
    return float(frame.loc[high, outcome].mean() - frame.loc[low, outcome].mean())


def _score_signal(casebook: pd.DataFrame, name: str, spec: dict) -> tuple[dict, list[dict]]:
    col = spec["column"]
    threshold = _thresholds(casebook, col, spec["mode"])
    high, low, valid = _masks(casebook, col, threshold)
    coverage = float(valid.mean()) if len(casebook) else 0.0
    corr = casebook.loc[valid, [col, "allocation_residual"]].dropna()
    spearman = float(corr[col].corr(corr["allocation_residual"], method="spearman")) if len(corr) > 2 and corr[col].nunique() > 1 else np.nan
    overall_gap = _gap(casebook, col, threshold)
    target_gap = _gap(casebook, col, threshold, "raw_target_error")
    tail = casebook["entitlement_miss_tail"].astype(bool)
    overall_tail_rate = float(tail.loc[valid].mean()) if int(valid.sum()) else np.nan
    high_tail_rate = float(tail.loc[high].mean()) if int(high.sum()) else np.nan
    tail_enrichment = float(high_tail_rate / overall_tail_rate) if np.isfinite(high_tail_rate) and np.isfinite(overall_tail_rate) and overall_tail_rate > 0 else np.nan
    slices = {
        "ALL_WR": pd.Series(True, index=casebook.index),
        "W2_18": casebook["week"].ge(2),
        "W13_18": casebook["week"].ge(13),
        "WR1": casebook["m38_wr_rank"].eq(1),
        "WR2": casebook["m38_wr_rank"].eq(2),
        "WR3": casebook["m38_wr_rank"].eq(3),
        "WR4+": casebook["m38_wr_rank"].ge(4),
    }
    slice_rows = []
    gaps = {}
    for slice_name, mask in slices.items():
        sub = casebook.loc[mask].copy()
        h, l, v = _masks(sub, col, threshold)
        gap = _gap(sub, col, threshold)
        gaps[slice_name] = gap
        slice_rows.append({
            "signal": name,
            "slice": slice_name,
            "n": int(len(sub)),
            "coverage": float(v.mean()) if len(sub) else np.nan,
            "high_n": int(h.sum()),
            "low_n": int(l.sum()),
            "allocation_residual_gap": gap,
            "raw_target_error_gap": _gap(sub, col, threshold, "raw_target_error"),
        })
    positive_count = int(sum(np.isfinite(gaps[r]) and gaps[r] > 0 for r in ("WR1", "WR2", "WR3")))
    passed = bool(
        coverage >= 0.85
        and np.isfinite(spearman) and spearman >= 0.08
        and np.isfinite(overall_gap) and overall_gap >= 0.025
        and np.isfinite(gaps["W2_18"]) and gaps["W2_18"] > 0
        and np.isfinite(gaps["W13_18"]) and gaps["W13_18"] > 0
        and positive_count >= 2
        and np.isfinite(tail_enrichment) and tail_enrichment >= 1.20
    )
    return {
        "signal": name,
        "column": col,
        "mode": spec["mode"],
        "coverage": coverage,
        "spearman_allocation_residual": spearman,
        "high_low_allocation_residual_gap": overall_gap,
        "high_low_raw_target_error_gap": target_gap,
        "tail_enrichment": tail_enrichment,
        "w2_18_gap": gaps["W2_18"],
        "w13_18_gap": gaps["W13_18"],
        "wr1_gap": gaps["WR1"],
        "wr2_gap": gaps["WR2"],
        "wr3_gap": gaps["WR3"],
        "wr1_wr2_wr3_positive_count": positive_count,
        "threshold_low": threshold["low"],
        "threshold_high": threshold["high"],
        "gate_passed": passed,
    }, slice_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--prior-season", type=int, default=2024)
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--team-weekly", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--universe-dir", type=Path, required=True)
    ap.add_argument("--injuries", type=Path, required=True)
    ap.add_argument("--weather", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/wr_nd5_snap_depth_entitlement"))
    a = ap.parse_args()

    predictions = _read(a.predictions, "M38 component predictions")
    parent_check = _parent_m38_check(predictions)
    logs = _read(a.player_logs, "player logs")
    team_weekly = _read(a.team_weekly, "team weekly")
    schedule = _read(a.schedule, "schedule")
    injuries = _optional(a.injuries)
    weather = _optional(a.weather)
    ids = _player_id_rows(logs)

    case_rows = []
    anomaly_rows = []
    for week in _parse_weeks(a.weeks):
        universe = _read(a.universe_dir / f"{a.season}_week_{week:02d}.csv", f"pregame universe W{week:02d}")
        bundle = build_historical_context_bundle(
            player_logs=logs, team_weekly=team_weekly, pregame_universe=universe, schedule=schedule,
            season=int(a.season), week=int(week), prior_season=int(a.prior_season),
            injuries=_exact_week(injuries, a.season, week), weather=_exact_week(weather, a.season, week),
        )
        pred = _m38_target_rows(_prepared_metrics(bundle), week)
        actual = _actual_target_rows(logs, a.season, week)
        x = pred.merge(actual, on=["team", "player_clean_key"], how="inner", validate="one_to_one")
        bad = x.loc[x["actual_targets"].le(0) & x["actual_rec_yards"].abs().gt(1e-9)].copy()
        if not bad.empty:
            bad["exclusion_reason"] = "NONZERO_REC_YARDS_WITH_ZERO_RECORDED_TARGETS"
            anomaly_rows.append(bad)
            x = x.drop(index=bad.index).copy()
        x["pred_wr_target_mass"] = x.groupby(["season", "week", "team"])["pred_targets"].transform("sum")
        x["actual_wr_targets"] = x.groupby(["season", "week", "team"])["actual_targets"].transform("sum")
        x["pred_wr_share"] = np.where(x["pred_wr_target_mass"].gt(0), x["pred_targets"] / x["pred_wr_target_mass"], np.nan)
        x["actual_wr_share"] = np.where(x["actual_wr_targets"].gt(0), x["actual_targets"] / x["actual_wr_targets"], np.nan)
        x["allocation_residual"] = x["actual_wr_share"] - x["pred_wr_share"]
        x["raw_target_error"] = x["actual_targets"] - x["pred_targets"]
        x["entitlement_miss_tail"] = x["actual_targets"].ge(10) & x["raw_target_error"].ge(3)
        case_rows.append(x)
        print(f"[wr-nd5] W{week:02d} target rows={len(x)} anomalies={len(bad)}")

    casebook = pd.concat(case_rows, ignore_index=True)
    if len(casebook) != EXPECTED_WR_ROWS:
        raise RuntimeError(f"WR-ND5 casebook drift: expected {EXPECTED_WR_ROWS}, got {len(casebook)}")
    target_mae = float(casebook["raw_target_error"].abs().mean())
    if abs(target_mae - EXPECTED_TARGET_MAE) > TARGET_MAE_TOL:
        raise RuntimeError(f"WR-ND5 target reconstruction drift: {target_mae:.12f}")
    casebook = casebook.merge(ids, on=["season", "week", "team", "player_clean_key"], how="left", validate="one_to_one")
    if casebook["player_id"].isna().any():
        bad = casebook.loc[casebook["player_id"].isna(), ["season", "week", "team", "player", "player_clean_key"]].head(20)
        raise RuntimeError(f"WR-ND5 missing player IDs:\n{bad.to_string(index=False)}")

    import nflreadpy as nfl

    pmap = _player_map(_to_pandas(nfl.load_players()))
    snaps = _normalize_snaps(_to_pandas(nfl.load_snap_counts(seasons=[2024, 2025])), pmap)
    casebook = _attach_snap_signals(casebook, snaps)
    depth = _normalize_depth(_to_pandas(nfl.load_depth_charts(seasons=[2025])))
    dates = _game_dates(int(a.season))
    casebook = _attach_depth_signals(casebook, depth, dates)

    signal_rows = []
    slice_rows = []
    for name, spec in SIGNALS.items():
        summary, slices = _score_signal(casebook, name, spec)
        signal_rows.append(summary)
        slice_rows.extend(slices)
    signal_summary = pd.DataFrame(signal_rows)
    slice_summary = pd.DataFrame(slice_rows)
    winners = signal_summary.loc[signal_summary["gate_passed"].astype(bool), "signal"].astype(str).tolist()
    if len(winners) == 1:
        disposition = f"{winners[0]}_ACTIONABLE"
    elif len(winners) >= 2:
        disposition = "MULTIPLE_ROLE_ENTITLEMENT_SIGNALS"
    else:
        disposition = "NO_ACTIONABLE_SNAP_DEPTH_ENTITLEMENT_SIGNAL"

    anomaly = pd.concat(anomaly_rows, ignore_index=True) if anomaly_rows else pd.DataFrame()
    source_audit = {
        "snap_level_prior1_coverage": float(casebook["snap_level_prior1"].notna().mean()),
        "snap_accel_1v4_coverage": float(casebook["snap_accel_1v4"].notna().mean()),
        "depth_top2_state_coverage": float(casebook["depth_top2_state"].notna().mean()),
        "depth_rank_promotion_coverage": float(casebook["depth_rank_promotion"].notna().mean()),
        "current_depth_timestamp_violations": int(casebook["current_depth_timestamp_violation"].sum()),
        "previous_depth_timestamp_violations": int(casebook["previous_depth_timestamp_violation"].sum()),
    }
    result = {
        "migration": "WR-ND5",
        "season": int(a.season),
        "prior_season": int(a.prior_season),
        "weeks": [int(w) for w in _parse_weeks(a.weeks)],
        "evaluation_rows": int(len(casebook)),
        "factorization_anomalies_excluded": int(len(anomaly)),
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "postseason_participation_used": False,
        "m38_parent_check": parent_check,
        "target_reconstruction_mae": target_mae,
        "source_audit": source_audit,
        "frozen_gate": {
            "coverage_min": 0.85,
            "spearman_min": 0.08,
            "allocation_gap_min": 0.025,
            "tail_enrichment_min": 1.20,
            "w2_18_positive": True,
            "w13_18_positive": True,
            "wr1_wr2_wr3_positive_min_count": 2,
        },
        "winners": winners,
        "disposition": disposition,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    casebook.to_csv(a.out_dir / "wr_nd5_casebook.csv", index=False)
    signal_summary.to_csv(a.out_dir / "wr_nd5_signal_summary.csv", index=False)
    slice_summary.to_csv(a.out_dir / "wr_nd5_slice_summary.csv", index=False)
    anomaly.to_csv(a.out_dir / "wr_nd5_factorization_anomalies.csv", index=False)
    (a.out_dir / "wr_nd5_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[wr-nd5] signal summary")
    print(signal_summary.to_string(index=False))
    print("[wr-nd5] result")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
