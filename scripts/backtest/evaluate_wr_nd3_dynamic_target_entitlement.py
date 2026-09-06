#!/usr/bin/env python3
"""WR-ND3 frozen dynamic target-entitlement diagnostic.

Diagnostic only. This script must be executed with PYTHONPATH pointed at the
exact M38 checkout so all canonical football components come from
b98518d97b3038f471aee9ae3201009b2c70bb29.

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
ELIGIBLE_REC_POS = WR_POS | {"TE", "RB", "FB"}
EXPECTED_M38_REC_N = 4647
EXPECTED_M38_REC_MAE = 17.099904733366
EXPECTED_TARGET_MAE = 2.076010
TARGET_MAE_TOL = 0.01

SIGNALS = {
    "RECENCY_ACCEL_2V8": {"column": "recency_accel_2v8", "mode": "quartile"},
    "HIGHER_WR_ABSENT_COUNT": {"column": "higher_wr_absent_count", "mode": "positive"},
    "VACATED_SHARE_ABOVE_PLAYER": {"column": "vacated_share_above_player", "mode": "positive"},
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
    with patch.object(
        simulation_rules,
        "load_model_contexts",
        return_value=(bundle.teams, bundle.players),
    ):
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
    probs = probs / probs.sum()
    return probs[:-1]


def _m38_target_rows(metrics: pd.DataFrame, week: int) -> pd.DataFrame:
    rows: list[dict] = []
    for (event_id, team), g0 in metrics.groupby(["event_id", "team"], dropna=False):
        g = g0.reset_index(drop=True).copy()
        raw = np.array(
            [
                _num(
                    r.get(
                        "rules_tgt_share",
                        r.get("bayes_tgt_share", r.get("target_share", r.get("tgt_share", 0.0))),
                    ),
                    0.0,
                )
                for _, r in g.iterrows()
            ],
            dtype=float,
        )
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
            rows.append(
                {
                    "season": int(r.get("season", 2025)),
                    "week": int(week),
                    "event_id": str(event_id),
                    "team": canon_team(r.get("team")),
                    "opponent": canon_team(r.get("opponent")),
                    "player": r.get("player", ""),
                    "player_clean_key": str(r.get("player_clean_key", "")),
                    "position": pos,
                    "pregame_role": str(r.get("role", "") or ""),
                    "m38_wr_rank": rank,
                    "m38_wr_role": f"WR{rank}" if rank is not None and rank <= 3 else "WR4+",
                    "raw_rules_tgt_share": float(raw[j]),
                    "m38_sharpened_tgt_share": float(sharpened[j]),
                    "allocator_probability": float(probs[j]),
                    "team_expected_targets": team_expected_targets,
                    "pred_targets": team_expected_targets * float(probs[j]),
                    "rules_injury_redistribution": int(
                        _num(r.get("rules_injury_redistribution"), 0.0) > 0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _normalize_history(hist: pd.DataFrame) -> pd.DataFrame:
    h = hist.copy()
    h.columns = [str(c).strip().lower() for c in h.columns]
    h["season"] = _series(h, "season")
    h["week"] = _series(h, "week")
    h["team"] = h.get("team", "").map(canon_team)
    h["player_clean_key"] = h.get("player_clean_key", h.get("player", "")).map(cp._key)
    h["position"] = h.get("position", "").fillna("").astype(str).str.upper().str.strip()
    h["targets"] = _series(h, "targets", 0.0).fillna(0.0)
    if "team_targets" in h.columns:
        h["team_targets"] = _series(h, "team_targets")
    else:
        h["team_targets"] = h.groupby(["season", "week", "team"])["targets"].transform("sum")
    if "tgt_share_game" in h.columns:
        share = _series(h, "tgt_share_game")
    else:
        share = pd.Series(np.nan, index=h.index, dtype=float)
    computed = np.where(h["team_targets"].gt(0), h["targets"] / h["team_targets"], np.nan)
    h["tgt_share_game"] = share.where(share.notna(), computed)
    return h


def _current_universe(universe: pd.DataFrame) -> pd.DataFrame:
    u = universe.copy()
    u.columns = [str(c).strip().lower() for c in u.columns]
    u["team"] = u.get("team", "").map(canon_team)
    u["player_clean_key"] = u.get("player_clean_key", u.get("player", "")).map(cp._key)
    u["position"] = u.get("position", "").fillna("").astype(str).str.upper().str.strip()
    if "role" not in u.columns:
        u["role"] = ""
    return u


def _dynamic_features(hist: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    h = _normalize_history(hist)
    u = _current_universe(universe)
    rows: list[dict] = []

    for team, team_u in u.groupby("team", dropna=False):
        team = canon_team(team)
        th = h.loc[h["team"].eq(team)].sort_values(["season", "week"]).copy()
        current_keys = set(team_u["player_clean_key"].astype(str))
        current_wrs = team_u.loc[team_u["position"].isin(WR_POS)].copy()
        if current_wrs.empty:
            continue

        recent_games = (
            th[["season", "week"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["season", "week"])
            .tail(4)
        )
        if recent_games.empty:
            recent4 = th.iloc[0:0].copy()
        else:
            recent4 = th.merge(recent_games, on=["season", "week"], how="inner")

        if recent4.empty:
            team_denom = np.nan
            share4: dict[str, float] = {}
            pos4: dict[str, str] = {}
        else:
            den = (
                recent4.groupby(["season", "week"], as_index=False)["team_targets"]
                .max()["team_targets"]
                .sum()
            )
            team_denom = float(den) if pd.notna(den) and float(den) > 0 else np.nan
            passcatch = recent4.loc[recent4["position"].isin(ELIGIBLE_REC_POS)].copy()
            if np.isfinite(team_denom):
                sums = passcatch.groupby("player_clean_key")["targets"].sum()
                share4 = {str(k): float(v) / team_denom for k, v in sums.items()}
            else:
                share4 = {}
            latest = passcatch.sort_values(["season", "week"]).drop_duplicates("player_clean_key", keep="last")
            pos4 = dict(zip(latest["player_clean_key"].astype(str), latest["position"].astype(str)))

        absent = {k: v for k, v in share4.items() if k not in current_keys}
        vacated_all = float(sum(absent.values()))
        vacated_wr = float(sum(v for k, v in absent.items() if pos4.get(k, "") in WR_POS))

        for _, cur in current_wrs.iterrows():
            key = str(cur["player_clean_key"])
            ph = th.loc[th["player_clean_key"].eq(key)].sort_values(["season", "week"])
            last8 = ph.tail(8)
            last2 = ph.tail(2)
            if last8.empty or last2.empty:
                accel = np.nan
                raw_delta = np.nan
            else:
                accel = float(last2["tgt_share_game"].mean() - last8["tgt_share_game"].mean())
                raw_delta = float(last2["targets"].mean() - last8["targets"].mean())

            pshare4 = share4.get(key, np.nan)
            if np.isfinite(pshare4):
                higher_wr = [
                    v
                    for k, v in absent.items()
                    if pos4.get(k, "") in WR_POS and float(v) > float(pshare4)
                ]
                vacated_above = [
                    v for _, v in absent.items() if float(v) > float(pshare4)
                ]
                higher_count = float(len(higher_wr))
                vacated_above_player = float(sum(vacated_above))
            else:
                higher_count = np.nan
                vacated_above_player = np.nan

            rows.append(
                {
                    "team": team,
                    "player_clean_key": key,
                    "recency_accel_2v8": accel,
                    "recent_raw_targets_delta_2v8": raw_delta,
                    "prior4_team_target_share": pshare4,
                    "higher_wr_absent_count": higher_count,
                    "vacated_share_above_player": vacated_above_player,
                    "vacated_all_passcatcher_share": vacated_all,
                    "vacated_wr_share": vacated_wr,
                    "pregame_role_nonempty": int(bool(str(cur.get("role", "") or "").strip())),
                }
            )
    return pd.DataFrame(rows)


def _parent_m38_check(predictions: pd.DataFrame) -> dict:
    p = predictions.copy()
    p.columns = [str(c).strip().lower() for c in p.columns]
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


def _thresholds(frame: pd.DataFrame, signal_col: str, mode: str) -> dict:
    v = pd.to_numeric(frame[signal_col], errors="coerce").dropna()
    if v.empty:
        return {"mode": mode, "low": np.nan, "high": np.nan}
    if mode == "quartile":
        return {
            "mode": mode,
            "low": float(v.quantile(0.25)),
            "high": float(v.quantile(0.75)),
        }
    return {"mode": mode, "low": 0.0, "high": 0.0}


def _masks(frame: pd.DataFrame, signal_col: str, threshold: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
    v = pd.to_numeric(frame[signal_col], errors="coerce")
    valid = v.notna()
    if threshold["mode"] == "quartile":
        if not np.isfinite(threshold["low"]) or not np.isfinite(threshold["high"]) or threshold["high"] <= threshold["low"]:
            return valid & False, valid & False, valid
        high = valid & v.ge(float(threshold["high"]))
        low = valid & v.le(float(threshold["low"]))
    else:
        high = valid & v.gt(0.0)
        low = valid & v.eq(0.0)
    return high, low, valid


def _gap(frame: pd.DataFrame, signal_col: str, threshold: dict) -> float:
    high, low, _ = _masks(frame, signal_col, threshold)
    if int(high.sum()) == 0 or int(low.sum()) == 0:
        return np.nan
    return float(frame.loc[high, "allocation_residual"].mean() - frame.loc[low, "allocation_residual"].mean())


def _score_signal(casebook: pd.DataFrame, name: str, spec: dict) -> tuple[dict, list[dict]]:
    col = spec["column"]
    threshold = _thresholds(casebook, col, spec["mode"])
    high, low, valid = _masks(casebook, col, threshold)
    coverage = float(valid.mean()) if len(casebook) else 0.0
    corr_frame = casebook.loc[valid, [col, "allocation_residual"]].dropna()
    spearman = (
        float(corr_frame[col].corr(corr_frame["allocation_residual"], method="spearman"))
        if len(corr_frame) > 2 and corr_frame[col].nunique() > 1
        else np.nan
    )
    overall_gap = _gap(casebook, col, threshold)
    tail = casebook["entitlement_miss_tail"].astype(bool)
    overall_tail_rate = float(tail.loc[valid].mean()) if int(valid.sum()) else np.nan
    high_tail_rate = float(tail.loc[high].mean()) if int(high.sum()) else np.nan
    tail_enrichment = (
        float(high_tail_rate / overall_tail_rate)
        if np.isfinite(high_tail_rate) and np.isfinite(overall_tail_rate) and overall_tail_rate > 0
        else np.nan
    )

    slices = {
        "ALL_WR": casebook.index == casebook.index,
        "W2_18": casebook["week"].ge(2),
        "W13_18": casebook["week"].ge(13),
        "WR1": casebook["m38_wr_rank"].eq(1),
        "WR2": casebook["m38_wr_rank"].eq(2),
        "WR3": casebook["m38_wr_rank"].eq(3),
        "WR4+": casebook["m38_wr_rank"].ge(4),
    }
    slice_rows: list[dict] = []
    gaps: dict[str, float] = {}
    for slice_name, mask in slices.items():
        sub = casebook.loc[mask].copy()
        gap = _gap(sub, col, threshold)
        gaps[slice_name] = gap
        h, l, v = _masks(sub, col, threshold)
        slice_rows.append(
            {
                "signal": name,
                "slice": slice_name,
                "n": int(len(sub)),
                "coverage": float(v.mean()) if len(sub) else np.nan,
                "high_n": int(h.sum()),
                "low_n": int(l.sum()),
                "allocation_residual_gap": gap,
                "raw_target_error_high_mean": float(sub.loc[h, "raw_target_error"].mean()) if int(h.sum()) else np.nan,
                "raw_target_error_low_mean": float(sub.loc[l, "raw_target_error"].mean()) if int(l.sum()) else np.nan,
            }
        )

    role_positive_count = int(sum(np.isfinite(gaps[r]) and gaps[r] > 0 for r in ("WR1", "WR2", "WR3")))
    passed = bool(
        coverage >= 0.70
        and np.isfinite(spearman)
        and spearman >= 0.08
        and np.isfinite(overall_gap)
        and overall_gap >= 0.025
        and np.isfinite(gaps["W2_18"])
        and gaps["W2_18"] > 0
        and np.isfinite(gaps["W13_18"])
        and gaps["W13_18"] > 0
        and role_positive_count >= 2
        and np.isfinite(tail_enrichment)
        and tail_enrichment >= 1.20
    )
    summary = {
        "signal": name,
        "column": col,
        "mode": spec["mode"],
        "coverage": coverage,
        "spearman_allocation_residual": spearman,
        "high_low_allocation_residual_gap": overall_gap,
        "tail_enrichment": tail_enrichment,
        "w2_18_gap": gaps["W2_18"],
        "w13_18_gap": gaps["W13_18"],
        "wr1_gap": gaps["WR1"],
        "wr2_gap": gaps["WR2"],
        "wr3_gap": gaps["WR3"],
        "wr1_wr2_wr3_positive_count": role_positive_count,
        "threshold_low": threshold["low"],
        "threshold_high": threshold["high"],
        "gate_passed": passed,
    }
    return summary, slice_rows


def main() -> int:
    q = argparse.ArgumentParser()
    q.add_argument("--season", type=int, default=2025)
    q.add_argument("--prior-season", type=int, default=2024)
    q.add_argument("--weeks", default="1-18")
    q.add_argument("--predictions", type=Path, required=True)
    q.add_argument("--player-logs", type=Path, required=True)
    q.add_argument("--team-weekly", type=Path, required=True)
    q.add_argument("--schedule", type=Path, required=True)
    q.add_argument("--universe-dir", type=Path, required=True)
    q.add_argument("--injuries", type=Path, required=True)
    q.add_argument("--weather", type=Path, required=True)
    q.add_argument("--out-dir", type=Path, default=Path("data/backtests/wr_nd3_dynamic_target_entitlement"))
    a = q.parse_args()

    predictions = _read(a.predictions, "M38 component predictions")
    parent_check = _parent_m38_check(predictions)
    logs = _read(a.player_logs, "player logs")
    team_weekly = _read(a.team_weekly, "team weekly")
    schedule = _read(a.schedule, "schedule")
    injuries = _optional(a.injuries)
    weather = _optional(a.weather)

    case_rows: list[pd.DataFrame] = []
    anomaly_rows: list[pd.DataFrame] = []
    route_audit_rows: list[dict] = []

    for week in _parse_weeks(a.weeks):
        universe_path = a.universe_dir / f"{a.season}_week_{week:02d}.csv"
        universe = _read(universe_path, f"pregame universe W{week:02d}")
        bundle = build_historical_context_bundle(
            player_logs=logs,
            team_weekly=team_weekly,
            pregame_universe=universe,
            schedule=schedule,
            season=int(a.season),
            week=int(week),
            prior_season=int(a.prior_season),
            injuries=_exact_week(injuries, a.season, week),
            weather=_exact_week(weather, a.season, week),
        )
        metrics = _prepared_metrics(bundle)
        pred = _m38_target_rows(metrics, week)
        dyn = _dynamic_features(bundle.player_history, universe)
        actual = _actual_target_rows(logs, a.season, week)
        x = pred.merge(dyn, on=["team", "player_clean_key"], how="left", validate="one_to_one")
        x = x.merge(actual, on=["team", "player_clean_key"], how="inner", validate="one_to_one")

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

        hist = _normalize_history(bundle.player_history)
        route_audit_rows.append(
            {
                "week": int(week),
                "history_rows": int(len(hist)),
                "routes_nonnull_rate": float(_series(hist, "routes").notna().mean()) if "routes" in hist.columns else 0.0,
                "route_rate_nonnull_rate": float(_series(hist, "route_rate_game").notna().mean()) if "route_rate_game" in hist.columns else 0.0,
                "pregame_role_nonempty_rate": float(_current_universe(universe)["role"].fillna("").astype(str).str.strip().ne("").mean()),
            }
        )
        print(f"[wr-nd3] W{week:02d} rows={len(x)} anomalies={len(bad)}")

    casebook = pd.concat(case_rows, ignore_index=True)
    if len(casebook) != 2130:
        raise RuntimeError(f"WR-ND3 casebook row drift: expected 2130, got {len(casebook)}")
    target_mae = float(casebook["raw_target_error"].abs().mean())
    if abs(target_mae - EXPECTED_TARGET_MAE) > TARGET_MAE_TOL:
        raise RuntimeError(
            f"M38 target-opportunity reconstruction drift: mae={target_mae:.12f}, expected≈{EXPECTED_TARGET_MAE:.6f}"
        )

    signal_rows: list[dict] = []
    slice_rows: list[dict] = []
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
        disposition = "MULTIPLE_DYNAMIC_ENTITLEMENT_SIGNALS"
    else:
        disposition = "NO_ACTIONABLE_DYNAMIC_ENTITLEMENT_SIGNAL"

    anomaly = pd.concat(anomaly_rows, ignore_index=True) if anomaly_rows else pd.DataFrame()
    context_audit = pd.DataFrame(route_audit_rows)
    result = {
        "migration": "WR-ND3",
        "season": int(a.season),
        "prior_season": int(a.prior_season),
        "weeks": [int(w) for w in _parse_weeks(a.weeks)],
        "evaluation_rows": int(len(casebook)),
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "m38_parent_check": parent_check,
        "target_reconstruction_mae": target_mae,
        "factorization_anomalies_excluded_from_evaluation": int(len(anomaly)),
        "source_constraints": {
            "route_participation_source_blocked": True,
            "mean_routes_nonnull_rate": float(context_audit["routes_nonnull_rate"].mean()),
            "mean_route_rate_nonnull_rate": float(context_audit["route_rate_nonnull_rate"].mean()),
            "mean_pregame_role_nonempty_rate": float(context_audit["pregame_role_nonempty_rate"].mean()),
        },
        "frozen_gate": {
            "coverage_min": 0.70,
            "spearman_min": 0.08,
            "allocation_gap_min": 0.025,
            "w2_18_positive": True,
            "w13_18_positive": True,
            "wr1_wr2_wr3_positive_min_count": 2,
            "tail_enrichment_min": 1.20,
        },
        "winners": winners,
        "disposition": disposition,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    casebook.to_csv(a.out_dir / "wr_nd3_player_casebook.csv", index=False)
    signal_summary.to_csv(a.out_dir / "wr_nd3_signal_summary.csv", index=False)
    slice_summary.to_csv(a.out_dir / "wr_nd3_slice_summary.csv", index=False)
    context_audit.to_csv(a.out_dir / "wr_nd3_context_audit.csv", index=False)
    anomaly.to_csv(a.out_dir / "wr_nd3_factorization_anomalies.csv", index=False)
    (a.out_dir / "wr_nd3_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[wr-nd3] signal summary")
    print(signal_summary.to_string(index=False))
    print("[wr-nd3] result")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
