#!/usr/bin/env python3
"""Audit individual-player pregame opportunity allocation for 2026 Weeks 1-4.

Frozen by docs/research/PLAYER_OPPORTUNITY_ALLOCATION_AUDIT_V1_CONTRACT.md.

This is diagnostic only. It records the opportunities already allocated by the
canonical simulator; it does not change allocation probabilities, RNG routing,
football means, or production behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import build_actual_rows, build_mc_predictions
from scripts.backtest.historical_context import (
    assert_no_future_rows,
    build_historical_context_bundle,
)
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.modeling.te_r5p_entitlement_adapter_v1 import apply_te_r5p_entitlement
from scripts.modeling.wr_r15_entitlement_adapter_v1 import apply_wr_r15_entitlement
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate
import scripts.simulation_v2 as simv2

SEASON = 2026
PRIOR_SEASON = 2025
WEEKS = (1, 2, 3, 4)
TOL = 1e-10
POSITION_FAMILIES = {"QB", "RB", "FB", "WR", "TE"}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _clean(v) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"", "nan", "none", "<na>"} else s


def _pos(v) -> str:
    p = str(v or "").upper().strip()
    if p in {"HB", "TB"} or p.startswith("RB"):
        return "RB"
    if p.startswith("FB"):
        return "FB"
    if p.startswith("QB"):
        return "QB"
    if p.startswith("WR") or p in {"LWR", "RWR", "SWR"}:
        return "WR"
    if p.startswith("TE"):
        return "TE"
    return p


def _num(v, default=np.nan) -> float:
    try:
        z = float(v)
        return z if np.isfinite(z) else float(default)
    except Exception:
        return float(default)


def _probability_transform(shares: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:
    clean = np.nan_to_num(np.asarray(shares, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.clip(clean, 0.0, 0.95)
    raw_sum = float(clean.sum())
    used = clean.copy()
    if raw_sum > 0.95:
        used *= 0.95 / raw_sum
    residual = max(0.0, 1.0 - float(used.sum()))
    probs = np.append(used, residual)
    probs = probs / probs.sum()
    return clean, raw_sum, probs[:-1], float(probs[-1])


def _prepare_specialists(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    player_cols = ["event_id", "team", "player_clean_key"]
    players = metrics.sort_values(player_cols).drop_duplicates(player_cols, keep="last").copy()
    if players.duplicated(player_cols).any():
        raise RuntimeError("opportunity audit player universe is not unique")

    base, _ = materialize_target_entitlement(players)
    te_final, _, te_audit = apply_te_r5p_entitlement(base)
    final, _, wr_audit = apply_wr_r15_entitlement(te_final)

    before = base.groupby(["event_id", "team"])["entitlement_tgt_share"].sum()
    after = final.groupby(["event_id", "team"])["entitlement_tgt_share"].sum()
    max_gap = float((after - before).abs().max()) if len(before) else 0.0
    if max_gap > TOL:
        raise RuntimeError(f"specialists changed team target mass max_gap={max_gap}")

    return final, {
        "te": te_audit,
        "wr": wr_audit,
        "max_team_entitlement_gap": max_gap,
    }


def _trace_explicit_simulation(
    final: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> tuple[object, pd.DataFrame]:
    """Run exact explicit-entitlement sim while observing target/carry allocator.

    The wrapped allocator invokes the original exactly once and returns its exact
    output unchanged. Call order mirrors canonical simulation_v2: targets then
    carries for each team within each game.
    """
    frame = final.copy()
    frame["player_clean_key"] = frame.apply(simv2._player_key, axis=1)
    game_key = "event_id" if "event_id" in frame.columns and frame["event_id"].notna().any() else None
    if game_key is None:
        frame["_game_key"] = frame.apply(
            lambda r: "|".join(sorted([str(r.get("team", "")), str(r.get("opponent", ""))])),
            axis=1,
        )
        game_key = "_game_key"

    player_cols = [game_key, "team", "player_clean_key"]
    players = frame.sort_values(player_cols).drop_duplicates(player_cols, keep="last")
    call_plan: list[tuple[str, str, str, pd.DataFrame]] = []
    for game, game_df in players.groupby(game_key, dropna=False):
        for team, team_df in game_df.groupby("team", dropna=False):
            if pd.isna(team) or not str(team).strip():
                continue
            call_plan.append((str(game), str(team), "targets", team_df.copy()))
            call_plan.append((str(game), str(team), "carries", team_df.copy()))

    original = simv2._allocate_counts
    rows: list[dict] = []
    call_index = 0

    def wrapped(rng, totals, shares):
        nonlocal call_index
        if call_index >= len(call_plan):
            raise RuntimeError("allocation trace received more calls than canonical call plan")
        game, team, kind, team_df = call_plan[call_index]
        call_index += 1

        out = original(rng, totals, shares)
        clean, raw_sum, probs, residual_prob = _probability_transform(np.asarray(shares, dtype=float))
        if len(team_df) != len(probs) or out.shape[1] != len(team_df):
            raise RuntimeError(
                f"allocation trace/player mismatch game={game} team={team} kind={kind}"
            )

        total_mean = float(np.mean(totals)) if len(totals) else np.nan
        allocated_means = out.mean(axis=0).astype(float) if len(out) else np.zeros(len(team_df))
        if np.any(out.sum(axis=1) > np.asarray(totals, dtype=int)):
            raise RuntimeError(f"{kind} allocation exceeded simulated team total")

        for j, (_, r) in enumerate(team_df.iterrows()):
            rows.append({
                "event_id": game,
                "team": canon_team(team),
                "player": r.get("player", ""),
                "player_clean_key": simv2._player_key(r),
                "position": r.get("position", ""),
                "opportunity_type": kind,
                "raw_player_share": float(clean[j]),
                "raw_team_share_sum": raw_sum,
                "final_player_probability": float(probs[j]),
                "residual_probability": residual_prob,
                "predicted_team_opportunity_mean": total_mean,
                "predicted_opportunities": float(allocated_means[j]),
                "expected_opportunities_from_probability": total_mean * float(probs[j]),
                "allocation_sampling_delta": float(allocated_means[j] - total_mean * float(probs[j])),
            })
        return out

    with patch.object(simv2, "_allocate_counts", side_effect=wrapped):
        result = explicit_simulate(final, iterations=int(iterations), seed=int(seed))

    if call_index != len(call_plan):
        raise RuntimeError(
            f"allocation trace expected {len(call_plan)} calls but observed {call_index}"
        )
    trace = pd.DataFrame(rows)
    if trace.empty:
        raise RuntimeError("allocation trace is empty")
    return result, trace


def _actual_map(player_logs: pd.DataFrame, week: int) -> dict[tuple[str, str, str], float]:
    a = build_actual_rows(player_logs, SEASON, int(week))
    out: dict[tuple[str, str, str], float] = {}
    for r in a.itertuples(index=False):
        key = (canon_team(r.team), str(r.player_clean_key), str(r.market))
        opp = _num(r.actual_opportunities, default=np.nan)
        if np.isfinite(opp):
            out[key] = float(opp)
    return out


def _actual_for(
    amap: dict[tuple[str, str, str], float],
    *,
    team: str,
    player_key: str,
    opportunity_type: str,
) -> tuple[float, str]:
    market = {
        "pass_attempts": "pass_yards",
        "targets": "rec_yards",
        "carries": "rush_yards",
    }[opportunity_type]
    key = (canon_team(team), str(player_key), market)
    if key in amap:
        return float(amap[key]), "NFLVERSE_WEEKLY_STATS"
    return 0.0, "PREGAME_UNIVERSE_NO_MATCHING_WEEKLY_OPPORTUNITY_ROW_ZERO"


def _opportunity_bin(position: str, opportunity_type: str, actual: float) -> str:
    x = float(actual)
    if x <= 0:
        return "ZERO"
    if opportunity_type == "pass_attempts":
        if x <= 20: return "01_20"
        if x <= 30: return "21_30"
        if x <= 40: return "31_40"
        return "41_PLUS"
    if opportunity_type == "carries":
        if x <= 3: return "01_03"
        if x <= 8: return "04_08"
        if x <= 14: return "09_14"
        return "15_PLUS"
    if x <= 2: return "01_02"
    if x <= 5: return "03_05"
    if x <= 8: return "06_08"
    return "09_PLUS"


def _attach_point_errors(rows: pd.DataFrame, point_path: Path) -> pd.DataFrame:
    point = _read(point_path, "frozen all-player point scoreboard")
    keep = [
        "week", "event_id", "team", "player_clean_key", "position_family",
        "market", "error", "absolute_error",
    ]
    missing = [c for c in keep if c not in point.columns]
    if missing:
        raise RuntimeError(f"point scoreboard missing required columns: {missing}")
    p = point[keep].copy()

    yards_parts = []
    count_parts = []
    for opp_type, market in [
        ("pass_attempts", "pass_yards"),
        ("carries", "rush_yards"),
        ("targets", "rec_yards"),
    ]:
        q = p.loc[p["market"].eq(market)].copy()
        q["opportunity_type"] = opp_type
        q = q.rename(columns={
            "error": "linked_yards_error",
            "absolute_error": "linked_yards_absolute_error",
        })
        yards_parts.append(q.drop(columns=["market"]))
    for opp_type, market in [("targets", "receptions")]:
        q = p.loc[p["market"].eq(market)].copy()
        q["opportunity_type"] = opp_type
        q = q.rename(columns={
            "error": "linked_count_error",
            "absolute_error": "linked_count_absolute_error",
        })
        count_parts.append(q.drop(columns=["market"]))

    y = pd.concat(yards_parts, ignore_index=True, sort=False)
    if y.duplicated(["week", "event_id", "team", "player_clean_key", "opportunity_type"]).any():
        raise RuntimeError("duplicate linked yard-error identities")
    out = rows.merge(
        y,
        on=["week", "event_id", "team", "player_clean_key", "position_family", "opportunity_type"],
        how="left",
        validate="one_to_one",
    )
    if count_parts:
        c = pd.concat(count_parts, ignore_index=True, sort=False)
        if c.duplicated(["week", "event_id", "team", "player_clean_key", "opportunity_type"]).any():
            raise RuntimeError("duplicate linked count-error identities")
        out = out.merge(
            c,
            on=["week", "event_id", "team", "player_clean_key", "position_family", "opportunity_type"],
            how="left",
            validate="one_to_one",
        )
    return out


def _summary_group(g: pd.DataFrame) -> dict:
    pred = pd.to_numeric(g["predicted_opportunities"], errors="coerce")
    actual = pd.to_numeric(g["actual_opportunities"], errors="coerce")
    err = pred - actual
    zero = actual.eq(0)
    nonzero = ~zero
    pearson = float(pred.corr(actual)) if pred.nunique() > 1 and actual.nunique() > 1 else np.nan
    spearman = float(pred.rank(method="average").corr(actual.rank(method="average"))) if pred.nunique() > 1 and actual.nunique() > 1 else np.nan
    linked_y = pd.to_numeric(g.get("linked_yards_error"), errors="coerce")
    linked_c = pd.to_numeric(g.get("linked_count_error"), errors="coerce")
    return {
        "rows": int(len(g)),
        "mae": float(err.abs().mean()),
        "median_absolute_error": float(err.abs().median()),
        "signed_bias": float(err.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "prediction_actual_pearson": pearson,
        "prediction_actual_spearman": spearman,
        "actual_zero_rows": int(zero.sum()),
        "actual_zero_rate": float(zero.mean()),
        "mean_prediction_when_actual_zero": float(pred.loc[zero].mean()) if zero.any() else np.nan,
        "median_prediction_when_actual_zero": float(pred.loc[zero].median()) if zero.any() else np.nan,
        "mean_prediction_when_actual_nonzero": float(pred.loc[nonzero].mean()) if nonzero.any() else np.nan,
        "opportunity_error_vs_linked_yards_error_pearson": (
            float(err.corr(linked_y)) if linked_y.notna().sum() > 2 and err.nunique() > 1 else np.nan
        ),
        "opportunity_error_vs_linked_count_error_pearson": (
            float(err.corr(linked_c)) if linked_c.notna().sum() > 2 and err.nunique() > 1 else np.nan
        ),
    }


def _build_summary(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (pos, opp), g in rows.groupby(["position_family", "opportunity_type"], dropna=False):
        out.append({
            "position_family": pos,
            "opportunity_type": opp,
            "actual_opportunity_bin": "ALL",
            **_summary_group(g),
        })
        for label, b in g.groupby("actual_opportunity_bin", dropna=False):
            out.append({
                "position_family": pos,
                "opportunity_type": opp,
                "actual_opportunity_bin": str(label),
                **_summary_group(b),
            })
    return pd.DataFrame(out)


def _zero_state_audit(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    # Fixed semantic diagnostic thresholds are reported together. No threshold
    # is selected, optimized, or promoted.
    for (pos, opp), g in rows.groupby(["position_family", "opportunity_type"], dropna=False):
        pred = pd.to_numeric(g["predicted_opportunities"], errors="coerce")
        actual_zero = pd.to_numeric(g["actual_opportunities"], errors="coerce").eq(0)
        rec = {
            "position_family": pos,
            "opportunity_type": opp,
            "rows": int(len(g)),
            "actual_zero_rows": int(actual_zero.sum()),
            "actual_zero_rate": float(actual_zero.mean()),
            "pred_zero_q10": float(pred.loc[actual_zero].quantile(0.10)) if actual_zero.any() else np.nan,
            "pred_zero_q50": float(pred.loc[actual_zero].quantile(0.50)) if actual_zero.any() else np.nan,
            "pred_zero_q90": float(pred.loc[actual_zero].quantile(0.90)) if actual_zero.any() else np.nan,
            "pred_nonzero_q10": float(pred.loc[~actual_zero].quantile(0.10)) if (~actual_zero).any() else np.nan,
            "pred_nonzero_q50": float(pred.loc[~actual_zero].quantile(0.50)) if (~actual_zero).any() else np.nan,
            "pred_nonzero_q90": float(pred.loc[~actual_zero].quantile(0.90)) if (~actual_zero).any() else np.nan,
        }
        for threshold in (0.5, 1.0):
            pred_zero = pred < threshold
            tp = int((pred_zero & actual_zero).sum())
            fp = int((pred_zero & ~actual_zero).sum())
            fn = int((~pred_zero & actual_zero).sum())
            rec[f"zero_precision_pred_lt_{str(threshold).replace('.', '_')}"] = (
                float(tp / (tp + fp)) if (tp + fp) else np.nan
            )
            rec[f"zero_recall_pred_lt_{str(threshold).replace('.', '_')}"] = (
                float(tp / (tp + fn)) if (tp + fn) else np.nan
            )
        out.append(rec)
    return pd.DataFrame(out)


def run_audit(
    *,
    player_logs_path: Path,
    team_weekly_path: Path,
    schedule_path: Path,
    universe_dir: Path,
    point_scoreboard_path: Path,
    out_dir: Path,
    iterations: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    player_logs = _read(player_logs_path, "player logs")
    team_weekly = _read(team_weekly_path, "team weekly history")
    schedule = _read(schedule_path, "schedule history")

    all_rows = []
    specialist_audits = {}
    for week in WEEKS:
        universe = _read(universe_dir / f"{SEASON}_week_{week:02d}.csv", f"pregame universe W{week}")
        bundle = build_historical_context_bundle(
            player_logs=player_logs,
            team_weekly=team_weekly,
            pregame_universe=universe,
            schedule=schedule,
            season=SEASON,
            week=week,
            prior_season=PRIOR_SEASON,
        )
        assert_no_future_rows(bundle.player_history, SEASON, week, f"W{week} player_history")
        assert_no_future_rows(bundle.team_history, SEASON, week, f"W{week} team_history")

        metrics = build_mc_predictions(bundle, iterations=int(iterations), seed=42 + week)
        metrics["position_family"] = metrics["position"].map(_pos)
        final, spec_audit = _prepare_specialists(metrics)
        _, trace = _trace_explicit_simulation(final, iterations=int(iterations), seed=42 + week)
        trace["week"] = week
        trace["season"] = SEASON
        trace["position_family"] = trace["position"].map(_pos)
        trace = trace.loc[trace["position_family"].isin(POSITION_FAMILIES)].copy()

        # Attach opponent from the frozen metrics identity map.
        ident = (
            metrics[["event_id", "team", "opponent", "player_clean_key", "player", "position_family"]]
            .sort_values(["event_id", "team", "player_clean_key"])
            .drop_duplicates(["event_id", "team", "player_clean_key"], keep="last")
        )
        trace = trace.merge(
            ident[["event_id", "team", "opponent", "player_clean_key"]],
            on=["event_id", "team", "player_clean_key"],
            how="left",
            validate="many_to_one",
        )

        amap = _actual_map(player_logs, week)
        rows = []

        # QB attempts: existing football-only authority.
        qbs = metrics.loc[
            metrics["position_family"].eq("QB") & metrics["market"].eq("pass_yards")
        ].copy()
        if qbs.empty:
            raise RuntimeError(f"W{week}: zero QB pass_yards rows")
        for _, r in qbs.iterrows():
            pred = _num(r.get("mc_expected_pass_attempts"))
            if not np.isfinite(pred):
                raise RuntimeError(f"W{week}: missing mc_expected_pass_attempts for {r.get('player')}")
            actual, src = _actual_for(
                amap,
                team=r["team"],
                player_key=r["player_clean_key"],
                opportunity_type="pass_attempts",
            )
            rows.append({
                "season": SEASON,
                "week": week,
                "event_id": str(r["event_id"]),
                "team": canon_team(r["team"]),
                "opponent": canon_team(r["opponent"]),
                "player": r["player"],
                "player_clean_key": r["player_clean_key"],
                "position_family": "QB",
                "opportunity_type": "pass_attempts",
                "predicted_opportunities": float(pred),
                "actual_opportunities": actual,
                "actual_source": src,
                "raw_player_share": _num(r.get("mc_qb_pass_att_share")),
                "raw_team_share_sum": np.nan,
                "final_player_probability": _num(r.get("mc_qb_pass_att_share")),
                "residual_probability": np.nan,
                "predicted_team_opportunity_mean": _num(r.get("mc_team_expected_pass_attempts")),
                "expected_opportunities_from_probability": float(pred),
                "allocation_sampling_delta": 0.0,
                "allocation_source": "MC_EXPECTED_PASS_ATTEMPTS",
            })

        # RB/FB carries and RB/FB/WR/TE targets from exact explicit simulation.
        for _, r in trace.iterrows():
            pos = str(r["position_family"])
            kind = str(r["opportunity_type"])
            if kind == "carries":
                if pos not in {"RB", "FB"}:
                    continue
                opp_type = "carries"
            else:
                if pos not in {"RB", "FB", "WR", "TE"}:
                    continue
                opp_type = "targets"
            actual, src = _actual_for(
                amap,
                team=r["team"],
                player_key=r["player_clean_key"],
                opportunity_type=opp_type,
            )
            rows.append({
                "season": SEASON,
                "week": week,
                "event_id": str(r["event_id"]),
                "team": canon_team(r["team"]),
                "opponent": canon_team(r["opponent"]),
                "player": r["player"],
                "player_clean_key": r["player_clean_key"],
                "position_family": pos,
                "opportunity_type": opp_type,
                "predicted_opportunities": float(r["predicted_opportunities"]),
                "actual_opportunities": actual,
                "actual_source": src,
                "raw_player_share": float(r["raw_player_share"]),
                "raw_team_share_sum": float(r["raw_team_share_sum"]),
                "final_player_probability": float(r["final_player_probability"]),
                "residual_probability": float(r["residual_probability"]),
                "predicted_team_opportunity_mean": float(r["predicted_team_opportunity_mean"]),
                "expected_opportunities_from_probability": float(r["expected_opportunities_from_probability"]),
                "allocation_sampling_delta": float(r["allocation_sampling_delta"]),
                "allocation_source": "CANONICAL_EXPLICIT_ENTITLEMENT_MC",
            })

        wk = pd.DataFrame(rows)
        if wk.empty:
            raise RuntimeError(f"W{week}: zero opportunity audit rows")
        wk["opportunity_error"] = wk["predicted_opportunities"] - wk["actual_opportunities"]
        wk["absolute_opportunity_error"] = wk["opportunity_error"].abs()
        wk["actual_zero_opportunity"] = wk["actual_opportunities"].eq(0)
        wk["actual_opportunity_bin"] = [
            _opportunity_bin(p, o, a)
            for p, o, a in zip(
                wk["position_family"], wk["opportunity_type"], wk["actual_opportunities"]
            )
        ]
        wk["sportsbook_inputs_used_upstream"] = False
        wk["rb_week5_room_allocation_shadow_applied"] = False
        all_rows.append(wk)
        specialist_audits[str(week)] = spec_audit

        print(
            f"[player-opportunity-audit] W{week} rows={len(wk)} "
            f"QB={int(wk.position_family.eq('QB').sum())} "
            f"RB/FB={int(wk.position_family.isin(['RB','FB']).sum())} "
            f"WR={int(wk.position_family.eq('WR').sum())} "
            f"TE={int(wk.position_family.eq('TE').sum())}"
        )

    rows = pd.concat(all_rows, ignore_index=True, sort=False)
    key = ["season", "week", "event_id", "team", "player_clean_key", "opportunity_type"]
    if rows.duplicated(key).any():
        bad = rows.loc[rows.duplicated(key, keep=False), key].head(20)
        raise RuntimeError(f"duplicate opportunity identities:\n{bad.to_string(index=False)}")
    if rows["sportsbook_inputs_used_upstream"].any():
        raise RuntimeError("sportsbook fields entered opportunity audit")
    if rows["rb_week5_room_allocation_shadow_applied"].any():
        raise RuntimeError("Week-5 RB shadow leaked into W1-4 opportunity audit")
    if float(rows["allocation_sampling_delta"].abs().max()) > 0.25:
        # 5k multinomial samples should be much tighter than this. This is an
        # integrity smoke gate, not a football calibration threshold.
        raise RuntimeError(
            "unexpectedly large allocation sampling delta "
            f"max={rows['allocation_sampling_delta'].abs().max()}"
        )

    rows = _attach_point_errors(rows, point_scoreboard_path)
    summary = _build_summary(rows)
    zero = _zero_state_audit(rows)

    rows.sort_values(key).to_csv(out_dir / "player_opportunity_allocation_rows.csv", index=False)
    summary.to_csv(out_dir / "player_opportunity_allocation_summary.csv", index=False)
    zero.to_csv(out_dir / "player_opportunity_zero_state_audit.csv", index=False)

    all_summary = summary.loc[summary["actual_opportunity_bin"].eq("ALL")].copy()
    payload = {
        "version": "PLAYER_OPPORTUNITY_ALLOCATION_AUDIT_V1",
        "season": SEASON,
        "weeks": list(WEEKS),
        "iterations": int(iterations),
        "rows": int(len(rows)),
        "unique_player_weeks": int(
            rows[["week", "team", "player_clean_key"]].drop_duplicates().shape[0]
        ),
        "summary": all_summary.to_dict("records"),
        "zero_state": zero.to_dict("records"),
        "max_abs_allocation_sampling_delta": float(rows["allocation_sampling_delta"].abs().max()),
        "sportsbook_inputs_used_upstream": False,
        "paid_odds_api_used": False,
        "rb_week5_room_shadow_rows": int(rows["rb_week5_room_allocation_shadow_applied"].sum()),
        "automatic_promotion": False,
        "parameters_fit": 0,
        "specialist_audits": specialist_audits,
    }
    (out_dir / "player_opportunity_audit_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--player-logs", type=Path, required=True)
    p.add_argument("--team-weekly", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--universe-dir", type=Path, required=True)
    p.add_argument("--point-scoreboard", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=5000)
    a = p.parse_args()
    run_audit(
        player_logs_path=a.player_logs,
        team_weekly_path=a.team_weekly,
        schedule_path=a.schedule,
        universe_dir=a.universe_dir,
        point_scoreboard_path=a.point_scoreboard,
        out_dir=a.out_dir,
        iterations=a.iterations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
