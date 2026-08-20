#!/usr/bin/env python3
"""Migration 27: surgical trace of canonical MC carry-allocation probabilities.

Diagnostic-only. The canonical simulator is executed unchanged while its private
_allocate_counts helper is wrapped so we can observe the exact rush totals,
raw shares, post-cap probabilities, residual bucket, and realized multinomial
means used for each team/player. This identifies whether the carry collapse
occurs before, inside, or after multinomial allocation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.backtest import component_predictions as cp
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.simulation_v2 import simulate as canonical_simulate
import scripts.simulation_v2 as simv2


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    return pd.read_csv(path)


def _optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _selected_players(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    frame["player_clean_key"] = frame.apply(simv2._player_key, axis=1)
    game_key = "event_id" if "event_id" in frame.columns and frame["event_id"].notna().any() else None
    if game_key is None:
        frame["_game_key"] = frame.apply(lambda r: "|".join(sorted([str(r.get("team", "")), str(r.get("opponent", ""))])), axis=1)
        game_key = "_game_key"
    cols = [game_key, "team", "player_clean_key"]
    return frame.sort_values(cols).drop_duplicates(cols, keep="last"), game_key


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


def _run_with_trace(metrics: pd.DataFrame, *, iterations: int, seed: int):
    selected, game_key = _selected_players(metrics)
    call_plan = []
    for game, game_df in selected.groupby(game_key, dropna=False):
        for team, team_df in game_df.groupby("team", dropna=False):
            if pd.isna(team) or not str(team).strip():
                continue
            call_plan.append((str(game), str(team), "targets", team_df.copy()))
            call_plan.append((str(game), str(team), "carries", team_df.copy()))

    original = simv2._allocate_counts
    traces = []
    call_index = 0

    def wrapped(rng, totals, shares):
        nonlocal call_index
        if call_index >= len(call_plan):
            raise RuntimeError("allocation trace received more calls than expected")
        game, team, kind, team_df = call_plan[call_index]
        call_index += 1
        out = original(rng, totals, shares)
        if kind == "carries":
            clean, raw_sum, final_probs, residual_prob = _probability_transform(shares)
            total_mean = float(np.mean(totals)) if len(totals) else np.nan
            if len(team_df) != len(final_probs) or out.shape[1] != len(team_df):
                raise RuntimeError(f"allocation/player alignment mismatch for {game} {team}")
            for j, (_, row) in enumerate(team_df.iterrows()):
                expected = total_mean * float(final_probs[j])
                realized = float(out[:, j].mean())
                traces.append({
                    "event_id": game,
                    "team": team,
                    "player_clean_key": simv2._player_key(row),
                    "position": row.get("position", ""),
                    "sim_selected_market": row.get("market", ""),
                    "team_rush_total_mean": total_mean,
                    "team_rush_total_min": int(np.min(totals)) if len(totals) else np.nan,
                    "team_rush_total_max": int(np.max(totals)) if len(totals) else np.nan,
                    "raw_player_rush_share": float(clean[j]),
                    "raw_team_rush_share_sum": raw_sum,
                    "final_player_probability": float(final_probs[j]),
                    "residual_probability": residual_prob,
                    "expected_carries_from_final_probability": expected,
                    "realized_multinomial_mean_carries": realized,
                    "allocation_sampling_delta": realized - expected,
                })
        return out

    with patch.object(simv2, "_allocate_counts", side_effect=wrapped):
        result = canonical_simulate(metrics, iterations=int(iterations), seed=int(seed))
    if call_index != len(call_plan):
        raise RuntimeError(f"allocation trace expected {len(call_plan)} calls but observed {call_index}")
    return result, pd.DataFrame(traces)


def _metrics(trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["mc_proj", "expected_carries_from_final_probability", "realized_multinomial_mean_carries"]:
        g = trace.loc[pd.to_numeric(trace[col], errors="coerce").notna() & pd.to_numeric(trace["actual"], errors="coerce").notna()].copy()
        pred = pd.to_numeric(g[col], errors="coerce"); actual = pd.to_numeric(g["actual"], errors="coerce"); err = pred - actual
        rows.append({"stage": col, "n": len(g), "mae": float(err.abs().mean()), "rmse": float(np.sqrt(np.mean(err * err))), "bias": float(err.mean()), "correlation": float(pred.corr(actual)) if pred.nunique()>1 and actual.nunique()>1 else np.nan, "pred_mean": float(pred.mean()), "actual_mean": float(actual.mean())})
    return pd.DataFrame(rows)


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--season",type=int,default=2025); p.add_argument("--prior-season",type=int,default=2024); p.add_argument("--weeks",default="1-18"); p.add_argument("--iterations",type=int,default=2000)
    p.add_argument("--player-logs",type=Path,default=Path("data/backtests/player_game_logs_history.csv")); p.add_argument("--team-weekly",type=Path,default=Path("data/backtests/team_weekly_history.csv")); p.add_argument("--schedule",type=Path,default=Path("data/backtests/schedule_history.csv")); p.add_argument("--universe-dir",type=Path,default=Path("data/backtests/pregame_universe")); p.add_argument("--injuries",type=Path,default=Path("data/backtests/injuries_history.csv")); p.add_argument("--weather",type=Path,default=Path("data/backtests/weather_history.csv")); p.add_argument("--out-dir",type=Path,default=Path("data/backtests/rushing_probability_trace")); a=p.parse_args()
    logs=_read(a.player_logs,"player logs"); team=_read(a.team_weekly,"team weekly"); sched=_read(a.schedule,"schedule"); injuries=_optional(a.injuries); weather=_optional(a.weather); all_rows=[]
    for week in _parse_weeks(a.weeks):
        u=_read(a.universe_dir/f"{a.season}_week_{week:02d}.csv",f"universe W{week}")
        bundle=build_historical_context_bundle(player_logs=logs,team_weekly=team,pregame_universe=u,schedule=sched,season=a.season,week=week,prior_season=a.prior_season,injuries=_exact_week(injuries,a.season,week),weather=_exact_week(weather,a.season,week))
        captured=[]
        def traced_sim(metrics, *, iterations=None, seed=None):
            result, t=_run_with_trace(metrics,iterations=int(iterations or a.iterations),seed=int(seed if seed is not None else 42+week)); captured.append(t); return result
        with patch.object(cp,"simulate",side_effect=traced_sim):
            metrics=cp.build_mc_predictions(bundle,iterations=a.iterations,seed=42+week)
        if not captured: raise RuntimeError(f"W{week} canonical simulator was not traced")
        t=captured[0]
        rush=metrics.loc[metrics.market.eq("rush_att"),["event_id","team","player_clean_key","mc_proj"]].drop_duplicates(["event_id","team","player_clean_key"])
        actual=cp.build_actual_rows(logs,a.season,week); actual=actual.loc[actual.market.eq("rush_att"),["team","player_clean_key","actual"]]
        t=t.merge(rush,on=["event_id","team","player_clean_key"],how="left",validate="one_to_one").merge(actual,on=["team","player_clean_key"],how="left",validate="one_to_one"); t.insert(0,"week",week)
        t["mc_vs_realized_abs_diff"]=(pd.to_numeric(t.mc_proj,errors="coerce")-pd.to_numeric(t.realized_multinomial_mean_carries,errors="coerce")).abs()
        all_rows.append(t)
        print(f"[rush27] W{week:02d} rows={len(t)} max_mc_vs_alloc={t.mc_vs_realized_abs_diff.max():.8f} mean_sampling_delta={t.allocation_sampling_delta.mean():.6f}")
    trace=pd.concat(all_rows,ignore_index=True); stages=_metrics(trace)
    team_summary=trace.groupby(["week","event_id","team"],as_index=False).agg(team_rush_total_mean=("team_rush_total_mean","first"),raw_team_rush_share_sum=("raw_team_rush_share_sum","first"),residual_probability=("residual_probability","first"),player_final_probability_sum=("final_player_probability","sum"),player_realized_carries_sum=("realized_multinomial_mean_carries","sum"))
    divergence=pd.DataFrame([{"rows":len(trace),"mc_vs_realized_mismatch_rows":int((trace.mc_vs_realized_abs_diff>1e-9).sum()),"mc_vs_realized_max_abs":float(trace.mc_vs_realized_abs_diff.max()),"mean_abs_sampling_delta":float(trace.allocation_sampling_delta.abs().mean()),"max_abs_sampling_delta":float(trace.allocation_sampling_delta.abs().max()),"mean_raw_team_share_sum":float(trace.drop_duplicates(["week","event_id","team"]).raw_team_rush_share_sum.mean()),"mean_residual_probability":float(trace.drop_duplicates(["week","event_id","team"]).residual_probability.mean())}])
    a.out_dir.mkdir(parents=True,exist_ok=True); trace.to_csv(a.out_dir/"rushing_probability_player_trace.csv",index=False); team_summary.to_csv(a.out_dir/"rushing_probability_team_trace.csv",index=False); stages.to_csv(a.out_dir/"rushing_probability_stage_summary.csv",index=False); divergence.to_csv(a.out_dir/"rushing_probability_divergence_summary.csv",index=False)
    print("\n[rush27] stage summary\n",stages.to_string(index=False)); print("\n[rush27] divergence summary\n",divergence.to_string(index=False)); return 0

if __name__=="__main__": raise SystemExit(main())
