#!/usr/bin/env python3
"""Migration 26: trace the canonical MC rushing path end-to-end.

This is diagnostic-only. It compares the persisted walk-forward rush-attempt
projection with a same-seed rebuild, a direct lookup from simulation_v2, and a
deterministic expectation derived from the exact player row selected by the
canonical simulator. The goal is to identify the first stage where rushing
signal diverges without changing production football logic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.component_predictions import build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.simulation_v2 import lookup, simulate


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    return pd.read_csv(path)


def _optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _finite(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _selected_player_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mirror simulation_v2's exact player-row selection and expose its inputs."""
    x = metrics.copy()
    game_key = "event_id"
    cols = [game_key, "team", "player_clean_key"]
    selected = x.sort_values(cols).drop_duplicates(cols, keep="last").copy()
    selected = selected.rename(columns={
        "market": "sim_selected_market",
        "rules_rush_share": "sim_selected_rush_share",
        "rules_plays_est": "sim_selected_plays",
        "rules_pass_rate": "sim_selected_pass_rate",
    })

    rows = []
    for (game, team), g in selected.groupby([game_key, "team"], dropna=False):
        raw = np.array([_finite(v, 0.0) for v in g.get("sim_selected_rush_share", 0.0)], dtype=float)
        clean = np.clip(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 0.95)
        raw_sum = float(clean.sum())
        used = clean.copy()
        if raw_sum > 0.95:
            used *= 0.95 / raw_sum
        plays = float(np.mean([_finite(v, 64.0) for v in g.get("sim_selected_plays", 64.0)]))
        pass_rate = float(np.mean([_finite(v, 0.57) for v in g.get("sim_selected_pass_rate", 0.57)]))
        team_rush_mean = float(np.clip(plays, 50.0, 80.0) * (1.0 - np.clip(pass_rate, 0.35, 0.75)))
        for j, (_, r) in enumerate(g.iterrows()):
            rows.append({
                "event_id": game,
                "team": team,
                "player_clean_key": r["player_clean_key"],
                "sim_selected_market": r.get("sim_selected_market", ""),
                "sim_selected_rush_share": float(clean[j]),
                "sim_used_rush_share": float(used[j]),
                "sim_team_raw_share_sum": raw_sum,
                "sim_team_player_share_sum": float(used.sum()),
                "sim_team_expected_rushes_deterministic": team_rush_mean,
                "sim_deterministic_player_carries": team_rush_mean * float(used[j]),
            })
    return pd.DataFrame(rows)


def _stage_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["component_mc_proj", "rebuilt_mc_proj", "direct_lookup_mean", "sim_deterministic_player_carries"]:
        g = frame.loc[pd.to_numeric(frame[col], errors="coerce").notna() & pd.to_numeric(frame["actual"], errors="coerce").notna()].copy()
        if g.empty:
            continue
        pred = pd.to_numeric(g[col], errors="coerce")
        actual = pd.to_numeric(g["actual"], errors="coerce")
        err = pred - actual
        corr = float(pred.corr(actual)) if pred.nunique() > 1 and actual.nunique() > 1 else np.nan
        rows.append({
            "stage": col,
            "n": len(g),
            "mae": float(err.abs().mean()),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "bias": float(err.mean()),
            "correlation": corr,
            "pred_mean": float(pred.mean()),
            "actual_mean": float(actual.mean()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--prior-season", type=int, default=2024)
    p.add_argument("--weeks", default="1-18")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--predictions", type=Path, default=Path("data/backtests/component_predictions.csv"))
    p.add_argument("--player-logs", type=Path, default=Path("data/backtests/player_game_logs_history.csv"))
    p.add_argument("--team-weekly", type=Path, default=Path("data/backtests/team_weekly_history.csv"))
    p.add_argument("--schedule", type=Path, default=Path("data/backtests/schedule_history.csv"))
    p.add_argument("--universe-dir", type=Path, default=Path("data/backtests/pregame_universe"))
    p.add_argument("--injuries", type=Path, default=Path("data/backtests/injuries_history.csv"))
    p.add_argument("--weather", type=Path, default=Path("data/backtests/weather_history.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/canonical_rushing_trace"))
    a = p.parse_args()

    component = _read(a.predictions, "component predictions")
    logs = _read(a.player_logs, "player logs")
    team = _read(a.team_weekly, "team weekly")
    sched = _read(a.schedule, "schedule")
    injuries = _optional(a.injuries)
    weather = _optional(a.weather)
    all_rows = []

    for week in _parse_weeks(a.weeks):
        u = _read(a.universe_dir / f"{a.season}_week_{week:02d}.csv", f"universe W{week}")
        bundle = build_historical_context_bundle(
            player_logs=logs, team_weekly=team, pregame_universe=u, schedule=sched,
            season=a.season, week=week, prior_season=a.prior_season,
            injuries=_exact_week(injuries, a.season, week),
            weather=_exact_week(weather, a.season, week),
        )
        seed = 42 + week
        rebuilt = build_mc_predictions(bundle, iterations=a.iterations, seed=seed)
        rush_rows = rebuilt.loc[rebuilt.market.eq("rush_att")].copy()

        # Direct lookup from simulation_v2 using the same fully prepared canonical metrics.
        direct = simulate(rebuilt, iterations=a.iterations, seed=seed)
        rush_rows["direct_lookup_mean"] = [
            float(np.mean(v)) if v is not None and len(v) else np.nan
            for v in (lookup(direct, r, "rush_att") for _, r in rush_rows.iterrows())
        ]
        rush_rows = rush_rows.rename(columns={"mc_proj": "rebuilt_mc_proj", "rules_rush_share": "rush_att_row_rush_share"})

        selected = _selected_player_rows(rebuilt)
        keep = [
            "event_id", "team", "player_clean_key", "player", "position",
            "rush_att_row_rush_share", "rebuilt_mc_proj", "direct_lookup_mean",
        ]
        keep = [c for c in keep if c in rush_rows.columns]
        trace = rush_rows[keep].merge(selected, on=["event_id", "team", "player_clean_key"], how="left", validate="one_to_one")

        comp = component.copy()
        comp.columns = [str(c).strip().lower() for c in comp.columns]
        comp = comp.loc[
            pd.to_numeric(comp.get("season"), errors="coerce").eq(a.season)
            & pd.to_numeric(comp.get("week"), errors="coerce").eq(week)
            & comp.get("market", "").eq("rush_att")
        ].copy()
        comp = comp[[c for c in ["team", "player_clean_key", "mc_proj", "actual"] if c in comp.columns]].drop_duplicates(["team", "player_clean_key"])
        comp = comp.rename(columns={"mc_proj": "component_mc_proj"})
        trace = trace.merge(comp, on=["team", "player_clean_key"], how="left", validate="one_to_one")
        trace.insert(0, "week", week)

        trace["abs_component_vs_rebuilt"] = (pd.to_numeric(trace.component_mc_proj, errors="coerce") - pd.to_numeric(trace.rebuilt_mc_proj, errors="coerce")).abs()
        trace["abs_rebuilt_vs_direct"] = (pd.to_numeric(trace.rebuilt_mc_proj, errors="coerce") - pd.to_numeric(trace.direct_lookup_mean, errors="coerce")).abs()
        trace["abs_rushrow_vs_selected_share"] = (pd.to_numeric(trace.rush_att_row_rush_share, errors="coerce") - pd.to_numeric(trace.sim_selected_rush_share, errors="coerce")).abs()
        det = pd.to_numeric(trace.sim_deterministic_player_carries, errors="coerce")
        trace["direct_to_deterministic_ratio"] = pd.to_numeric(trace.direct_lookup_mean, errors="coerce") / det.where(det.abs() > 1e-9)
        all_rows.append(trace)
        print(
            f"[rush26] W{week:02d} rows={len(trace)} "
            f"component/rebuilt maxdiff={trace.abs_component_vs_rebuilt.max():.6f} "
            f"rebuilt/direct maxdiff={trace.abs_rebuilt_vs_direct.max():.6f} "
            f"share-row mismatch rows={(trace.abs_rushrow_vs_selected_share > 1e-9).sum()}"
        )

    out = pd.concat(all_rows, ignore_index=True)
    stages = _stage_metrics(out)
    divergence = pd.DataFrame([{
        "rows": len(out),
        "component_vs_rebuilt_mismatch_rows": int((out.abs_component_vs_rebuilt > 1e-9).sum()),
        "component_vs_rebuilt_max_abs": float(out.abs_component_vs_rebuilt.max()),
        "rebuilt_vs_direct_mismatch_rows": int((out.abs_rebuilt_vs_direct > 1e-9).sum()),
        "rebuilt_vs_direct_max_abs": float(out.abs_rebuilt_vs_direct.max()),
        "rushrow_vs_selected_share_mismatch_rows": int((out.abs_rushrow_vs_selected_share > 1e-9).sum()),
        "rushrow_vs_selected_share_max_abs": float(out.abs_rushrow_vs_selected_share.max()),
        "mean_direct_to_deterministic_ratio": float(pd.to_numeric(out.direct_to_deterministic_ratio, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().mean()),
    }])

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_dir / "canonical_rushing_player_trace.csv", index=False)
    stages.to_csv(a.out_dir / "canonical_rushing_stage_summary.csv", index=False)
    divergence.to_csv(a.out_dir / "canonical_rushing_divergence_summary.csv", index=False)
    print("\n[rush26] stage summary\n", stages.to_string(index=False))
    print("\n[rush26] divergence summary\n", divergence.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
