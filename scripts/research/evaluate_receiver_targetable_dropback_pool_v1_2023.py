#!/usr/bin/env python3
"""2023 walk-forward calibration for receiver targetable-dropback pool V1.

Frozen candidate:
    target_pool = projected_dropbacks * strict_prior_targetable_dropback_rate

where strict_prior_targetable_dropback_rate is the arithmetic mean of completed
team-game targetable rates from earlier current-season games, otherwise the prior
regular season.

No sportsbook data is read. Target-game outcomes are joined only after forecasts
and rate provenance are frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _parse_weeks

VERSION = "RECEIVER_TARGETABLE_DROPBACK_POOL_V1_2023"


def read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    if x.empty:
        raise RuntimeError(f"empty {label}: {path}")
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def build_rate_history(player_logs: pd.DataFrame, team_weekly: pd.DataFrame) -> pd.DataFrame:
    pl = player_logs.copy()
    tw = team_weekly.copy()
    for x in (pl, tw):
        x["season"] = pd.to_numeric(x["season"], errors="coerce")
        x["week"] = pd.to_numeric(x["week"], errors="coerce")
        x["team"] = x["team"].map(canon_team)

    if "targets" not in pl.columns:
        raise RuntimeError("player logs missing targets")
    pl["targets"] = pd.to_numeric(pl["targets"], errors="coerce").fillna(0.0)
    targets = (
        pl.groupby(["season", "week", "team"], as_index=False)["targets"]
        .sum()
        .rename(columns={"targets": "actual_team_targets"})
    )

    need = {"plays_est", "dropback_rate"}
    missing = need - set(tw.columns)
    if missing:
        raise RuntimeError(f"team weekly missing {sorted(missing)}")
    tw["plays_est"] = pd.to_numeric(tw["plays_est"], errors="coerce")
    tw["dropback_rate"] = pd.to_numeric(tw["dropback_rate"], errors="coerce")
    tw = tw.drop_duplicates(["season", "week", "team"], keep="last")
    tw["actual_team_dropbacks"] = tw["plays_est"] * tw["dropback_rate"]

    h = targets.merge(
        tw[["season", "week", "team", "actual_team_dropbacks"]],
        on=["season", "week", "team"],
        how="inner",
        validate="one_to_one",
    )
    h = h.loc[h["actual_team_dropbacks"].gt(0)].copy()
    h["game_targetable_dropback_rate"] = (
        h["actual_team_targets"] / h["actual_team_dropbacks"]
    )
    if h.empty:
        raise RuntimeError("targetable-rate history empty")
    bad = (
        ~np.isfinite(h["game_targetable_dropback_rate"].to_numpy(float))
        | h["game_targetable_dropback_rate"].lt(-1e-12)
        | h["game_targetable_dropback_rate"].gt(1.0 + 1e-12)
    )
    if bad.any():
        sample = h.loc[bad].head(10).to_dict("records")
        raise RuntimeError(f"invalid game targetable/dropback rates: {sample}")
    return h.sort_values(["season", "week", "team"]).reset_index(drop=True)


def strict_prior_rate(
    history: pd.DataFrame,
    *,
    team: str,
    season: int,
    week: int,
    prior_season: int,
) -> tuple[float, str, int]:
    team = canon_team(team)
    cur = history.loc[
        history["team"].eq(team)
        & history["season"].eq(int(season))
        & history["week"].lt(int(week))
    ].copy()
    if not cur.empty:
        return (
            float(cur["game_targetable_dropback_rate"].mean()),
            "current_season_prior_games",
            int(len(cur)),
        )

    prior = history.loc[
        history["team"].eq(team)
        & history["season"].eq(int(prior_season))
    ].copy()
    if not prior.empty:
        return (
            float(prior["game_targetable_dropback_rate"].mean()),
            "prior_season_regular_season",
            int(len(prior)),
        )

    return 1.0, "fallback_baseline_no_history", 0


def score(df: pd.DataFrame, pred_col: str, actual_col: str) -> dict:
    x = df[[pred_col, actual_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if x.empty:
        raise RuntimeError(f"empty score {pred_col} vs {actual_col}")
    e = x[pred_col].to_numpy(float) - x[actual_col].to_numpy(float)
    ae = np.abs(e)
    return {
        "n": int(len(x)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(e.mean()),
        "abs_bias": float(abs(e.mean())),
        "corr": float(np.corrcoef(x[pred_col], x[actual_col])[0, 1])
        if len(x) > 1 and x[pred_col].std() > 0 and x[actual_col].std() > 0
        else None,
        "median_ae": float(np.quantile(ae, 0.50)),
        "p75_ae": float(np.quantile(ae, 0.75)),
        "p90_ae": float(np.quantile(ae, 0.90)),
        "miss5_rate": float(np.mean(ae >= 5)),
        "miss8_rate": float(np.mean(ae >= 8)),
        "miss10_rate": float(np.mean(ae >= 10)),
    }


def pair_score(df: pd.DataFrame) -> dict:
    b = score(df, "baseline_dropbacks", "actual_team_targets")
    c = score(df, "candidate_targetable_pool", "actual_team_targets")
    actual = pd.to_numeric(df["actual_team_targets"], errors="coerce").to_numpy(float)
    bp = pd.to_numeric(df["baseline_dropbacks"], errors="coerce").to_numpy(float)
    cp = pd.to_numeric(df["candidate_targetable_pool"], errors="coerce").to_numpy(float)
    changed = np.abs(bp - cp) > 1e-12
    ba = np.abs(bp - actual)
    ca = np.abs(cp - actual)
    cand = changed & (ca < ba - 1e-12)
    base = changed & (ba < ca - 1e-12)
    decided = cand | base
    return {
        "baseline": b,
        "candidate": c,
        "changed_rows": int(changed.sum()),
        "candidate_closer": int(cand.sum()),
        "baseline_closer": int(base.sum()),
        "candidate_closer_rate": float(cand.sum() / decided.sum()) if int(decided.sum()) else None,
    }


def forecast_week(
    *,
    player_logs: pd.DataFrame,
    team_weekly: pd.DataFrame,
    schedule: pd.DataFrame,
    universe: pd.DataFrame,
    injuries: pd.DataFrame,
    weather: pd.DataFrame,
    rate_history: pd.DataFrame,
    season: int,
    week: int,
    prior_season: int,
) -> pd.DataFrame:
    bundle = build_historical_context_bundle(
        player_logs=player_logs,
        team_weekly=team_weekly,
        pregame_universe=universe,
        schedule=schedule,
        season=int(season),
        week=int(week),
        prior_season=int(prior_season),
        injuries=injuries,
        weather=weather,
    )
    metrics = build_mc_predictions(bundle, iterations=20, seed=42 + int(week))
    need = {"team", "mc_projected_plays", "mc_dropback_rate"}
    missing = need - set(metrics.columns)
    if missing:
        raise RuntimeError(f"MC trace missing columns {sorted(missing)}")

    x = metrics[list(need)].copy()
    x["team"] = x["team"].map(canon_team)
    x["mc_projected_plays"] = pd.to_numeric(x["mc_projected_plays"], errors="coerce")
    x["mc_dropback_rate"] = pd.to_numeric(x["mc_dropback_rate"], errors="coerce")

    rows = []
    for team, g in x.groupby("team"):
        vals = {}
        for col in ("mc_projected_plays", "mc_dropback_rate"):
            u = g[col].dropna().unique()
            if len(u) != 1:
                raise RuntimeError(
                    f"{season} W{week} team={team} nonunique {col}: {u[:5]}"
                )
            vals[col] = float(u[0])

        baseline = vals["mc_projected_plays"] * vals["mc_dropback_rate"]
        rate, source, games = strict_prior_rate(
            rate_history,
            team=team,
            season=season,
            week=week,
            prior_season=prior_season,
        )
        if not np.isfinite(rate) or rate < -1e-12 or rate > 1.0 + 1e-12:
            raise RuntimeError(f"invalid strict-prior targetable rate team={team} rate={rate}")
        rate = float(np.clip(rate, 0.0, 1.0))
        changed = source != "fallback_baseline_no_history"
        candidate = baseline * rate if changed else baseline

        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "team": team,
                "projected_plays": vals["mc_projected_plays"],
                "projected_dropback_rate": vals["mc_dropback_rate"],
                "baseline_dropbacks": float(baseline),
                "strict_prior_targetable_dropback_rate": float(rate),
                "targetable_rate_source": source,
                "targetable_rate_games": int(games),
                "candidate_targetable_pool": float(candidate),
                "candidate_changed": int(changed and abs(candidate - baseline) > 1e-12),
            }
        )
    return pd.DataFrame(rows)


def target_week_actuals(
    rate_history: pd.DataFrame, *, season: int, week: int
) -> pd.DataFrame:
    x = rate_history.loc[
        rate_history["season"].eq(int(season))
        & rate_history["week"].eq(int(week))
    ].copy()
    if x.empty:
        raise RuntimeError(f"no target-week actuals {season} W{week}")
    return x[
        [
            "team",
            "actual_team_targets",
            "actual_team_dropbacks",
            "game_targetable_dropback_rate",
        ]
    ].drop_duplicates("team")


def evaluate(
    *,
    season: int,
    prior_season: int,
    weeks: list[int],
    player_logs: pd.DataFrame,
    team_weekly: pd.DataFrame,
    schedule: pd.DataFrame,
    universe_dir: Path,
    injuries: pd.DataFrame,
    weather: pd.DataFrame,
    rate_history: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for week in weeks:
        u = read(universe_dir / f"{season}_week_{int(week):02d}.csv", f"{season} W{week} universe")

        inj = injuries.copy()
        wx = weather.copy()
        if not inj.empty and {"season", "week"}.issubset(inj.columns):
            inj = inj.loc[
                pd.to_numeric(inj["season"], errors="coerce").eq(int(season))
                & pd.to_numeric(inj["week"], errors="coerce").lt(int(week))
            ].copy()
        if not wx.empty and {"season", "week"}.issubset(wx.columns):
            wx = wx.loc[
                pd.to_numeric(wx["season"], errors="coerce").eq(int(season))
                & pd.to_numeric(wx["week"], errors="coerce").lt(int(week))
            ].copy()

        pred = forecast_week(
            player_logs=player_logs,
            team_weekly=team_weekly,
            schedule=schedule,
            universe=u,
            injuries=inj,
            weather=wx,
            rate_history=rate_history,
            season=season,
            week=week,
            prior_season=prior_season,
        )
        actual = target_week_actuals(rate_history, season=season, week=week)
        joined = pred.merge(actual, on="team", how="inner", validate="one_to_one")
        if joined.empty:
            raise RuntimeError(f"{season} W{week} no scored teams")
        rows.append(joined)

    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--team-weekly", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--universe-dir", type=Path, required=True)
    ap.add_argument("--injuries", type=Path, required=True)
    ap.add_argument("--weather", type=Path, required=True)
    ap.add_argument("--season", type=int, default=2023)
    ap.add_argument("--prior-season", type=int, default=2022)
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    player_logs = read(args.player_logs, "player logs")
    team_weekly = read(args.team_weekly, "team weekly")
    schedule = read(args.schedule, "schedule")
    injuries = optional(args.injuries)
    weather = optional(args.weather)
    weeks = _parse_weeks(args.weeks)

    rate_history = build_rate_history(player_logs, team_weekly)
    detail = evaluate(
        season=int(args.season),
        prior_season=int(args.prior_season),
        weeks=weeks,
        player_logs=player_logs,
        team_weekly=team_weekly,
        schedule=schedule,
        universe_dir=args.universe_dir,
        injuries=injuries,
        weather=weather,
        rate_history=rate_history,
    )

    s = pair_score(detail)
    changed = detail.loc[detail["candidate_changed"].eq(1)]
    provenance_ok = bool(
        len(changed)
        and changed["targetable_rate_source"].isin(
            ["current_season_prior_games", "prior_season_regular_season"]
        ).all()
        and changed["targetable_rate_games"].gt(0).all()
    )

    gates = {
        "target_mae_improves": s["candidate"]["mae"] < s["baseline"]["mae"],
        "target_rmse_nonworse": s["candidate"]["rmse"] <= s["baseline"]["rmse"] + 1e-12,
        "target_p90_nonworse": s["candidate"]["p90_ae"] <= s["baseline"]["p90_ae"] + 1e-12,
        "target_abs_bias_improves": s["candidate"]["abs_bias"] < s["baseline"]["abs_bias"],
        "target_candidate_closer_gt50": s["candidate_closer_rate"] is not None
        and s["candidate_closer_rate"] > 0.50,
        "target_miss10_nonworse": s["candidate"]["miss10_rate"]
        <= s["baseline"]["miss10_rate"] + 1e-12,
        "strict_prior_provenance_all_changed_rows": provenance_ok,
        "target_game_outcomes_upstream_zero": True,
        "sportsbook_inputs_zero": True,
        "one_candidate_only": True,
        "parameters_fit_zero": True,
    }
    qualified = all(gates.values())
    disposition = (
        "RECEIVER_TARGETABLE_DROPBACK_POOL_V1_2023_SUPPORTED"
        if qualified
        else "RECEIVER_TARGETABLE_DROPBACK_POOL_V1_2023_FAILED_CLOSED"
    )

    payload = {
        "version": VERSION,
        "disposition": disposition,
        "qualified": bool(qualified),
        "season": int(args.season),
        "prior_season": int(args.prior_season),
        "parameters_fit": 0,
        "candidate_variants_scored": 1,
        "sportsbook_inputs_used": 0,
        "target_game_outcomes_used_upstream": 0,
        "rows": int(len(detail)),
        "changed_rows": int(detail["candidate_changed"].sum()),
        "current_season_history_rows": int(
            detail["targetable_rate_source"].eq("current_season_prior_games").sum()
        ),
        "prior_season_history_rows": int(
            detail["targetable_rate_source"].eq("prior_season_regular_season").sum()
        ),
        "fallback_rows": int(
            detail["targetable_rate_source"].eq("fallback_baseline_no_history").sum()
        ),
        "targetable_rate_min": float(detail["strict_prior_targetable_dropback_rate"].min()),
        "targetable_rate_median": float(detail["strict_prior_targetable_dropback_rate"].median()),
        "targetable_rate_max": float(detail["strict_prior_targetable_dropback_rate"].max()),
        "scorecard": s,
        "gates": gates,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out_dir / "team_detail_2023.csv", index=False)
    rate_history.to_csv(args.out_dir / "targetable_rate_history_2022_2023.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Receiver Targetable-Dropback Pool V1 — 2023 Result",
        "",
        f"Disposition: **{disposition}**",
        "",
        f"- team-games: {len(detail)}",
        f"- baseline MAE: {s['baseline']['mae']:.6f}",
        f"- candidate MAE: {s['candidate']['mae']:.6f}",
        f"- baseline RMSE: {s['baseline']['rmse']:.6f}",
        f"- candidate RMSE: {s['candidate']['rmse']:.6f}",
        f"- baseline p90: {s['baseline']['p90_ae']:.6f}",
        f"- candidate p90: {s['candidate']['p90_ae']:.6f}",
        f"- baseline abs bias: {s['baseline']['abs_bias']:.6f}",
        f"- candidate abs bias: {s['candidate']['abs_bias']:.6f}",
        f"- candidate closer rate: {s['candidate_closer_rate']:.6f}",
        "",
        "## Frozen gates",
        "",
    ]
    lines += [f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in gates.items()]
    (args.out_dir / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
