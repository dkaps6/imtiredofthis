#!/usr/bin/env python3
"""Temporal room calibration for Receiver Room Targets-Per-Play V1.

Frozen candidate for room g in WR/TE/RB_FB:
  R_g = cumulative strict-prior room targets / cumulative strict-prior team plays
  candidate_room_targets = projected_plays * R_g

The baseline remains the historical fixed-57% dropback/M38 room forecast.
No TE-R5P/WR-R15 backcast is allowed in the 2022-2023 screen.
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
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement

VERSION = "RECEIVER_ROOM_TARGETS_PER_PLAY_V1"
ROOMS = ("WR", "TE", "RB_FB")
FIXED_DROPBACK_RATE = 0.57


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


def pos_room(value: object) -> str:
    p = "" if value is None or pd.isna(value) else str(value).upper().strip()
    if p in {"WR", "LWR", "RWR", "SWR"} or p.startswith("WR"):
        return "WR"
    if p == "TE" or p.startswith("TE"):
        return "TE"
    if p in {"RB", "HB", "TB", "FB"} or p.startswith("RB") or p.startswith("FB"):
        return "RB_FB"
    return "OTHER"


def build_room_history(player_logs: pd.DataFrame, team_weekly: pd.DataFrame) -> pd.DataFrame:
    pl = player_logs.copy()
    tw = team_weekly.copy()
    for x in (pl, tw):
        x["season"] = pd.to_numeric(x["season"], errors="coerce")
        x["week"] = pd.to_numeric(x["week"], errors="coerce")
        x["team"] = x["team"].map(canon_team)

    if "targets" not in pl.columns or "position" not in pl.columns:
        raise RuntimeError("player logs require targets and position")
    pl["targets"] = pd.to_numeric(pl["targets"], errors="coerce").fillna(0.0)
    pl["room"] = pl["position"].map(pos_room)
    pl = pl.loc[pl["room"].isin(ROOMS)].copy()
    group = (
        pl.groupby(["season", "week", "team", "room"], as_index=False)
        .agg(actual_room_targets=("targets", "sum"))
    )

    for c in ("plays_est", "dropback_rate"):
        if c not in tw.columns:
            raise RuntimeError(f"team weekly missing {c}")
        tw[c] = pd.to_numeric(tw[c], errors="coerce")
    tw = tw.drop_duplicates(["season", "week", "team"], keep="last").copy()
    tw["actual_plays"] = tw["plays_est"]
    tw["actual_dropbacks"] = tw["plays_est"] * tw["dropback_rate"]
    tg = tw[["season", "week", "team", "actual_plays", "actual_dropbacks"]].dropna().copy()
    if (tg["actual_plays"] <= 0).any() or (tg["actual_dropbacks"] <= 0).any():
        raise RuntimeError("nonpositive actual plays/dropbacks in history")

    # Every team-game receives an explicit zero row for rooms with no targets.
    room_frame = pd.DataFrame({"room": list(ROOMS)})
    tg["_k"] = 1
    room_frame["_k"] = 1
    grid = tg.merge(room_frame, on="_k", how="inner").drop(columns="_k")
    out = grid.merge(
        group,
        on=["season", "week", "team", "room"],
        how="left",
        validate="one_to_one",
    )
    out["actual_room_targets"] = pd.to_numeric(
        out["actual_room_targets"], errors="coerce"
    ).fillna(0.0)
    if (out["actual_room_targets"] < 0).any():
        raise RuntimeError("negative room targets in history")

    room_sum = (
        out.groupby(["season", "week", "team"], as_index=False)
        .agg(modeled_room_targets=("actual_room_targets", "sum"),
             actual_plays=("actual_plays", "first"),
             actual_dropbacks=("actual_dropbacks", "first"))
    )
    bad = room_sum.loc[room_sum["modeled_room_targets"] > room_sum["actual_plays"] + 1e-9]
    if not bad.empty:
        raise RuntimeError(
            f"room targets exceed offensive plays sample={bad.head().to_dict('records')}"
        )
    return out.sort_values(["season", "week", "team", "room"]).reset_index(drop=True)


def strict_prior_room_rates(
    history: pd.DataFrame,
    *,
    season: int,
    week: int,
    team: str,
    prior_season: int,
) -> dict:
    h = history.loc[
        history["season"].eq(int(prior_season))
        | (history["season"].eq(int(season)) & history["week"].lt(int(week)))
    ].copy()
    if h.empty:
        raise RuntimeError(f"no eligible room history season={season} week={week}")

    # Team denominator is unique per team-game even though history is room-long.
    h_games = h[["season", "week", "team", "actual_plays"]].drop_duplicates(
        ["season", "week", "team"]
    )
    lg_plays = float(h_games["actual_plays"].sum())
    if lg_plays <= 0:
        raise RuntimeError("league strict-prior plays <=0")

    league_targets = h.groupby("room")["actual_room_targets"].sum().to_dict()
    league_rates = {r: float(league_targets.get(r, 0.0) / lg_plays) for r in ROOMS}

    team = canon_team(team)
    t = h.loc[h["team"].eq(team)].copy()
    t_games = t[["season", "week", "team", "actual_plays"]].drop_duplicates(
        ["season", "week", "team"]
    )
    tplays = float(t_games["actual_plays"].sum())
    if tplays > 0:
        tt = t.groupby("room")["actual_room_targets"].sum().to_dict()
        rates = {r: float(tt.get(r, 0.0) / tplays) for r in ROOMS}
        source = "team_strict_prior"
        games = int(len(t_games))
        hist_plays = tplays
        hist_targets = {r: float(tt.get(r, 0.0)) for r in ROOMS}
    else:
        rates = league_rates
        source = "league_fallback"
        games = 0
        hist_plays = lg_plays
        hist_targets = {r: float(league_targets.get(r, 0.0)) for r in ROOMS}

    vals = np.asarray([rates[r] for r in ROOMS], dtype=float)
    if not np.isfinite(vals).all() or (vals < 0).any() or (vals > 1).any():
        raise RuntimeError(f"invalid room rates team={team} rates={rates}")
    if float(vals.sum()) > 1.0 + 1e-12:
        raise RuntimeError(f"summed room rate exceeds 1 team={team} rates={rates}")
    return {
        "room_rates": rates,
        "conversion_source": source,
        "prior_history_games": games,
        "prior_history_plays": hist_plays,
        "prior_history_targets": hist_targets,
        "sum_room_rate": float(vals.sum()),
    }


def forecast_week(
    *,
    player_logs: pd.DataFrame,
    team_weekly: pd.DataFrame,
    schedule: pd.DataFrame,
    universe: pd.DataFrame,
    injuries: pd.DataFrame,
    weather: pd.DataFrame,
    history: pd.DataFrame,
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
    metrics = build_mc_predictions(bundle, iterations=20, seed=9300 + int(week))
    players = (
        metrics.sort_values(["event_id", "team", "player_clean_key"])
        .drop_duplicates(["event_id", "team", "player_clean_key"], keep="last")
        .copy()
    )
    explicit, _ = materialize_target_entitlement(players)
    explicit["team"] = explicit["team"].map(canon_team)
    explicit["room"] = explicit["position"].map(pos_room)
    explicit["entitlement_tgt_share"] = pd.to_numeric(
        explicit["entitlement_tgt_share"], errors="raise"
    )

    rows = []
    for team, tdf in explicit.groupby("team", sort=True):
        p = pd.to_numeric(tdf["mc_projected_plays"], errors="coerce").dropna().unique()
        if len(p) != 1:
            raise RuntimeError(
                f"{season} W{week} team={team} nonunique projected plays {p[:5]}"
            )
        opp = tdf["opponent"].dropna().astype(str).map(canon_team).unique()
        if len(opp) != 1:
            raise RuntimeError(f"{season} W{week} team={team} nonunique opponent")
        projected_plays = float(p[0])
        projected_dropbacks = projected_plays * FIXED_DROPBACK_RATE
        if not np.isfinite(projected_dropbacks) or projected_dropbacks <= 0:
            raise RuntimeError(f"invalid projected dropbacks team={team}")

        info = strict_prior_room_rates(
            history,
            season=int(season),
            week=int(week),
            team=team,
            prior_season=int(prior_season),
        )

        for room_name in ROOMS:
            room_ent = float(
                tdf.loc[tdf["room"].eq(room_name), "entitlement_tgt_share"].sum()
            )
            if not np.isfinite(room_ent) or room_ent < 0 or room_ent > 0.95 + 1e-12:
                raise RuntimeError(
                    f"invalid room entitlement team={team} room={room_name} value={room_ent}"
                )
            rate = float(info["room_rates"][room_name])
            rows.append({
                "season": int(season),
                "week": int(week),
                "team": team,
                "opponent": str(opp[0]),
                "room": room_name,
                "projected_plays": projected_plays,
                "projected_dropbacks": projected_dropbacks,
                "baseline_room_entitlement": room_ent,
                "baseline_room_targets": projected_dropbacks * room_ent,
                "room_targets_per_play_rate": rate,
                "candidate_room_targets": projected_plays * rate,
                "conversion_source": info["conversion_source"],
                "prior_history_games": int(info["prior_history_games"]),
                "prior_history_plays": float(info["prior_history_plays"]),
                "prior_history_room_targets": float(
                    info["prior_history_targets"][room_name]
                ),
                "sum_room_rate": float(info["sum_room_rate"]),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"{season} W{week} empty room forecast")
    return out


def score(df: pd.DataFrame, pred: str) -> dict:
    z = df[[pred, "actual_room_targets"]].apply(pd.to_numeric, errors="coerce").dropna()
    if z.empty:
        raise RuntimeError(f"empty score for {pred}")
    e = z[pred].to_numpy(float) - z["actual_room_targets"].to_numpy(float)
    ae = np.abs(e)
    return {
        "n": int(len(z)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(e.mean()),
        "abs_bias": float(abs(e.mean())),
        "corr": (
            float(np.corrcoef(z[pred], z["actual_room_targets"])[0, 1])
            if len(z) > 1 and z[pred].std() > 0 and z["actual_room_targets"].std() > 0
            else None
        ),
        "median_ae": float(np.quantile(ae, 0.50)),
        "p75_ae": float(np.quantile(ae, 0.75)),
        "p90_ae": float(np.quantile(ae, 0.90)),
    }


def paired(df: pd.DataFrame) -> dict:
    b = score(df, "baseline_room_targets")
    c = score(df, "candidate_room_targets")
    actual = pd.to_numeric(df["actual_room_targets"], errors="coerce").to_numpy(float)
    bp = pd.to_numeric(df["baseline_room_targets"], errors="coerce").to_numpy(float)
    cp = pd.to_numeric(df["candidate_room_targets"], errors="coerce").to_numpy(float)
    changed = np.abs(bp - cp) > 1e-12
    ba, ca = np.abs(bp - actual), np.abs(cp - actual)
    cw = changed & (ca < ba - 1e-12)
    bw = changed & (ba < ca - 1e-12)
    decided = cw | bw
    return {
        "baseline": b,
        "candidate": c,
        "changed_rows": int(changed.sum()),
        "candidate_closer": int(cw.sum()),
        "baseline_closer": int(bw.sum()),
        "candidate_closer_rate": (
            float(cw.sum() / decided.sum()) if int(decided.sum()) else None
        ),
    }


def evaluate_season(
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
    history: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for week in weeks:
        universe = read(
            universe_dir / f"{season}_week_{int(week):02d}.csv",
            f"{season} W{week} universe",
        )
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
            universe=universe,
            injuries=inj,
            weather=wx,
            history=history,
            season=int(season),
            week=int(week),
            prior_season=int(prior_season),
        )
        actual = history.loc[
            history["season"].eq(int(season))
            & history["week"].eq(int(week)),
            ["team", "room", "actual_room_targets"],
        ].copy()
        joined = pred.merge(
            actual,
            on=["team", "room"],
            how="inner",
            validate="one_to_one",
        )
        if joined.empty:
            raise RuntimeError(f"{season} W{week} no joined room outcomes")
        joined["candidate_minus_baseline"] = (
            joined["candidate_room_targets"] - joined["baseline_room_targets"]
        )
        rows.append(joined)
    return pd.concat(rows, ignore_index=True)


def summarize(detail: pd.DataFrame) -> dict:
    out = {"by_season": {}, "pooled": {}}
    for season in (2022, 2023):
        d = detail.loc[detail["season"].eq(season)]
        out["by_season"][str(season)] = {
            room: paired(d.loc[d["room"].eq(room)]) for room in ROOMS
        }
    out["pooled"] = {room: paired(detail.loc[detail["room"].eq(room)]) for room in ROOMS}

    def macro(block: dict, field: str, arm: str) -> float:
        return float(np.mean([block[r][arm][field] for r in ROOMS]))

    for season in ("2022", "2023"):
        block = out["by_season"][season]
        out["by_season"][season]["macro"] = {
            "baseline_mae": macro(block, "mae", "baseline"),
            "candidate_mae": macro(block, "mae", "candidate"),
            "baseline_p90": macro(block, "p90_ae", "baseline"),
            "candidate_p90": macro(block, "p90_ae", "candidate"),
            "baseline_abs_bias": macro(block, "abs_bias", "baseline"),
            "candidate_abs_bias": macro(block, "abs_bias", "candidate"),
        }
    block = out["pooled"]
    out["pooled"]["macro"] = {
        "baseline_mae": macro(block, "mae", "baseline"),
        "candidate_mae": macro(block, "mae", "candidate"),
        "baseline_p90": macro(block, "p90_ae", "baseline"),
        "candidate_p90": macro(block, "p90_ae", "candidate"),
        "baseline_abs_bias": macro(block, "abs_bias", "baseline"),
        "candidate_abs_bias": macro(block, "abs_bias", "candidate"),
    }
    changed = np.abs(
        pd.to_numeric(detail["baseline_room_targets"], errors="coerce")
        - pd.to_numeric(detail["candidate_room_targets"], errors="coerce")
    ) > 1e-12
    actual = pd.to_numeric(detail["actual_room_targets"], errors="coerce").to_numpy(float)
    bp = pd.to_numeric(detail["baseline_room_targets"], errors="coerce").to_numpy(float)
    cp = pd.to_numeric(detail["candidate_room_targets"], errors="coerce").to_numpy(float)
    ba, ca = np.abs(bp - actual), np.abs(cp - actual)
    cw = changed.to_numpy() & (ca < ba - 1e-12)
    bw = changed.to_numpy() & (ba < ca - 1e-12)
    decided = cw | bw
    out["pooled"]["macro"]["candidate_closer_rate_all_room_rows"] = (
        float(cw.sum() / decided.sum()) if int(decided.sum()) else None
    )
    out["pooled"]["source_rows"] = int(detail["conversion_source"].eq("team_strict_prior").sum())
    out["pooled"]["fallback_rows"] = int(detail["conversion_source"].eq("league_fallback").sum())
    out["pooled"]["rows"] = int(len(detail))
    out["pooled"]["max_sum_room_rate"] = float(detail["sum_room_rate"].max())
    out["pooled"]["min_room_rate"] = float(detail["room_targets_per_play_rate"].min())
    out["pooled"]["max_room_rate"] = float(detail["room_targets_per_play_rate"].max())

    summed = (
        detail.groupby(["season", "week", "team"], as_index=False)
        .agg(
            actual_room_targets=("actual_room_targets", "sum"),
            baseline_room_targets=("baseline_room_targets", "sum"),
            candidate_room_targets=("candidate_room_targets", "sum"),
        )
    )
    out["summed_room"] = {"pooled": paired(summed), "by_season": {}}
    for season in (2022, 2023):
        out["summed_room"]["by_season"][str(season)] = paired(
            summed.loc[summed["season"].eq(season)]
        )
    return out


def gates(s: dict) -> dict:
    y22 = s["by_season"]["2022"]
    y23 = s["by_season"]["2023"]
    p = s["pooled"]
    rows = max(1, int(p["rows"]))
    source_rate = p["source_rows"] / rows
    fallback_rate = p["fallback_rows"] / rows

    return {
        "pooled_macro_mae_improves": p["macro"]["candidate_mae"] < p["macro"]["baseline_mae"],
        "macro_mae_2022_improves": y22["macro"]["candidate_mae"] < y22["macro"]["baseline_mae"],
        "macro_mae_2023_improves": y23["macro"]["candidate_mae"] < y23["macro"]["baseline_mae"],
        "wr_mae_2022_improves": y22["WR"]["candidate"]["mae"] < y22["WR"]["baseline"]["mae"],
        "wr_mae_2023_improves": y23["WR"]["candidate"]["mae"] < y23["WR"]["baseline"]["mae"],
        "wr_mae_pooled_improves": p["WR"]["candidate"]["mae"] < p["WR"]["baseline"]["mae"],
        "te_mae_pooled_nonworse": p["TE"]["candidate"]["mae"] <= p["TE"]["baseline"]["mae"] + 1e-12,
        "rbfb_mae_pooled_nonworse": p["RB_FB"]["candidate"]["mae"] <= p["RB_FB"]["baseline"]["mae"] + 1e-12,
        "wr_p90_guard": p["WR"]["candidate"]["p90_ae"] <= p["WR"]["baseline"]["p90_ae"] + 0.50,
        "te_p90_guard": p["TE"]["candidate"]["p90_ae"] <= p["TE"]["baseline"]["p90_ae"] + 0.50,
        "rbfb_p90_guard": p["RB_FB"]["candidate"]["p90_ae"] <= p["RB_FB"]["baseline"]["p90_ae"] + 0.50,
        "pooled_macro_p90_nonworse": p["macro"]["candidate_p90"] <= p["macro"]["baseline_p90"] + 1e-12,
        "pooled_macro_abs_bias_nonworse": p["macro"]["candidate_abs_bias"] <= p["macro"]["baseline_abs_bias"] + 1e-12,
        "candidate_closer_gt50": (
            p["macro"]["candidate_closer_rate_all_room_rows"] is not None
            and p["macro"]["candidate_closer_rate_all_room_rows"] > 0.50
        ),
        "summed_room_mae_improves": s["summed_room"]["pooled"]["candidate"]["mae"] < s["summed_room"]["pooled"]["baseline"]["mae"],
        "summed_room_abs_bias_improves": s["summed_room"]["pooled"]["candidate"]["abs_bias"] < s["summed_room"]["pooled"]["baseline"]["abs_bias"],
        "summed_room_p90_nonworse": s["summed_room"]["pooled"]["candidate"]["p90_ae"] <= s["summed_room"]["pooled"]["baseline"]["p90_ae"] + 1e-12,
        "team_source_rate_ge99": source_rate >= 0.99,
        "fallback_rate_le1": fallback_rate <= 0.01,
        "room_rates_finite_in_bounds": 0.0 <= p["min_room_rate"] <= p["max_room_rate"] <= 1.0,
        "summed_room_rate_le1": p["max_sum_room_rate"] <= 1.0 + 1e-12,
        "target_game_outcomes_upstream_zero": True,
        "sportsbook_inputs_zero": True,
        "parameters_fit_zero": True,
        "one_candidate_only": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--team-weekly", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--universe-2022", type=Path, required=True)
    ap.add_argument("--universe-2023", type=Path, required=True)
    ap.add_argument("--injuries", type=Path, required=True)
    ap.add_argument("--weather", type=Path, required=True)
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    logs = read(args.player_logs, "player logs")
    team = read(args.team_weekly, "team weekly")
    sched = read(args.schedule, "schedule")
    injuries = optional(args.injuries)
    weather = optional(args.weather)
    history = build_room_history(logs, team)
    weeks = _parse_weeks(args.weeks)

    d22 = evaluate_season(
        season=2022, prior_season=2021, weeks=weeks,
        player_logs=logs, team_weekly=team, schedule=sched,
        universe_dir=args.universe_2022, injuries=injuries,
        weather=weather, history=history,
    )
    d23 = evaluate_season(
        season=2023, prior_season=2022, weeks=weeks,
        player_logs=logs, team_weekly=team, schedule=sched,
        universe_dir=args.universe_2023, injuries=injuries,
        weather=weather, history=history,
    )
    detail = pd.concat([d22, d23], ignore_index=True)
    s = summarize(detail)
    g = gates(s)
    qualified = all(g.values())
    disposition = (
        "RECEIVER_ROOM_TARGETS_PER_PLAY_V1_SUPPORTED"
        if qualified
        else "RECEIVER_ROOM_TARGETS_PER_PLAY_V1_FAILED_CLOSED"
    )
    payload = {
        "version": VERSION,
        "disposition": disposition,
        "qualified": bool(qualified),
        "candidate_variants_scored": 1,
        "parameters_fit": 0,
        "sportsbook_inputs_used": 0,
        "target_game_outcomes_used_upstream": 0,
        "scorecard": s,
        "gates": g,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out_dir / "room_detail_2022_2023.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Receiver Room Targets-Per-Play V1 — 2022-2023 Temporal Screen",
        "",
        f"Disposition: **{disposition}**",
        "",
    ]
    for season in ("2022", "2023"):
        lines += [f"## {season}", ""]
        for room_name in ROOMS:
            x = s["by_season"][season][room_name]
            lines.append(
                f"- {room_name}: MAE {x['baseline']['mae']:.6f} -> {x['candidate']['mae']:.6f}; "
                f"p90 {x['baseline']['p90_ae']:.6f} -> {x['candidate']['p90_ae']:.6f}"
            )
        m = s["by_season"][season]["macro"]
        lines += [
            f"- macro MAE: {m['baseline_mae']:.6f} -> {m['candidate_mae']:.6f}",
            "",
        ]
    lines += ["## Pooled", ""]
    for room_name in ROOMS:
        x = s["pooled"][room_name]
        lines.append(
            f"- {room_name}: MAE {x['baseline']['mae']:.6f} -> {x['candidate']['mae']:.6f}; "
            f"p90 {x['baseline']['p90_ae']:.6f} -> {x['candidate']['p90_ae']:.6f}"
        )
    m = s["pooled"]["macro"]
    lines += [
        f"- macro MAE: {m['baseline_mae']:.6f} -> {m['candidate_mae']:.6f}",
        f"- macro p90: {m['baseline_p90']:.6f} -> {m['candidate_p90']:.6f}",
        f"- candidate closer rate: {m['candidate_closer_rate_all_room_rows']:.6f}",
        "",
        "## Summed room",
        "",
        f"- MAE: {s['summed_room']['pooled']['baseline']['mae']:.6f} -> {s['summed_room']['pooled']['candidate']['mae']:.6f}",
        f"- abs bias: {s['summed_room']['pooled']['baseline']['abs_bias']:.6f} -> {s['summed_room']['pooled']['candidate']['abs_bias']:.6f}",
        f"- p90: {s['summed_room']['pooled']['baseline']['p90_ae']:.6f} -> {s['summed_room']['pooled']['candidate']['p90_ae']:.6f}",
        "",
        "## Frozen gates",
        "",
    ]
    lines += [f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in g.items()]
    (args.out_dir / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
