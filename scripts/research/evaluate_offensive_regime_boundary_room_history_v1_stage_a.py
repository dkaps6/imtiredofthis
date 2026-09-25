#!/usr/bin/env python3
"""Stage-A falsification for Offensive Regime Boundary Room History V1.

Frozen before scoring in:
docs/research/OFFENSIVE_REGIME_BOUNDARY_ROOM_HISTORY_V1_PLAN.md

One candidate only:
- if the prior-season primary QB is on the target-week pregame roster, preserve
  the parent prior-season + strict-current targets-per-play history;
- if that QB is absent, cut history at the season boundary and use strict-current
  team games only;
- at zero strict-current games, use the prior-season league room rate.

The boundary applies to WR, TE and RB_FB. No fitted parameters, windows,
recency weights, blends, sportsbook inputs or target-week outcomes are used
upstream.
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
from scripts.utils.canonical_names import canonicalize_player_name_safe

VERSION = "OFFENSIVE_REGIME_BOUNDARY_ROOM_HISTORY_V1_STAGE_A"
ROOMS = ("WR", "TE", "RB_FB")
FIXED_DROPBACK_RATE = 0.57
STAGE_SEASONS = (2020, 2021)


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


def player_key(value: object) -> str:
    raw = "" if value is None or pd.isna(value) else str(value).strip()
    if not raw:
        return ""
    try:
        _, key = canonicalize_player_name_safe(raw)
        if key:
            return str(key)
    except Exception:
        pass
    return "".join(ch.lower() for ch in raw if ch.isalnum())


def pos_room(value: object) -> str:
    p = "" if value is None or pd.isna(value) else str(value).upper().strip()
    if p in {"WR", "LWR", "RWR", "SWR"} or p.startswith("WR"):
        return "WR"
    if p == "TE" or p.startswith("TE"):
        return "TE"
    if p in {"RB", "HB", "TB", "FB"} or p.startswith("RB") or p.startswith("FB"):
        return "RB_FB"
    return "OTHER"


def is_qb(value: object) -> bool:
    p = "" if value is None or pd.isna(value) else str(value).upper().strip()
    return p == "QB" or p.startswith("QB")


def prep_logs(player_logs: pd.DataFrame) -> pd.DataFrame:
    x = player_logs.copy()
    req = {"season", "week", "team", "position", "targets", "pass_att"}
    missing = req - set(x.columns)
    if missing:
        raise RuntimeError(f"player logs missing {sorted(missing)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce").astype("Int64")
    x["week"] = pd.to_numeric(x["week"], errors="coerce").astype("Int64")
    x["team"] = x["team"].map(canon_team)
    x["position"] = x["position"].fillna("").astype(str).str.upper().str.strip()
    x["targets"] = pd.to_numeric(x["targets"], errors="coerce").fillna(0.0)
    x["pass_att"] = pd.to_numeric(x["pass_att"], errors="coerce").fillna(0.0)
    if "player_clean_key" in x.columns:
        x["player_clean_key"] = x["player_clean_key"].fillna("").astype(str)
    elif "player" in x.columns:
        x["player_clean_key"] = x["player"].map(player_key)
    else:
        raise RuntimeError("player logs missing player_clean_key/player")
    return x


def build_prior_primary_qb(logs: pd.DataFrame) -> dict[tuple[int, str], str]:
    q = logs.loc[logs["position"].map(is_qb)].copy()
    agg = (
        q.groupby(["season", "team", "player_clean_key"], as_index=False)
        .agg(pass_att=("pass_att", "sum"))
        .sort_values(
            ["season", "team", "pass_att", "player_clean_key"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
    )
    top = agg.drop_duplicates(["season", "team"], keep="first")
    return {
        (int(r.season), canon_team(r.team)): str(r.player_clean_key)
        for r in top.itertuples(index=False)
        if str(r.player_clean_key)
    }


def build_room_history(player_logs: pd.DataFrame, team_weekly: pd.DataFrame) -> pd.DataFrame:
    pl = player_logs.copy()
    tw = team_weekly.copy()
    for x in (pl, tw):
        x["season"] = pd.to_numeric(x["season"], errors="coerce")
        x["week"] = pd.to_numeric(x["week"], errors="coerce")
        x["team"] = x["team"].map(canon_team)

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
    return out.sort_values(["season", "week", "team", "room"]).reset_index(drop=True)


def rate_block(frame: pd.DataFrame, *, team: str | None = None) -> dict:
    z = frame.copy()
    if team is not None:
        z = z.loc[z["team"].eq(canon_team(team))].copy()
    games = z[["season", "week", "team", "actual_plays"]].drop_duplicates(
        ["season", "week", "team"]
    )
    plays = float(pd.to_numeric(games["actual_plays"], errors="coerce").sum())
    if plays <= 0:
        return {"plays": 0.0, "games": 0, "targets": {r: 0.0 for r in ROOMS}, "rates": None}
    targets = z.groupby("room")["actual_room_targets"].sum().to_dict()
    rates = {r: float(targets.get(r, 0.0) / plays) for r in ROOMS}
    vals = np.asarray([rates[r] for r in ROOMS], dtype=float)
    if not np.isfinite(vals).all() or (vals < 0).any() or (vals > 1).any():
        raise RuntimeError(f"invalid room rates rates={rates}")
    if float(vals.sum()) > 1.0 + 1e-12:
        raise RuntimeError(f"summed room rate exceeds 1 rates={rates}")
    return {
        "plays": plays,
        "games": int(len(games)),
        "targets": {r: float(targets.get(r, 0.0)) for r in ROOMS},
        "rates": rates,
    }


def frozen_room_rates(
    history: pd.DataFrame,
    *,
    season: int,
    week: int,
    team: str,
    prior_season: int,
    qb_regime_break: bool,
) -> dict:
    team = canon_team(team)
    parent_hist = history.loc[
        history["season"].eq(int(prior_season))
        | (history["season"].eq(int(season)) & history["week"].lt(int(week)))
    ].copy()
    if parent_hist.empty:
        raise RuntimeError(f"no parent history for {season} W{week}")

    parent = rate_block(parent_hist, team=team)
    if parent["rates"] is None:
        parent = rate_block(parent_hist)
        parent_source = "parent_league_fallback"
    else:
        parent_source = "parent_team_strict_prior"

    if not qb_regime_break:
        candidate = parent
        candidate_source = "stable_parent_history"
    else:
        current = history.loc[
            history["season"].eq(int(season))
            & history["week"].lt(int(week))
        ].copy()
        current_team = rate_block(current, team=team)
        if current_team["rates"] is not None:
            candidate = current_team
            candidate_source = "boundary_current_only"
        else:
            prior_league = history.loc[history["season"].eq(int(prior_season))].copy()
            candidate = rate_block(prior_league)
            if candidate["rates"] is None:
                raise RuntimeError(f"no prior league fallback for {season} W{week}")
            candidate_source = "boundary_prior_league_fallback"

    return {
        "parent": parent,
        "candidate": candidate,
        "parent_source": parent_source,
        "candidate_source": candidate_source,
    }


def universe_qb_keys(universe: pd.DataFrame, team: str) -> set[str]:
    u = universe.copy()
    u.columns = [str(c).strip().lower() for c in u.columns]
    if not {"team", "position", "player"}.issubset(u.columns):
        raise RuntimeError("pregame universe missing team/position/player")
    u["team"] = u["team"].map(canon_team)
    u["position"] = u["position"].fillna("").astype(str).str.upper().str.strip()
    if "player_clean_key" in u.columns:
        u["player_key"] = u["player_clean_key"].fillna("").astype(str)
    else:
        u["player_key"] = u["player"].map(player_key)
    return set(
        u.loc[u["team"].eq(canon_team(team)) & u["position"].map(is_qb), "player_key"]
        .astype(str)
        .loc[lambda s: s.ne("")]
    )


def forecast_week(
    *,
    player_logs: pd.DataFrame,
    team_weekly: pd.DataFrame,
    schedule: pd.DataFrame,
    universe: pd.DataFrame,
    injuries: pd.DataFrame,
    weather: pd.DataFrame,
    history: pd.DataFrame,
    prior_primary_qb: dict[tuple[int, str], str],
    season: int,
    week: int,
    prior_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    metrics = build_mc_predictions(bundle, iterations=20, seed=9700 + int(week))
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
    audits = []
    for team, tdf in explicit.groupby("team", sort=True):
        p = pd.to_numeric(tdf["mc_projected_plays"], errors="coerce").dropna().unique()
        if len(p) != 1:
            raise RuntimeError(f"{season} W{week} team={team} nonunique projected plays")
        opp = tdf["opponent"].dropna().astype(str).map(canon_team).unique()
        if len(opp) != 1:
            raise RuntimeError(f"{season} W{week} team={team} nonunique opponent")
        projected_plays = float(p[0])
        projected_dropbacks = projected_plays * FIXED_DROPBACK_RATE

        prior_qb = prior_primary_qb.get((int(prior_season), canon_team(team)), "")
        if not prior_qb:
            raise RuntimeError(f"missing prior primary QB {prior_season} team={team}")
        qbs = universe_qb_keys(universe, team)
        if not qbs:
            raise RuntimeError(f"missing pregame QB roster {season} W{week} team={team}")
        qb_break = prior_qb not in qbs

        info = frozen_room_rates(
            history,
            season=int(season),
            week=int(week),
            team=team,
            prior_season=int(prior_season),
            qb_regime_break=bool(qb_break),
        )
        cand_rates = info["candidate"]["rates"]
        parent_rates = info["parent"]["rates"]
        if cand_rates is None or parent_rates is None:
            raise RuntimeError("missing room rate")

        audits.append({
            "season": int(season),
            "week": int(week),
            "team": canon_team(team),
            "opponent": str(opp[0]),
            "prior_primary_qb_key": prior_qb,
            "prior_primary_qb_on_pregame_roster": int(not qb_break),
            "qb_regime_break": int(qb_break),
            "candidate_history_source": info["candidate_source"],
            "parent_history_source": info["parent_source"],
            "candidate_history_games": int(info["candidate"]["games"]),
            "candidate_history_plays": float(info["candidate"]["plays"]),
            "parent_history_games": int(info["parent"]["games"]),
            "parent_history_plays": float(info["parent"]["plays"]),
            "stable_rates_identical": int(
                qb_break or all(
                    abs(float(cand_rates[r]) - float(parent_rates[r])) <= 1e-12
                    for r in ROOMS
                )
            ),
        })

        for room_name in ROOMS:
            room_ent = float(
                tdf.loc[tdf["room"].eq(room_name), "entitlement_tgt_share"].sum()
            )
            if not np.isfinite(room_ent) or room_ent < 0 or room_ent > 0.95 + 1e-12:
                raise RuntimeError(
                    f"invalid room entitlement team={team} room={room_name} value={room_ent}"
                )
            cr = float(cand_rates[room_name])
            pr = float(parent_rates[room_name])
            rows.append({
                "season": int(season),
                "week": int(week),
                "team": canon_team(team),
                "opponent": str(opp[0]),
                "room": room_name,
                "projected_plays": projected_plays,
                "projected_dropbacks": projected_dropbacks,
                "baseline_room_entitlement": room_ent,
                "baseline_room_targets": projected_dropbacks * room_ent,
                "parent_room_targets_per_play_rate": pr,
                "parent_room_targets": projected_plays * pr,
                "candidate_room_targets_per_play_rate": cr,
                "candidate_room_targets": projected_plays * cr,
                "qb_regime_break": int(qb_break),
                "candidate_history_source": info["candidate_source"],
                "candidate_history_games": int(info["candidate"]["games"]),
                "candidate_history_plays": float(info["candidate"]["plays"]),
                "sum_candidate_room_rate": float(sum(cand_rates.values())),
            })

    out = pd.DataFrame(rows)
    audit = pd.DataFrame(audits)
    if out.empty or audit.empty:
        raise RuntimeError(f"{season} W{week} empty forecast/audit")
    return out, audit


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
        "median_ae": float(np.quantile(ae, 0.50)),
        "p75_ae": float(np.quantile(ae, 0.75)),
        "p90_ae": float(np.quantile(ae, 0.90)),
    }


def compare(df: pd.DataFrame, a: str, b: str) -> dict:
    sa, sb = score(df, a), score(df, b)
    actual = pd.to_numeric(df["actual_room_targets"], errors="coerce").to_numpy(float)
    ap = pd.to_numeric(df[a], errors="coerce").to_numpy(float)
    bp = pd.to_numeric(df[b], errors="coerce").to_numpy(float)
    changed = np.abs(ap - bp) > 1e-12
    aa, ba = np.abs(ap - actual), np.abs(bp - actual)
    bw = changed & (ba < aa - 1e-12)
    aw = changed & (aa < ba - 1e-12)
    decided = bw | aw
    return {
        "arm_a": sa,
        "arm_b": sb,
        "changed_rows": int(changed.sum()),
        "arm_b_closer": int(bw.sum()),
        "arm_a_closer": int(aw.sum()),
        "arm_b_closer_rate": float(bw.sum() / decided.sum()) if int(decided.sum()) else None,
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
    prior_primary_qb: dict[tuple[int, str], str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    details, audits = [], []
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

        pred, audit = forecast_week(
            player_logs=player_logs,
            team_weekly=team_weekly,
            schedule=schedule,
            universe=universe,
            injuries=inj,
            weather=wx,
            history=history,
            prior_primary_qb=prior_primary_qb,
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
        details.append(joined)
        audits.append(audit)
    return pd.concat(details, ignore_index=True), pd.concat(audits, ignore_index=True)


def room_block(d: pd.DataFrame) -> dict:
    return {
        room: {
            "baseline": score(d.loc[d["room"].eq(room)], "baseline_room_targets"),
            "parent": score(d.loc[d["room"].eq(room)], "parent_room_targets"),
            "candidate": score(d.loc[d["room"].eq(room)], "candidate_room_targets"),
        }
        for room in ROOMS
    }


def summarize(detail: pd.DataFrame) -> dict:
    out = {"by_season": {}, "pooled": room_block(detail)}
    for season in STAGE_SEASONS:
        out["by_season"][str(season)] = room_block(detail.loc[detail["season"].eq(season)])

    def macro(block: dict, metric: str, arm: str) -> float:
        return float(np.mean([block[r][arm][metric] for r in ROOMS]))

    for key in [str(s) for s in STAGE_SEASONS]:
        b = out["by_season"][key]
        b["macro"] = {
            f"{arm}_{metric}": macro(b, metric, arm)
            for arm in ("baseline", "parent", "candidate")
            for metric in ("mae", "p90_ae", "abs_bias")
        }
    b = out["pooled"]
    b["macro"] = {
        f"{arm}_{metric}": macro(b, metric, arm)
        for arm in ("baseline", "parent", "candidate")
        for metric in ("mae", "p90_ae", "abs_bias")
    }

    cmp_all = compare(detail, "baseline_room_targets", "candidate_room_targets")
    out["pooled"]["candidate_vs_baseline"] = cmp_all

    boundary = detail.loc[detail["qb_regime_break"].eq(1)].copy()
    out["boundary"] = {
        "rows": int(len(boundary)),
        "wr": compare(
            boundary.loc[boundary["room"].eq("WR")],
            "parent_room_targets",
            "candidate_room_targets",
        ),
        "all_rooms": compare(
            boundary,
            "parent_room_targets",
            "candidate_room_targets",
        ),
    }

    summed = (
        detail.groupby(["season", "week", "team"], as_index=False)
        .agg(
            actual_room_targets=("actual_room_targets", "sum"),
            baseline_room_targets=("baseline_room_targets", "sum"),
            parent_room_targets=("parent_room_targets", "sum"),
            candidate_room_targets=("candidate_room_targets", "sum"),
        )
    )
    out["summed_room"] = {
        "baseline": score(summed, "baseline_room_targets"),
        "parent": score(summed, "parent_room_targets"),
        "candidate": score(summed, "candidate_room_targets"),
    }
    return out


def build_gates(s: dict, detail: pd.DataFrame, audit: pd.DataFrame) -> dict:
    p = s["pooled"]
    y20, y21 = s["by_season"]["2020"], s["by_season"]["2021"]
    stable = detail.loc[detail["qb_regime_break"].eq(0)]
    stable_identical = bool(
        np.allclose(
            pd.to_numeric(stable["candidate_room_targets"], errors="coerce"),
            pd.to_numeric(stable["parent_room_targets"], errors="coerce"),
            rtol=0,
            atol=1e-12,
        )
    ) if len(stable) else False

    candidate_rates = pd.to_numeric(
        detail["candidate_room_targets_per_play_rate"], errors="coerce"
    )
    source_coverage = float(audit["prior_primary_qb_key"].fillna("").astype(str).ne("").mean())

    return {
        "pooled_macro_mae_improves_vs_baseline":
            p["macro"]["candidate_mae"] < p["macro"]["baseline_mae"],
        "macro_mae_2020_improves_vs_baseline":
            y20["macro"]["candidate_mae"] < y20["macro"]["baseline_mae"],
        "macro_mae_2021_improves_vs_baseline":
            y21["macro"]["candidate_mae"] < y21["macro"]["baseline_mae"],
        "pooled_wr_mae_improves_vs_baseline":
            p["WR"]["candidate"]["mae"] < p["WR"]["baseline"]["mae"],
        "wr_mae_2020_improves_vs_baseline":
            y20["WR"]["candidate"]["mae"] < y20["WR"]["baseline"]["mae"],
        "wr_mae_2021_improves_vs_baseline":
            y21["WR"]["candidate"]["mae"] < y21["WR"]["baseline"]["mae"],
        "pooled_te_mae_nonworse_vs_baseline":
            p["TE"]["candidate"]["mae"] <= p["TE"]["baseline"]["mae"] + 1e-12,
        "pooled_rbfb_mae_nonworse_vs_baseline":
            p["RB_FB"]["candidate"]["mae"] <= p["RB_FB"]["baseline"]["mae"] + 1e-12,
        "pooled_macro_p90_nonworse_vs_baseline":
            p["macro"]["candidate_p90_ae"] <= p["macro"]["baseline_p90_ae"] + 1e-12,
        "pooled_macro_abs_bias_nonworse_vs_baseline":
            p["macro"]["candidate_abs_bias"] <= p["macro"]["baseline_abs_bias"] + 1e-12,
        "summed_room_mae_improves_vs_baseline":
            s["summed_room"]["candidate"]["mae"] < s["summed_room"]["baseline"]["mae"],
        "summed_room_p90_nonworse_vs_baseline":
            s["summed_room"]["candidate"]["p90_ae"] <= s["summed_room"]["baseline"]["p90_ae"] + 1e-12,
        "candidate_closer_gt50":
            s["pooled"]["candidate_vs_baseline"]["arm_b_closer_rate"] is not None
            and s["pooled"]["candidate_vs_baseline"]["arm_b_closer_rate"] > 0.50,
        "boundary_wr_mae_better_than_parent":
            s["boundary"]["wr"]["arm_b"]["mae"] < s["boundary"]["wr"]["arm_a"]["mae"],
        "boundary_all_room_mae_better_than_parent":
            s["boundary"]["all_rooms"]["arm_b"]["mae"] < s["boundary"]["all_rooms"]["arm_a"]["mae"],
        "stable_regime_identical_to_parent": stable_identical,
        "target_game_outcomes_upstream_zero": True,
        "sportsbook_inputs_zero": True,
        "parameters_fit_zero": True,
        "candidate_variants_scored_one": True,
        "room_rates_finite_in_bounds":
            bool(candidate_rates.notna().all() and candidate_rates.between(0, 1).all()),
        "summed_room_rate_le1":
            float(pd.to_numeric(detail["sum_candidate_room_rate"], errors="coerce").max()) <= 1.0 + 1e-12,
        "qb_boundary_source_coverage_100": source_coverage >= 1.0 - 1e-12,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--team-weekly", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--universe-2020", type=Path, required=True)
    ap.add_argument("--universe-2021", type=Path, required=True)
    ap.add_argument("--injuries", type=Path, required=True)
    ap.add_argument("--weather", type=Path, required=True)
    ap.add_argument("--weeks-2020", default="1-17")
    ap.add_argument("--weeks-2021", default="1-18")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    logs = prep_logs(read(args.player_logs, "player logs"))
    team = read(args.team_weekly, "team weekly")
    sched = read(args.schedule, "schedule")
    injuries = optional(args.injuries)
    weather = optional(args.weather)
    history = build_room_history(logs, team)
    primary = build_prior_primary_qb(logs)

    d20, a20 = evaluate_season(
        season=2020, prior_season=2019, weeks=_parse_weeks(args.weeks_2020),
        player_logs=logs, team_weekly=team, schedule=sched,
        universe_dir=args.universe_2020, injuries=injuries, weather=weather,
        history=history, prior_primary_qb=primary,
    )
    d21, a21 = evaluate_season(
        season=2021, prior_season=2020, weeks=_parse_weeks(args.weeks_2021),
        player_logs=logs, team_weekly=team, schedule=sched,
        universe_dir=args.universe_2021, injuries=injuries, weather=weather,
        history=history, prior_primary_qb=primary,
    )
    detail = pd.concat([d20, d21], ignore_index=True)
    audit = pd.concat([a20, a21], ignore_index=True)

    s = summarize(detail)
    g = build_gates(s, detail, audit)
    qualified = all(g.values())
    disposition = (
        "OFFENSIVE_REGIME_BOUNDARY_ROOM_HISTORY_V1_STAGE_A_SUPPORTED"
        if qualified
        else "OFFENSIVE_REGIME_BOUNDARY_ROOM_HISTORY_V1_STAGE_A_FAILED_CLOSED"
    )

    season_rows = []
    for season in STAGE_SEASONS:
        b = s["by_season"][str(season)]
        season_rows.append({
            "season": season,
            "baseline_macro_mae": b["macro"]["baseline_mae"],
            "parent_macro_mae": b["macro"]["parent_mae"],
            "candidate_macro_mae": b["macro"]["candidate_mae"],
            "baseline_wr_mae": b["WR"]["baseline"]["mae"],
            "parent_wr_mae": b["WR"]["parent"]["mae"],
            "candidate_wr_mae": b["WR"]["candidate"]["mae"],
        })
    season_summary = pd.DataFrame(season_rows)

    boundary_rows = []
    for (season, qb_break, room), z in detail.groupby(
        ["season", "qb_regime_break", "room"], sort=True
    ):
        boundary_rows.append({
            "season": int(season),
            "qb_regime_break": int(qb_break),
            "room": str(room),
            "n": int(len(z)),
            "baseline_mae": score(z, "baseline_room_targets")["mae"],
            "parent_mae": score(z, "parent_room_targets")["mae"],
            "candidate_mae": score(z, "candidate_room_targets")["mae"],
        })
    boundary_summary = pd.DataFrame(boundary_rows)

    source_coverage = (
        audit.groupby(["season", "candidate_history_source"], as_index=False)
        .agg(
            team_games=("team", "size"),
            qb_break_rows=("qb_regime_break", "sum"),
            mean_history_games=("candidate_history_games", "mean"),
        )
    )

    payload = {
        "version": VERSION,
        "disposition": disposition,
        "qualified": bool(qualified),
        "candidate_variants_scored": 1,
        "parameters_fit": 0,
        "sportsbook_inputs_used": 0,
        "target_game_outcomes_used_upstream": 0,
        "production_mutations": 0,
        "stage": "A_UNTOUCHED_2020_2021_REVERSE_TIME_FALSIFICATION",
        "scorecard": s,
        "gates": g,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out_dir / "room_detail_2020_2021.csv", index=False)
    audit.to_csv(args.out_dir / "boundary_audit_2020_2021.csv", index=False)
    season_summary.to_csv(args.out_dir / "season_summary.csv", index=False)
    boundary_summary.to_csv(args.out_dir / "boundary_summary.csv", index=False)
    source_coverage.to_csv(args.out_dir / "source_coverage.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Offensive Regime Boundary Room History V1 — Stage A",
        "",
        f"Disposition: **{disposition}**",
        "",
        "- candidate variants scored: **1**",
        "- parameters fit: **0**",
        "- sportsbook inputs: **0**",
        "- target-game outcomes upstream: **0**",
        "",
        "## Season scorecard",
        "",
    ]
    for r in season_summary.itertuples(index=False):
        lines += [
            f"### {int(r.season)}",
            "",
            f"- macro MAE baseline -> parent -> candidate: {r.baseline_macro_mae:.6f} -> {r.parent_macro_mae:.6f} -> {r.candidate_macro_mae:.6f}",
            f"- WR MAE baseline -> parent -> candidate: {r.baseline_wr_mae:.6f} -> {r.parent_wr_mae:.6f} -> {r.candidate_wr_mae:.6f}",
            "",
        ]
    p = s["pooled"]
    lines += [
        "## Pooled",
        "",
        f"- macro MAE baseline -> parent -> candidate: {p['macro']['baseline_mae']:.6f} -> {p['macro']['parent_mae']:.6f} -> {p['macro']['candidate_mae']:.6f}",
        f"- WR MAE baseline -> parent -> candidate: {p['WR']['baseline']['mae']:.6f} -> {p['WR']['parent']['mae']:.6f} -> {p['WR']['candidate']['mae']:.6f}",
        f"- TE MAE baseline -> parent -> candidate: {p['TE']['baseline']['mae']:.6f} -> {p['TE']['parent']['mae']:.6f} -> {p['TE']['candidate']['mae']:.6f}",
        f"- RB_FB MAE baseline -> parent -> candidate: {p['RB_FB']['baseline']['mae']:.6f} -> {p['RB_FB']['parent']['mae']:.6f} -> {p['RB_FB']['candidate']['mae']:.6f}",
        f"- boundary WR parent -> candidate MAE: {s['boundary']['wr']['arm_a']['mae']:.6f} -> {s['boundary']['wr']['arm_b']['mae']:.6f}",
        f"- boundary all-room parent -> candidate MAE: {s['boundary']['all_rooms']['arm_a']['mae']:.6f} -> {s['boundary']['all_rooms']['arm_b']['mae']:.6f}",
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
