#!/usr/bin/env python3
"""Stage-B player targeted-route x opponent route-family matchup diagnostic.

Research only. Exact scoring contract is frozen in
PLAYER_TARGETED_ROUTE_MATCHUP_V1_STAGE_B_FREEZE.md.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.research.audit_player_targeted_route_matchup_source_v1 import (
    ROUTE_FAMILIES,
    SEASONS,
    SCORED_SEASONS,
    join_frames,
    run as run_source_audit,
    to_pandas,
)

VERSION = "PLAYER_TARGETED_ROUTE_MATCHUP_V1_STAGE_B"
MIN_PLAYER_GAMES = 4
MAX_PLAYER_GAMES = 8
MIN_DEF_ROUTE_ROWS = 100
MAX_DEF_GAMES = 8
MIN_DEF_ROUTE_FAMILY_ROWS = 10
BOOT_REPS = 2000
PLAYER_SEED = 2026101401
GAME_SEED = 2026101402
DEPTH_COVERAGE_FLOOR = 0.70
BH_Q = 0.10
POSITIONS = ("WR", "TE")
FAMILY_ORDER = (
    ("WR", "ypt"),
    ("WR", "catch"),
    ("WR", "rec_yards"),
    ("TE", "ypt"),
    ("TE", "catch"),
    ("TE", "rec_yards"),
)


def txt(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def roster_map(rosters: pd.DataFrame) -> pd.DataFrame:
    r = rosters.copy()
    r.columns = [str(c).strip().lower() for c in r.columns]
    need = {"season", "week", "gsis_id", "position"}
    if not need.issubset(r.columns):
        raise RuntimeError(f"weekly roster missing {sorted(need-set(r.columns))}")
    r = r[["season", "week", "gsis_id", "position"]].copy()
    r["season"] = num(r["season"]).astype("Int64")
    r["week"] = num(r["week"]).astype("Int64")
    r["gsis_id"] = txt(r["gsis_id"])
    r["position"] = txt(r["position"]).str.upper()
    return r.drop_duplicates(["season", "week", "gsis_id"], keep="last")


def build_target_games(pbp: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    b = pbp.copy()
    b.columns = [str(c).strip().lower() for c in b.columns]
    required = {
        "game_id", "season", "week", "posteam", "defteam",
        "receiver_player_id", "complete_pass", "yards_gained",
    }
    if not required.issubset(b.columns):
        raise RuntimeError(f"pbp missing {sorted(required-set(b.columns))}")

    if "season_type" in b.columns:
        b = b.loc[txt(b["season_type"]).str.upper().eq("REG")].copy()
    if "play_type" in b.columns:
        b = b.loc[txt(b["play_type"]).str.lower().eq("pass")].copy()
    if "pass_attempt" in b.columns:
        b = b.loc[num(b["pass_attempt"]).eq(1)].copy()

    b["receiver_id"] = txt(b["receiver_player_id"])
    b = b.loc[b["receiver_id"].ne("")].copy()
    b["season"] = num(b["season"]).astype("Int64")
    b["week"] = num(b["week"]).astype("Int64")
    b["posteam"] = txt(b["posteam"]).str.upper()
    b["defteam"] = txt(b["defteam"]).str.upper()
    b["game_id"] = txt(b["game_id"])
    b["complete"] = num(b["complete_pass"]).fillna(0.0).clip(0, 1)
    yg = num(b["yards_gained"]).fillna(0.0)
    b["rec_yards_play"] = np.where(b["complete"].eq(1.0), yg, 0.0)
    if "air_yards" in b.columns:
        b["air_yards_num"] = num(b["air_yards"])
    else:
        b["air_yards_num"] = np.nan

    rr = roster_map(rosters)
    b = b.merge(
        rr,
        left_on=["season", "week", "receiver_id"],
        right_on=["season", "week", "gsis_id"],
        how="left",
        validate="many_to_one",
    )
    b["position"] = txt(b["position"]).str.upper()
    b = b.loc[b["position"].isin(POSITIONS)].copy()

    g = (
        b.groupby(
            ["season", "week", "game_id", "posteam", "defteam", "receiver_id", "position"],
            as_index=False,
        )
        .agg(
            targets=("receiver_id", "size"),
            receptions=("complete", "sum"),
            rec_yards=("rec_yards_play", "sum"),
            air_yards_sum=("air_yards_num", "sum"),
            air_yards_n=("air_yards_num", "count"),
        )
    )
    g["ypt"] = g["rec_yards"] / g["targets"].clip(lower=1)
    g["catch_rate"] = g["receptions"] / g["targets"].clip(lower=1)
    g["mean_air_yards"] = np.where(g["air_yards_n"] > 0, g["air_yards_sum"] / g["air_yards_n"], np.nan)
    g["ord"] = g["season"].astype(int) * 100 + g["week"].astype(int)
    return g.sort_values(["ord", "game_id", "receiver_id"]).reset_index(drop=True)


def build_route_games(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = joined.copy()
    x = x.loc[
        x["valid_route_taxonomy"]
        & x["valid_receiver_identity"]
        & x["valid_wr_te"]
        & x["season"].isin(SEASONS)
        & x["week"].notna()
        & x["posteam"].ne("")
        & x["defteam"].ne("")
    ].copy()
    x["complete"] = num(x.get("complete_pass", pd.Series(0, index=x.index))).fillna(0).clip(0, 1)
    yg = num(x.get("yards_gained", pd.Series(0, index=x.index))).fillna(0)
    x["rec_yards_play"] = np.where(x["complete"].eq(1.0), yg, 0.0)
    x["game_id2"] = txt(x["game_id"] if "game_id" in x.columns else x["nflverse_game_id"])
    x["ord"] = x["season"].astype(int) * 100 + x["week"].astype(int)

    player_game = (
        x.groupby(
            ["season", "week", "ord", "game_id2", "posteam", "defteam", "receiver_id", "position"],
            as_index=False,
        )["route_family"]
        .value_counts()
        .rename(columns={"count": "route_count"})
    )

    defense_game = (
        x.groupby(
            ["season", "week", "ord", "game_id2", "defteam", "route_family"],
            as_index=False,
        )
        .agg(
            n=("receiver_id", "size"),
            completions=("complete", "sum"),
            rec_yards=("rec_yards_play", "sum"),
        )
    )
    return player_game, defense_game


def append_2022_histories(
    player_route_hist: dict,
    player_outcome_hist: dict,
    defense_hist: dict,
    league_stats: dict,
    route_pg: pd.DataFrame,
    def_g: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    for (_, _, _, game_id, _, _, pid, _), g in route_pg.loc[route_pg["season"].eq(2022)].groupby(
        ["season", "week", "ord", "game_id2", "posteam", "defteam", "receiver_id", "position"], sort=True
    ):
        counts = dict(zip(g["route_family"].astype(str), g["route_count"].astype(float)))
        player_route_hist[pid].append({"game_id": game_id, "counts": counts})
    for r in outcomes.loc[outcomes["season"].eq(2022)].itertuples(index=False):
        player_outcome_hist[r.receiver_id].append({
            "game_id": r.game_id, "ypt": float(r.ypt), "catch": float(r.catch_rate),
            "rec_yards": float(r.rec_yards), "air_sum": float(r.air_yards_sum),
            "air_n": int(r.air_yards_n),
        })
    for (game_id, dteam), g in def_g.loc[def_g["season"].eq(2022)].groupby(["game_id2", "defteam"], sort=True):
        stats = {
            str(r.route_family): {"n": float(r.n), "comp": float(r.completions), "yards": float(r.rec_yards)}
            for r in g.itertuples(index=False)
        }
        defense_hist[dteam].append({"game_id": game_id, "stats": stats})
        for fam, st in stats.items():
            z = league_stats[fam]
            z["n"] += st["n"]; z["comp"] += st["comp"]; z["yards"] += st["yards"]


def aggregate_route_mix(entries: list[dict]) -> dict[str, float]:
    counts = defaultdict(float)
    for e in entries[-MAX_PLAYER_GAMES:]:
        for fam, n in e["counts"].items():
            counts[fam] += float(n)
    total = sum(counts.values())
    return {fam: n / total for fam, n in counts.items()} if total > 0 else {}


def aggregate_player_baseline(entries: list[dict]) -> dict | None:
    use = entries[-MAX_PLAYER_GAMES:]
    if len(use) < MIN_PLAYER_GAMES:
        return None
    ypt = np.array([e["ypt"] for e in use], dtype=float)
    catch = np.array([e["catch"] for e in use], dtype=float)
    recy = np.array([e["rec_yards"] for e in use], dtype=float)
    air_sum = sum(float(e["air_sum"]) for e in use)
    air_n = sum(int(e["air_n"]) for e in use)
    return {
        "ypt": float(np.mean(ypt)),
        "catch": float(np.mean(catch)),
        "rec_yards": float(np.mean(recy)),
        "air_yards": float(air_sum / air_n) if air_n > 0 else np.nan,
    }


def aggregate_defense(entries: list[dict]) -> tuple[dict[str, dict], int]:
    use = entries[-MAX_DEF_GAMES:]
    out = defaultdict(lambda: {"n": 0.0, "comp": 0.0, "yards": 0.0})
    total = 0
    for e in use:
        for fam, st in e["stats"].items():
            out[fam]["n"] += st["n"]; out[fam]["comp"] += st["comp"]; out[fam]["yards"] += st["yards"]
            total += int(st["n"])
    return dict(out), total


def matchup_features(mix: dict, defense: dict, league: dict) -> tuple[float, float] | None:
    if not mix:
        return None
    dypt = 0.0
    dcatch = 0.0
    used = 0.0
    for fam, w in mix.items():
        l = league.get(fam)
        if not l or l["n"] <= 0:
            continue
        lypt = l["yards"] / l["n"]
        lcatch = l["comp"] / l["n"]
        d = defense.get(fam, {"n": 0.0, "comp": 0.0, "yards": 0.0})
        if d["n"] >= MIN_DEF_ROUTE_FAMILY_ROWS:
            dypt_f = d["yards"] / d["n"]
            dcatch_f = d["comp"] / d["n"]
        else:
            dypt_f = lypt
            dcatch_f = lcatch
        dypt += w * (dypt_f - lypt)
        dcatch += w * (dcatch_f - lcatch)
        used += w
    if used <= 0:
        return None
    return float(dypt / used), float(dcatch / used)


def build_casebook(route_pg: pd.DataFrame, def_g: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    player_route_hist = defaultdict(list)
    player_outcome_hist = defaultdict(list)
    defense_hist = defaultdict(list)
    league_stats = defaultdict(lambda: {"n": 0.0, "comp": 0.0, "yards": 0.0})
    append_2022_histories(player_route_hist, player_outcome_hist, defense_hist, league_stats, route_pg, def_g, outcomes)

    rows = []
    for season in SCORED_SEASONS:
        weeks = sorted(outcomes.loc[outcomes["season"].eq(season), "week"].dropna().astype(int).unique())
        for week in weeks:
            week_out = outcomes.loc[outcomes["season"].eq(season) & outcomes["week"].eq(week)].copy()
            for r in week_out.itertuples(index=False):
                pr = player_route_hist[r.receiver_id]
                po = player_outcome_hist[r.receiver_id]
                if len(pr) < MIN_PLAYER_GAMES or len(po) < MIN_PLAYER_GAMES:
                    continue
                mix = aggregate_route_mix(pr)
                baseline = aggregate_player_baseline(po)
                if baseline is None:
                    continue
                dstat, dtotal = aggregate_defense(defense_hist[r.defteam])
                if dtotal < MIN_DEF_ROUTE_ROWS:
                    continue
                mf = matchup_features(mix, dstat, league_stats)
                if mf is None:
                    continue
                dypt, dcatch = mf
                rows.append({
                    "season": int(season), "week": int(week), "game_id": str(r.game_id),
                    "team": str(r.posteam), "opponent": str(r.defteam),
                    "player_id": str(r.receiver_id), "position": str(r.position),
                    "player_prior_route_games": int(len(pr[-MAX_PLAYER_GAMES:])),
                    "player_prior_outcome_games": int(len(po[-MAX_PLAYER_GAMES:])),
                    "opponent_prior_route_rows": int(dtotal),
                    "matchup_delta_ypt": dypt,
                    "matchup_delta_catch": dcatch,
                    "prior_player_ypt": baseline["ypt"],
                    "prior_player_catch": baseline["catch"],
                    "prior_player_rec_yards": baseline["rec_yards"],
                    "prior_player_air_yards": baseline["air_yards"],
                    "actual_ypt": float(r.ypt),
                    "actual_catch": float(r.catch_rate),
                    "actual_rec_yards": float(r.rec_yards),
                    "ypt_residual": float(r.ypt - baseline["ypt"]),
                    "catch_residual": float(r.catch_rate - baseline["catch"]),
                    "rec_yards_residual": float(r.rec_yards - baseline["rec_yards"]),
                })

            # Only after the entire target week is scored can that week become history.
            wroute = route_pg.loc[route_pg["season"].eq(season) & route_pg["week"].eq(week)]
            for (_, _, _, game_id, _, _, pid, _), g in wroute.groupby(
                ["season", "week", "ord", "game_id2", "posteam", "defteam", "receiver_id", "position"], sort=True
            ):
                counts = dict(zip(g["route_family"].astype(str), g["route_count"].astype(float)))
                player_route_hist[pid].append({"game_id": game_id, "counts": counts})
            for r in week_out.itertuples(index=False):
                player_outcome_hist[r.receiver_id].append({
                    "game_id": r.game_id, "ypt": float(r.ypt), "catch": float(r.catch_rate),
                    "rec_yards": float(r.rec_yards), "air_sum": float(r.air_yards_sum),
                    "air_n": int(r.air_yards_n),
                })
            wd = def_g.loc[def_g["season"].eq(season) & def_g["week"].eq(week)]
            for (game_id, dteam), g in wd.groupby(["game_id2", "defteam"], sort=True):
                stats = {
                    str(z.route_family): {"n": float(z.n), "comp": float(z.completions), "yards": float(z.rec_yards)}
                    for z in g.itertuples(index=False)
                }
                defense_hist[dteam].append({"game_id": game_id, "stats": stats})
                for fam, st in stats.items():
                    q = league_stats[fam]
                    q["n"] += st["n"]; q["comp"] += st["comp"]; q["yards"] += st["yards"]

    return pd.DataFrame(rows).sort_values(["season", "week", "game_id", "player_id"]).reset_index(drop=True)


def rho(x, y) -> float:
    a = np.asarray(x, dtype=float); b = np.asarray(y, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return np.nan
    return float(spearmanr(a[mask], b[mask]).statistic)


def cluster_boot(q: pd.DataFrame, xcol: str, ycol: str, cluster: str, seed: int) -> dict:
    gmap = {k: v.index.to_numpy() for k, v in q.groupby(cluster)}
    keys = np.array(list(gmap.keys()), dtype=object)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(BOOT_REPS):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([gmap[k] for k in sampled])
        v = rho(q.loc[idx, xcol], q.loc[idx, ycol])
        if np.isfinite(v):
            vals.append(v)
    arr = np.asarray(vals, dtype=float)
    return {
        "reps": int(len(arr)),
        "observed": rho(q[xcol], q[ycol]),
        "ci_low": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
        "ci_high": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
        "p_one_sided": float((1 + np.sum(arr <= 0)) / (1 + len(arr))) if len(arr) else 1.0,
    }


def residualize_depth(q: pd.DataFrame, xcol: str, ycol: str) -> tuple[pd.DataFrame, float]:
    z = q.loc[
        np.isfinite(num(q[xcol])) & np.isfinite(num(q[ycol])) & np.isfinite(num(q["prior_player_air_yards"]))
    ].copy()
    coverage = len(z) / len(q) if len(q) else 0.0
    if len(z) < 3:
        z["x_depth_resid"] = np.nan; z["y_depth_resid"] = np.nan
        return z, coverage
    a = np.column_stack([np.ones(len(z)), z["prior_player_air_yards"].to_numpy(float)])
    bx, *_ = np.linalg.lstsq(a, z[xcol].to_numpy(float), rcond=None)
    by, *_ = np.linalg.lstsq(a, z[ycol].to_numpy(float), rcond=None)
    z["x_depth_resid"] = z[xcol].to_numpy(float) - a @ bx
    z["y_depth_resid"] = z[ycol].to_numpy(float) - a @ by
    return z, coverage


def player_diagnostics(q: pd.DataFrame, xcol: str, ycol: str) -> dict:
    vals = []
    for pid, g in q.groupby("player_id"):
        if len(g) < 6:
            continue
        v = rho(g[xcol], g[ycol])
        if np.isfinite(v):
            vals.append(v)
    arr = np.asarray(vals, dtype=float)
    return {
        "qualifying_players_ge6": int(len(arr)),
        "median_per_player_rho": float(np.median(arr)) if len(arr) else np.nan,
        "fraction_player_rho_positive": float(np.mean(arr > 0)) if len(arr) else np.nan,
    }


def bh_adjust(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for rank_rev, idx in enumerate(order[::-1], start=1):
        rank = n - rank_rev + 1
        val = min(prev, p[idx] * n / rank)
        q[idx] = val
        prev = val
    return q.tolist()


def evaluate(casebook: pd.DataFrame) -> dict:
    families = []
    pvals = []
    for pos, fam in FAMILY_ORDER:
        xcol = "matchup_delta_catch" if fam == "catch" else "matchup_delta_ypt"
        ycol = {"ypt": "ypt_residual", "catch": "catch_residual", "rec_yards": "rec_yards_residual"}[fam]
        q = casebook.loc[casebook["position"].eq(pos)].copy()
        season = {str(s): {"n": int(len(q.loc[q["season"].eq(s)])), "rho": rho(q.loc[q["season"].eq(s), xcol], q.loc[q["season"].eq(s), ycol])} for s in SCORED_SEASONS}
        conf = q.loc[q["season"].isin([2024, 2025])].copy()
        pb = cluster_boot(conf, xcol, ycol, "player_id", PLAYER_SEED)
        gb = cluster_boot(conf, xcol, ycol, "game_id", GAME_SEED)

        depth, depth_cov = residualize_depth(conf, xcol, ycol)
        dpb = cluster_boot(depth, "x_depth_resid", "y_depth_resid", "player_id", PLAYER_SEED) if len(depth) else {"observed":np.nan,"ci_low":np.nan,"ci_high":np.nan,"p_one_sided":1.0,"reps":0}
        dgb = cluster_boot(depth, "x_depth_resid", "y_depth_resid", "game_id", GAME_SEED) if len(depth) else {"observed":np.nan,"ci_low":np.nan,"ci_high":np.nan,"p_one_sided":1.0,"reps":0}
        pdg = player_diagnostics(conf, xcol, ycol)
        famres = {
            "position": pos, "family": fam, "x": xcol, "y": ycol,
            "season": season,
            "confirmation_rows": int(len(conf)),
            "confirmation_players": int(conf["player_id"].nunique()),
            "pooled_rho": rho(conf[xcol], conf[ycol]),
            "player_cluster": pb,
            "game_cluster": gb,
            "depth_control_coverage": float(depth_cov),
            "depth_control_player_cluster": dpb,
            "depth_control_game_cluster": dgb,
            "player_diagnostics": pdg,
        }
        families.append(famres)
        pvals.append(pb["p_one_sided"])

    qvals = bh_adjust(pvals)
    any_confirmed = False
    for famres, qv in zip(families, qvals):
        famres["bh_q"] = float(qv)
        s24 = famres["season"]["2024"]["rho"]
        s25 = famres["season"]["2025"]["rho"]
        depth_limited = famres["depth_control_coverage"] < DEPTH_COVERAGE_FLOOR
        gates = {
            "2024_rho_positive": bool(np.isfinite(s24) and s24 > 0),
            "2025_rho_positive": bool(np.isfinite(s25) and s25 > 0),
            "pooled_rho_positive": bool(np.isfinite(famres["pooled_rho"]) and famres["pooled_rho"] > 0),
            "player_ci_low_positive": bool(famres["player_cluster"]["ci_low"] > 0),
            "game_ci_low_positive": bool(famres["game_cluster"]["ci_low"] > 0),
            "depth_coverage_ge_70pct": bool(not depth_limited),
            "depth_rho_positive": bool(famres["depth_control_player_cluster"]["observed"] > 0),
            "depth_player_ci_low_positive": bool(famres["depth_control_player_cluster"]["ci_low"] > 0),
            "depth_game_ci_low_positive": bool(famres["depth_control_game_cluster"]["ci_low"] > 0),
            "bh_q_le_010": bool(qv <= BH_Q),
        }
        famres["gates"] = gates
        if depth_limited:
            disp = "DEPTH_CONTROL_SOURCE_LIMITED"
        elif all(gates.values()):
            disp = "PLAYER_TARGETED_ROUTE_MATCHUP_FAMILY_CONFIRMED"
            any_confirmed = True
        else:
            disp = "NO_ACTIONABLE_PLAYER_TARGETED_ROUTE_MATCHUP_FAMILY"
        famres["disposition"] = disp
        famres["failed_gates"] = [k for k,v in gates.items() if not v]

    return {
        "version": VERSION,
        "disposition": "PLAYER_TARGETED_ROUTE_MATCHUP_CONFIRMED" if any_confirmed else "NO_ACTIONABLE_PLAYER_TARGETED_ROUTE_MATCHUP",
        "families": families,
        "casebook_rows": int(len(casebook)),
        "casebook_players": int(casebook["player_id"].nunique()) if len(casebook) else 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "outcomes_2026_read": 0,
        "production_changed": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import nflreadpy as nfl
    participation = to_pandas(nfl.load_participation(seasons=list(SEASONS)))
    pbp = to_pandas(nfl.load_pbp(seasons=list(SEASONS)))
    rosters = to_pandas(nfl.load_rosters_weekly(seasons=list(SEASONS)))

    source_result, _ = run_source_audit(participation, pbp, rosters)
    if source_result["disposition"] != "PLAYER_TARGETED_ROUTE_MATCHUP_SOURCE_READY":
        raise RuntimeError(f"Stage A not ready: {source_result['disposition']}")

    joined, _ = join_frames(participation, pbp, rosters)
    route_pg, def_g = build_route_games(joined)
    outcomes = build_target_games(pbp, rosters)
    casebook = build_casebook(route_pg, def_g, outcomes)
    result = evaluate(casebook)

    casebook.to_csv(args.out_dir / "player_targeted_route_matchup_casebook.csv", index=False)
    (args.out_dir / "player_targeted_route_matchup_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    pd.DataFrame([
        {
            "position": f["position"], "family": f["family"], "disposition": f["disposition"],
            "rho_2023": f["season"]["2023"]["rho"], "rho_2024": f["season"]["2024"]["rho"],
            "rho_2025": f["season"]["2025"]["rho"], "pooled_rho": f["pooled_rho"],
            "player_ci_low": f["player_cluster"]["ci_low"], "game_ci_low": f["game_cluster"]["ci_low"],
            "depth_rho": f["depth_control_player_cluster"]["observed"],
            "depth_player_ci_low": f["depth_control_player_cluster"]["ci_low"],
            "depth_game_ci_low": f["depth_control_game_cluster"]["ci_low"],
            "bh_q": f["bh_q"], "confirmation_rows": f["confirmation_rows"],
            "confirmation_players": f["confirmation_players"],
            "median_per_player_rho": f["player_diagnostics"]["median_per_player_rho"],
            "fraction_player_rho_positive": f["player_diagnostics"]["fraction_player_rho_positive"],
        }
        for f in result["families"]
    ]).to_csv(args.out_dir / "player_targeted_route_matchup_family_summary.csv", index=False)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
