#!/usr/bin/env python3
"""Frozen Joint Pass/Receiving Conservation V1 trace.

Runs exactly four predeclared architectures:
  B0_CURRENT
  C1_GROUP_ONLY
  C2_CONSERVATION_ONLY
  C3_JOINT

Target-week outcomes are exported only after projection arrays are built and are
never used as candidate inputs. Sportsbook inputs are not used.
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
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from scripts.modeling import simulation_rules
from scripts import simulation_v2

PASS_CATCHER_POSITIONS = {"WR", "LWR", "RWR", "SWR", "TE", "RB", "FB"}
WR_POSITIONS = {"WR", "LWR", "RWR", "SWR"}
GROUPS = ("WR", "TE", "RB_FB")
TEAM_ALIAS = {"JAC": "JAX", "JAX": "JAX", "LA": "LAR", "LAR": "LAR"}
PSEUDO_TARGETS = 105.0
RESIDUAL_CATCH_RATE = 0.64
RESIDUAL_YPT = 7.5
YPR_MIN = 3.0
YPR_MAX = 35.0


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing required input: {path}")
    return pd.read_csv(path, low_memory=False)


def opt(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() and path.stat().st_size else pd.DataFrame()


def num(s):
    return pd.to_numeric(s, errors="coerce")


def finite(value, default=0.0) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def canon_team(v) -> str:
    raw = str(v or "").strip().upper()
    return TEAM_ALIAS.get(raw, raw)


def position_group(position) -> str:
    pos = str(position or "").upper().strip()
    if pos in WR_POSITIONS:
        return "WR"
    if pos == "TE":
        return "TE"
    if pos == "RB":
        return "RB"
    if pos == "FB":
        return "FB"
    return "OTHER"


def mass_group(position) -> str:
    g = position_group(position)
    return "RB_FB" if g in {"RB", "FB"} else g


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, got {len(hits)}")
    return hits[0]


def load_m89(root: Path) -> pd.DataFrame:
    q = pd.read_csv(one(root, "m89_2024_2025_synthesis_trace.csv"), low_memory=False)
    q.columns = [str(c).strip().lower() for c in q.columns]
    q = q.loc[num(q["season"]).isin([2024, 2025])].copy()
    q["season"] = num(q["season"]).astype(int)
    q["week"] = num(q["week"]).astype(int)
    q["team"] = q["team"].map(canon_team)
    q["football_synthesis"] = num(q["football_synthesis"])
    q["actual_pass_yards"] = num(q["actual_pass_yards"])
    if q.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate M89 season/week/team rows")
    return q


def prepared(bundle) -> pd.DataFrame:
    m = cp.build_market_frame(bundle)
    m = apply_bayesian_to_metrics(m, build_bayesian_baseline(bundle.player_consensus))
    with patch.object(
        simulation_rules,
        "load_model_contexts",
        return_value=(bundle.teams, bundle.players),
    ):
        m = simulation_rules.apply_rules_to_metrics(m)
    m["player_clean_key"] = m["player_clean_key"].fillna("").astype(str)
    return m.copy()


def unique_players(metrics: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["event_id", "team", "player_clean_key"]
    return (
        metrics.sort_values(key_cols)
        .drop_duplicates(key_cols, keep="last")
        .reset_index(drop=True)
    )


def baseline_probabilities(team_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    raw = np.array(
        [
            simulation_v2._num(
                r,
                "rules_tgt_share",
                "bayes_tgt_share",
                "target_share",
                "tgt_share",
                default=0.0,
            )
            for _, r in team_df.iterrows()
        ],
        dtype=float,
    )
    sharpened = simulation_v2._sharpen_wr_target_shares(team_df, raw)
    clean = np.clip(
        np.nan_to_num(np.asarray(sharpened, dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        0.95,
    )
    if float(clean.sum()) > 0.95:
        clean *= 0.95 / float(clean.sum())
    return clean, max(0.0, 1.0 - float(clean.sum()))


def historical_group_shares(logs: pd.DataFrame, season: int, week: int, team: str) -> tuple[dict[str, float] | None, dict]:
    x = logs.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    for c in ("season", "week", "targets"):
        if c not in x.columns:
            raise RuntimeError(f"historical logs missing {c}")
    x["season"] = num(x["season"])
    x["week"] = num(x["week"])
    x["targets"] = num(x["targets"]).fillna(0.0)
    x["team"] = x["team"].map(canon_team)
    x["position"] = x.get("position", "").fillna("").astype(str).str.upper().str.strip()
    x["mass_group"] = x["position"].map(mass_group)
    past = x.loc[
        (x["season"] < int(season))
        | ((x["season"] == int(season)) & (x["week"] < int(week)))
    ].copy()
    if past.empty:
        return None, {"reason": "no_past_rows"}

    game_tot = (
        past.groupby(["season", "week", "team"], as_index=False)["targets"]
        .sum()
        .rename(columns={"targets": "team_targets"})
    )
    group_tot = (
        past.loc[past["mass_group"].isin(GROUPS)]
        .groupby(["season", "week", "team", "mass_group"], as_index=False)["targets"]
        .sum()
    )

    tg = game_tot.loc[game_tot["team"].eq(canon_team(team))].sort_values(["season", "week"], ascending=[False, False]).head(8)
    if tg.empty or float(tg["team_targets"].sum()) <= 0:
        return None, {"reason": "no_team_target_history"}
    keys = tg[["season", "week", "team"]]
    th = group_tot.merge(keys, on=["season", "week", "team"], how="inner")
    team_total = float(tg["team_targets"].sum())
    team_group = th.groupby("mass_group")["targets"].sum().to_dict()

    league = past.loc[past["season"].isin([int(season) - 1, int(season)])].copy()
    league_total = float(league["targets"].sum())
    if league_total <= 0:
        return None, {"reason": "zero_league_prior_targets"}
    league_group = (
        league.loc[league["mass_group"].isin(GROUPS)]
        .groupby("mass_group")["targets"]
        .sum()
        .to_dict()
    )

    shrunk = {}
    for g in GROUPS:
        lg = float(league_group.get(g, 0.0)) / league_total
        shrunk[g] = (float(team_group.get(g, 0.0)) + PSEUDO_TARGETS * lg) / (team_total + PSEUDO_TARGETS)
    denom = float(sum(shrunk.values()))
    if not np.isfinite(denom) or denom <= 0:
        return None, {"reason": "invalid_shrunk_denominator"}
    shares = {g: float(shrunk[g] / denom) for g in GROUPS}
    audit = {
        "reason": "ok",
        "team_history_games": int(len(tg)),
        "team_history_targets": team_total,
        "league_prior_targets": league_total,
        **{f"history_share_{g.lower()}": shares[g] for g in GROUPS},
    }
    return shares, audit


def calibrated_probabilities(team_df: pd.DataFrame, base: np.ndarray, hist: dict[str, float] | None) -> tuple[np.ndarray, dict]:
    out = np.asarray(base, dtype=float).copy()
    positions = team_df.get("position", pd.Series("", index=team_df.index)).fillna("").astype(str).str.upper().to_numpy()
    group_labels = np.array([mass_group(v) for v in positions], dtype=object)
    pass_mask = np.isin(positions, list(PASS_CATCHER_POSITIONS))
    modeled_mass = float(out[pass_mask].sum())
    audit = {"modeled_receiver_mass_b0": modeled_mass, "unassignable_mass": 0.0, "calibration_fallback": hist is None}
    if hist is None or modeled_mass <= 0:
        audit["modeled_receiver_mass_candidate"] = modeled_mass
        return out, audit

    candidate = out.copy()
    candidate[pass_mask] = 0.0
    unassignable = 0.0
    for g in GROUPS:
        idx = np.flatnonzero(group_labels == g)
        target_mass = modeled_mass * float(hist[g])
        denom = float(base[idx].sum()) if len(idx) else 0.0
        if len(idx) and denom > 0:
            candidate[idx] = target_mass * (base[idx] / denom)
        else:
            unassignable += target_mass
    # Any unassignable receiver mass becomes residual. Non-pass-catcher B0 mass is untouched.
    audit["unassignable_mass"] = float(unassignable)
    audit["modeled_receiver_mass_candidate"] = float(candidate[pass_mask].sum())
    return candidate, audit


def apply_probability_map(metrics: pd.DataFrame, mapping: dict[tuple[str, str, str], float]) -> pd.DataFrame:
    out = metrics.copy()
    vals = []
    for _, r in out.iterrows():
        key = (str(r.get("event_id", "")), str(r.get("team", "")), str(r.get("player_clean_key", "")))
        vals.append(mapping.get(key, 0.0))
    out["rules_tgt_share"] = np.asarray(vals, dtype=float)
    return out


def sim_arr(sim, game, pkey, market):
    a = sim.values.get((str(game), str(pkey), str(market)))
    return a if a is not None and len(a) else None


def primary_qb_arrays(sim, team_df: pd.DataFrame, game) -> tuple[str | None, np.ndarray | None]:
    candidates = []
    for _, r in team_df.iterrows():
        pos = str(r.get("position", "") or "").upper().strip()
        role = str(r.get("model_role", r.get("role", "")) or "").upper()
        if pos != "QB" and not role.startswith("QB"):
            continue
        pkey = str(r.get("player_clean_key", "") or "")
        a = sim_arr(sim, game, pkey, "pass_yards")
        if a is None:
            continue
        candidates.append((finite(r.get("qb_projection_eligible"), 0.0), finite(r.get("qb_role_score"), 0.0), pkey, a))
    if not candidates:
        return None, None
    candidates.sort(key=lambda z: (z[0], z[1]), reverse=True)
    return candidates[0][2], np.asarray(candidates[0][3], dtype=float)


def sample_crps(samples: np.ndarray, actual: float) -> float:
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x) or not np.isfinite(actual):
        return np.nan
    x.sort()
    n = len(x)
    first = float(np.mean(np.abs(x - float(actual))))
    # E|X-X'| over ordered pairs, including diagonal, via sorted-sample identity.
    weights = 2.0 * np.arange(1, n + 1, dtype=float) - n - 1.0
    pair = float(2.0 * np.sum(weights * x) / (n * n))
    return first - 0.5 * pair


def distribution_stats(samples: np.ndarray, actual: float, prefix: str) -> dict:
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {f"{prefix}_mean": np.nan, f"{prefix}_crps": np.nan}
    q = np.quantile(x, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_p05": float(q[0]),
        f"{prefix}_p10": float(q[1]),
        f"{prefix}_p25": float(q[2]),
        f"{prefix}_p50": float(q[3]),
        f"{prefix}_p75": float(q[4]),
        f"{prefix}_p90": float(q[5]),
        f"{prefix}_p95": float(q[6]),
        f"{prefix}_crps": sample_crps(x, actual),
        f"{prefix}_cover50": int(np.isfinite(actual) and q[2] <= actual <= q[4]),
        f"{prefix}_cover80": int(np.isfinite(actual) and q[1] <= actual <= q[5]),
        f"{prefix}_cover90": int(np.isfinite(actual) and q[0] <= actual <= q[6]),
    }


def conserved_receiving(
    players: pd.DataFrame,
    probability_map: dict[tuple[str, str, str], float],
    *,
    iterations: int,
    seed: int,
    anchor_map: dict[tuple[str, str], float],
) -> tuple[dict[tuple[str, str, str], np.ndarray], dict[tuple[str, str], np.ndarray], list[dict], int]:
    """Generate receiver-derived team passing yards with exact per-iteration conservation."""
    rng = np.random.default_rng(int(seed))
    receiver_values: dict[tuple[str, str, str], np.ndarray] = {}
    qb_values: dict[tuple[str, str], np.ndarray] = {}
    conservation_rows: list[dict] = []
    zero_rec_positive_yards = 0

    for game, game_df in players.groupby("event_id", dropna=False, sort=False):
        game_pace_shock = rng.normal(0.0, 2.0, int(iterations))
        for team, team_df in game_df.groupby("team", dropna=False, sort=False):
            if pd.isna(team) or not str(team).strip():
                continue
            team_df = team_df.reset_index(drop=True)
            plays_mean, pass_rate_mean = simulation_v2._team_inputs(team_df)
            plays = np.rint(np.clip(rng.normal(plays_mean, 3.5, iterations) + game_pace_shock, 45, 85)).astype(int)
            pass_rate = np.clip(rng.normal(pass_rate_mean, 0.035, iterations), 0.25, 0.82)
            pass_att = rng.binomial(plays, pass_rate)
            pass_eff_shock = np.clip(rng.normal(1.0, 0.09, iterations), 0.65, 1.35)

            shares = np.zeros(len(team_df), dtype=float)
            pc_mask = np.zeros(len(team_df), dtype=bool)
            for j, (_, r) in enumerate(team_df.iterrows()):
                pos = str(r.get("position", "") or "").upper().strip()
                pc_mask[j] = pos in PASS_CATCHER_POSITIONS
                key = (str(game), str(team), str(r.get("player_clean_key", "")))
                shares[j] = float(probability_map.get(key, 0.0)) if pc_mask[j] else 0.0
            targets = simulation_v2._allocate_counts(rng, pass_att, shares)
            residual_targets = np.maximum(0, pass_att - targets.sum(axis=1))

            raw_arrays: dict[str, np.ndarray] = {}
            for j, (_, r) in enumerate(team_df.iterrows()):
                if not pc_mask[j]:
                    continue
                pkey = str(r.get("player_clean_key", "") or "")
                catch_rate = simulation_v2._clip_prob(
                    simulation_v2._num(r, "rules_catch_rate", "bayes_receptions_per_target", "receptions_per_target", "catch_rate", default=RESIDUAL_CATCH_RATE),
                    RESIDUAL_CATCH_RATE,
                )
                receptions = rng.binomial(targets[:, j], catch_rate)
                ypt = simulation_v2._num(r, "rules_ypt", "bayes_ypt", "ypt")
                ypt = RESIDUAL_YPT if not np.isfinite(ypt) or ypt <= 0 else float(ypt)
                ypr = float(np.clip(ypt / catch_rate, YPR_MIN, YPR_MAX))
                vol = float(np.clip(simulation_v2._num(r, "rules_volatility_mult", default=1.0), 0.75, 1.50))
                mu = receptions.astype(float) * ypr * pass_eff_shock
                sd = np.maximum(3.0, np.sqrt(np.maximum(receptions, 1)) * ypr * 0.55) * vol
                yards = np.clip(rng.normal(mu, sd), 0.0, None)
                yards = np.where(receptions > 0, yards, 0.0)
                zero_rec_positive_yards += int(((receptions == 0) & (yards > 0)).sum())
                raw_arrays[pkey] = yards
                receiver_values[(str(game), str(team), pkey, "receptions_raw")] = receptions.astype(float)

            residual_rec = rng.binomial(residual_targets, RESIDUAL_CATCH_RATE)
            residual_ypr = RESIDUAL_YPT / RESIDUAL_CATCH_RATE
            residual_mu = residual_rec.astype(float) * residual_ypr * pass_eff_shock
            residual_sd = np.maximum(3.0, np.sqrt(np.maximum(residual_rec, 1)) * residual_ypr * 0.55)
            residual_yards = np.clip(rng.normal(residual_mu, residual_sd), 0.0, None)
            residual_yards = np.where(residual_rec > 0, residual_yards, 0.0)

            if raw_arrays:
                raw_modeled = np.sum(np.vstack(list(raw_arrays.values())), axis=0)
            else:
                raw_modeled = np.zeros(iterations, dtype=float)
            raw_total = raw_modeled + residual_yards
            raw_mean = float(np.mean(raw_total))
            anchor = float(anchor_map.get((str(game), str(team)), np.nan))
            if not np.isfinite(anchor) or anchor <= 0:
                raise RuntimeError(f"missing/invalid pass-yard mean anchor game={game} team={team}: {anchor}")
            if not np.isfinite(raw_mean) or raw_mean <= 0:
                raise RuntimeError(f"invalid raw receiver mean game={game} team={team}: {raw_mean}")
            scale = anchor / raw_mean

            scaled_arrays = {}
            for pkey, raw in raw_arrays.items():
                scaled = raw * scale
                scaled_arrays[pkey] = scaled
                receiver_values[(str(game), str(team), pkey, "rec_yards")] = scaled
                rec_raw = receiver_values.pop((str(game), str(team), pkey, "receptions_raw"))
                receiver_values[(str(game), str(team), pkey, "receptions")] = rec_raw
            scaled_residual = residual_yards * scale
            if scaled_arrays:
                modeled_sum = np.sum(np.vstack(list(scaled_arrays.values())), axis=0)
            else:
                modeled_sum = np.zeros(iterations, dtype=float)
            qb = modeled_sum + scaled_residual
            qb_values[(str(game), str(team))] = qb
            gap = qb - (modeled_sum + scaled_residual)
            conservation_rows.append({
                "event_id": str(game),
                "team": str(team),
                "anchor_mean": anchor,
                "candidate_qb_mean": float(np.mean(qb)),
                "scale": float(scale),
                "residual_receiver_yards_mean": float(np.mean(scaled_residual)),
                "max_abs_gap": float(np.max(np.abs(gap))),
                "mean_abs_gap": float(np.mean(np.abs(gap))),
                "median_abs_gap": float(np.median(np.abs(gap))),
                "p90_abs_gap": float(np.quantile(np.abs(gap), 0.90)),
                "p95_abs_gap": float(np.quantile(np.abs(gap), 0.95)),
                "pct_gap_gt_0_01": float((np.abs(gap) > 0.01).mean()),
                "pct_gap_gt_1": float((np.abs(gap) > 1.0).mean()),
                "pct_gap_gt_10": float((np.abs(gap) > 10.0).mean()),
            })
    return receiver_values, qb_values, conservation_rows, zero_rec_positive_yards


def actual_usage(logs: pd.DataFrame, season: int, weeks: set[int]) -> pd.DataFrame:
    x = logs.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    x["season"] = num(x["season"])
    x["week"] = num(x["week"])
    x = x.loc[x["season"].eq(int(season)) & x["week"].isin(sorted(weeks))].copy()
    for c in ["targets", "receptions", "rec_yards", "rushes", "rush_yards", "pass_att", "pass_yards"]:
        x[c] = num(x[c]).fillna(0.0) if c in x.columns else 0.0
    for c in ["player", "player_clean_key", "player_identity_key", "team", "position"]:
        if c not in x.columns:
            x[c] = ""
        x[c] = x[c].fillna("").astype(str)
    x["team"] = x["team"].map(canon_team)
    x["position_group"] = x["position"].map(position_group)
    x["mass_group"] = x["position"].map(mass_group)
    x["rush_rec_yards"] = x["rush_yards"] + x["rec_yards"]
    x["join_key"] = np.where(
        x["player_identity_key"].str.strip().ne(""),
        "id:" + x["player_identity_key"].str.strip(),
        "name:" + x["player_clean_key"].str.strip(),
    )
    keep = [
        "season", "week", "team", "player", "player_clean_key", "player_identity_key", "join_key",
        "position", "position_group", "mass_group", "targets", "receptions", "rec_yards", "rushes",
        "rush_yards", "rush_rec_yards", "pass_att", "pass_yards",
    ]
    return x[keep].sort_values(["season", "week", "team", "join_key"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--prior-season", type=int, required=True)
    p.add_argument("--weeks", required=True)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--player-logs", type=Path, required=True)
    p.add_argument("--team-weekly", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--universe-dir", type=Path, required=True)
    p.add_argument("--injuries", type=Path, required=True)
    p.add_argument("--weather", type=Path, required=True)
    p.add_argument("--m89-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    logs = read(args.player_logs)
    team_weekly = read(args.team_weekly)
    schedule = read(args.schedule)
    injuries = opt(args.injuries)
    weather = opt(args.weather)
    m89 = load_m89(args.m89_root)
    weeks = _parse_weeks(args.weeks)

    player_rows: list[dict] = []
    calibration_rows: list[dict] = []
    conservation_rows: list[dict] = []
    qb_rows: list[dict] = []
    zero_counts = {"C2": 0, "C3": 0}

    for week in weeks:
        universe = read(args.universe_dir / f"{args.season}_week_{week:02d}.csv")
        bundle = build_historical_context_bundle(
            player_logs=logs,
            team_weekly=team_weekly,
            pregame_universe=universe,
            schedule=schedule,
            season=args.season,
            week=week,
            prior_season=args.prior_season,
            injuries=_exact_week(injuries, args.season, week),
            weather=_exact_week(weather, args.season, week),
        )
        metrics = prepared(bundle)
        players = unique_players(metrics)

        b0_prob_map: dict[tuple[str, str, str], float] = {}
        c1_prob_map: dict[tuple[str, str, str], float] = {}
        team_audits = {}
        for (game, team), team_df in players.groupby(["event_id", "team"], dropna=False, sort=False):
            team_df = team_df.reset_index(drop=True)
            base, base_resid = baseline_probabilities(team_df)
            hist, h_audit = historical_group_shares(logs, args.season, int(week), str(team))
            cand, c_audit = calibrated_probabilities(team_df, base, hist)
            for j, (_, r) in enumerate(team_df.iterrows()):
                key = (str(game), str(team), str(r.get("player_clean_key", "")))
                b0_prob_map[key] = float(base[j])
                c1_prob_map[key] = float(cand[j])
            team_audits[(str(game), str(team))] = {**h_audit, **c_audit, "b0_residual_probability": float(base_resid), "candidate_residual_probability": float(max(0.0, 1.0 - cand.sum()))}
            calibration_rows.append({
                "season": int(args.season), "week": int(week), "event_id": str(game), "team": str(team),
                **team_audits[(str(game), str(team))],
            })

        b0 = simulation_v2.simulate(metrics, iterations=args.iterations, seed=42 + int(week))
        metrics_c1 = apply_probability_map(metrics, c1_prob_map)
        with patch.object(simulation_v2, "_sharpen_wr_target_shares", side_effect=lambda team_df, shares: np.asarray(shares, dtype=float)):
            c1 = simulation_v2.simulate(metrics_c1, iterations=args.iterations, seed=42 + int(week))

        # Mean anchors: promoted M89/M90 for 2024-25; otherwise exact B0 primary-QB MC mean.
        anchor_map: dict[tuple[str, str], float] = {}
        b0_qb_raw: dict[tuple[str, str], np.ndarray] = {}
        b0_qb_key: dict[tuple[str, str], str] = {}
        for (game, team), team_df in players.groupby(["event_id", "team"], dropna=False, sort=False):
            pkey, qarr = primary_qb_arrays(b0, team_df.reset_index(drop=True), game)
            if qarr is None:
                continue
            k = (str(game), str(team))
            b0_qb_raw[k] = qarr
            b0_qb_key[k] = str(pkey)
            if int(args.season) in (2024, 2025):
                hit = m89.loc[(m89.season.eq(int(args.season))) & (m89.week.eq(int(week))) & (m89.team.eq(canon_team(team)))]
                if len(hit) != 1:
                    raise RuntimeError(f"M89 anchor mismatch season={args.season} week={week} team={team} rows={len(hit)}")
                anchor_map[k] = float(hit.iloc[0].football_synthesis)
            else:
                anchor_map[k] = float(np.mean(qarr))

        c2_rec, c2_qb, c2_cons, c2_zero = conserved_receiving(
            players, b0_prob_map, iterations=args.iterations, seed=200000 + 42 + int(week), anchor_map=anchor_map
        )
        c3_rec, c3_qb, c3_cons, c3_zero = conserved_receiving(
            players, c1_prob_map, iterations=args.iterations, seed=300000 + 42 + int(week), anchor_map=anchor_map
        )
        zero_counts["C2"] += int(c2_zero)
        zero_counts["C3"] += int(c3_zero)
        for r in c2_cons:
            conservation_rows.append({"season": int(args.season), "week": int(week), "variant": "C2", **r})
        for r in c3_cons:
            conservation_rows.append({"season": int(args.season), "week": int(week), "variant": "C3", **r})

        # Player-level means for all four frozen architectures.
        for _, r in players.iterrows():
            game = str(r.get("event_id", "")); team = str(r.get("team", "")); pkey = str(r.get("player_clean_key", "") or "")
            pos = str(r.get("position", "") or "").upper().strip()
            pg = position_group(pos); mg = mass_group(pos)
            if pos not in PASS_CATCHER_POSITIONS:
                continue
            identity = str(r.get("player_identity_key", "") or "").strip()
            join_key = f"id:{identity}" if identity else f"name:{pkey}"
            b0_rec_a = sim_arr(b0, game, pkey, "receptions")
            b0_y_a = sim_arr(b0, game, pkey, "rec_yards")
            b0_rush_a = sim_arr(b0, game, pkey, "rush_yards")
            c1_rec_a = sim_arr(c1, game, pkey, "receptions")
            c1_y_a = sim_arr(c1, game, pkey, "rec_yards")
            c2_rec_a = c2_rec.get((game, team, pkey, "receptions"))
            c2_y_a = c2_rec.get((game, team, pkey, "rec_yards"))
            c3_rec_a = c3_rec.get((game, team, pkey, "receptions"))
            c3_y_a = c3_rec.get((game, team, pkey, "rec_yards"))
            plays_mean, pass_rate_mean = simulation_v2._team_inputs(players.loc[(players.event_id.astype(str).eq(game)) & (players.team.astype(str).eq(team))])
            team_pass_mean = float(plays_mean * pass_rate_mean)
            b0_prob = float(b0_prob_map.get((game, team, pkey), 0.0))
            c1_prob = float(c1_prob_map.get((game, team, pkey), 0.0))
            b0_rush_mean = float(np.mean(b0_rush_a)) if b0_rush_a is not None else np.nan
            b0_y_mean = float(np.mean(b0_y_a)) if b0_y_a is not None else np.nan
            c1_y_mean = float(np.mean(c1_y_a)) if c1_y_a is not None else np.nan
            c2_y_mean = float(np.mean(c2_y_a)) if c2_y_a is not None else np.nan
            c3_y_mean = float(np.mean(c3_y_a)) if c3_y_a is not None else np.nan
            player_rows.append({
                "season": int(args.season), "week": int(week), "event_id": game, "team": team,
                "player": r.get("player", ""), "player_clean_key": pkey, "player_identity_key": identity, "join_key": join_key,
                "position": pos, "position_group": pg, "mass_group": mg,
                "b0_target_probability": b0_prob, "c1_target_probability": c1_prob,
                "c2_target_probability": b0_prob, "c3_target_probability": c1_prob,
                "b0_expected_targets": team_pass_mean * b0_prob, "c1_expected_targets": team_pass_mean * c1_prob,
                "c2_expected_targets": team_pass_mean * b0_prob, "c3_expected_targets": team_pass_mean * c1_prob,
                "b0_receptions": float(np.mean(b0_rec_a)) if b0_rec_a is not None else np.nan,
                "c1_receptions": float(np.mean(c1_rec_a)) if c1_rec_a is not None else np.nan,
                "c2_receptions": float(np.mean(c2_rec_a)) if c2_rec_a is not None else np.nan,
                "c3_receptions": float(np.mean(c3_rec_a)) if c3_rec_a is not None else np.nan,
                "b0_rec_yards": b0_y_mean, "c1_rec_yards": c1_y_mean, "c2_rec_yards": c2_y_mean, "c3_rec_yards": c3_y_mean,
                "b0_rush_yards": b0_rush_mean, "c1_rush_yards": b0_rush_mean, "c2_rush_yards": b0_rush_mean, "c3_rush_yards": b0_rush_mean,
                "b0_rush_rec_yards": b0_rush_mean + b0_y_mean if np.isfinite(b0_rush_mean) and np.isfinite(b0_y_mean) else np.nan,
                "c1_rush_rec_yards": b0_rush_mean + c1_y_mean if np.isfinite(b0_rush_mean) and np.isfinite(c1_y_mean) else np.nan,
                "c2_rush_rec_yards": b0_rush_mean + c2_y_mean if np.isfinite(b0_rush_mean) and np.isfinite(c2_y_mean) else np.nan,
                "c3_rush_rec_yards": b0_rush_mean + c3_y_mean if np.isfinite(b0_rush_mean) and np.isfinite(c3_y_mean) else np.nan,
            })

        # QB distribution diagnostics for the exact 2024-25 promoted mean era.
        if int(args.season) in (2024, 2025):
            for (game, team), raw_b0 in b0_qb_raw.items():
                hit = m89.loc[(m89.season.eq(int(args.season))) & (m89.week.eq(int(week))) & (m89.team.eq(canon_team(team)))]
                if len(hit) != 1:
                    continue
                anchor = float(hit.iloc[0].football_synthesis)
                actual = float(hit.iloc[0].actual_pass_yards)
                raw_mean = float(np.mean(raw_b0))
                if not np.isfinite(raw_mean) or raw_mean <= 0:
                    raise RuntimeError(f"invalid B0 QB raw mean {game} {team}")
                b0_anchor = np.asarray(raw_b0, dtype=float) * (anchor / raw_mean)
                c2a = c2_qb.get((game, team)); c3a = c3_qb.get((game, team))
                if c2a is None or c3a is None:
                    raise RuntimeError(f"missing candidate QB array game={game} team={team}")
                qb_rows.append({
                    "season": int(args.season), "week": int(week), "event_id": game, "team": canon_team(team),
                    "primary_qb_key": b0_qb_key.get((game, team), ""), "football_synthesis": anchor, "actual_pass_yards": actual,
                    **distribution_stats(b0_anchor, actual, "b0"),
                    **distribution_stats(c2a, actual, "c2"),
                    **distribution_stats(c3a, actual, "c3"),
                })

        print(f"[joint-v1] season={args.season} week={int(week):02d} players={len(players)}")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(player_rows).to_csv(out / "joint_v1_player_projection_trace.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(out / "joint_v1_group_calibration_trace.csv", index=False)
    pd.DataFrame(conservation_rows).to_csv(out / "joint_v1_conservation_trace.csv", index=False)
    pd.DataFrame(qb_rows).to_csv(out / "joint_v1_qb_distribution_trace.csv", index=False)
    actual_usage(logs, args.season, set(int(w) for w in weeks)).to_csv(out / "joint_v1_actual_usage.csv", index=False)
    pd.DataFrame([{"season": int(args.season), "c2_zero_rec_positive_yards": zero_counts["C2"], "c3_zero_rec_positive_yards": zero_counts["C3"]}]).to_csv(out / "joint_v1_integrity_counts.csv", index=False)
    print(f"[joint-v1] wrote player={len(player_rows)} qb={len(qb_rows)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
