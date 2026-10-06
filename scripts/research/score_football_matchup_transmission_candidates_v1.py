#!/usr/bin/env python3
"""Score frozen Football Matchup Transmission integration candidates.

Frozen before scoring in:
docs/research/FOOTBALL_MATCHUP_TRANSMISSION_V1_INTEGRATION_CANDIDATES_FREEZE.md

No sportsbook inputs. No coefficient fitting. No production changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import (
    _attach_component_projection,
    build_actual_rows,
    build_mc_predictions,
)
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.modeling.ensemble_v2 import apply_ensemble
from scripts.modeling.ml_v2 import build_and_train as build_ml
from scripts.modeling.state_v2 import build_state_predictions
from scripts.research.audit_football_matchup_transmission_phase_bc_v1 import (
    _mean_col,
    _prior_team_rows,
)
from scripts.simulation_v2 import lookup, simulate

VERSION = "FOOTBALL_MATCHUP_TRANSMISSION_INTEGRATION_CANDIDATES_V1"
TARGET_SEASONS = (2024, 2025)
TARGET_WEEKS = tuple(range(2, 19))
ITERATIONS = 2000
BOOT_REPS = 5000
BOOT_SEED = 20261006
MIN_ROWS = 200
MIN_GAMES = 50
MIN_PLAYERS = 25
RB_POS = {"RB", "FB", "HB"}
PASS_RATE_FLOOR = 0.35
PASS_RATE_CEIL = 0.75
MATCHUP_MULT_FLOOR = 0.50
MATCHUP_MULT_CEIL = 1.80

CANDIDATES = (
    "FMT-INT-RB-DEF-PASS-RATE-FACED-V1",
    "FMT-INT-WR-TRUE-PROE-V1",
    "FMT-INT-TE-DEF-PASS-SUCCESS-V1",
)

COLLATERAL = {
    "RB_RUSH": ("rush_yards", RB_POS),
    "RB_RUSH_REC": ("rush_rec_yards", RB_POS),
    "RB_REC": ("rec_yards", RB_POS),
    "WR_REC": ("rec_yards", {"WR"}),
    "TE_REC": ("rec_yards", {"TE"}),
}
PRIMARY = {
    CANDIDATES[0]: "RB_RUSH",
    CANDIDATES[1]: "WR_REC",
    CANDIDATES[2]: "TE_REC",
}
FORBIDDEN = (
    "sportsbook", "bookmaker", "prop_line", "market_line", "over_odds",
    "under_odds", "spread_line", "total_line", "moneyline", "no_vig",
    "implied_prob", "closing_line",
)


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _key(v) -> str:
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _check_forbidden(label: str, frame: pd.DataFrame) -> None:
    bad = [c for c in frame.columns if any(t in c.lower() for t in FORBIDDEN)]
    if bad:
        raise RuntimeError(f"forbidden sportsbook/odds columns in {label}: {bad}")


def rb_def_pass_rate_candidate(rate: float, baseline: float = 0.57) -> float:
    if not np.isfinite(rate):
        return float(baseline)
    return float(np.clip(rate, PASS_RATE_FLOOR, PASS_RATE_CEIL))


def wr_true_proe_candidate(proe: float, baseline: float = 0.57) -> float:
    if not np.isfinite(proe):
        return float(baseline)
    return float(np.clip(float(baseline) + float(proe), PASS_RATE_FLOOR, PASS_RATE_CEIL))


def te_pass_success_multiplier(value: float, league_mean: float) -> float:
    if not np.isfinite(value) or not np.isfinite(league_mean):
        return 1.0
    return float(np.clip(
        1.0 + (float(value) - float(league_mean)),
        MATCHUP_MULT_FLOOR,
        MATCHUP_MULT_CEIL,
    ))


def _strict_state(corrected: pd.DataFrame, season: int, week: int, team: str) -> dict:
    h = _prior_team_rows(corrected, int(season), int(week), canon_team(team))
    return {
        "true_proe": _mean_col(h, ["true_proe", "proe"]),
        "pass_rate_faced": _mean_col(h, ["pass_rate_faced"]),
        "def_pass_success_allowed": _mean_col(
            h, ["def_pass_success_allowed", "success_rate_def"]
        ),
        "history_games": int(len(h)),
    }


def build_target_state(
    corrected: pd.DataFrame,
    schedule: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    s = schedule.copy()
    s["season"] = _num(s["season"])
    s["week"] = _num(s["week"])
    s["team"] = s["team"].map(canon_team)
    s["opponent"] = s["opponent"].map(canon_team)
    s = s.loc[
        s["season"].eq(int(season)) & s["week"].eq(int(week)),
        ["season", "week", "team", "opponent"],
    ].drop_duplicates(["team"])
    rows = []
    for r in s.itertuples(index=False):
        off = _strict_state(corrected, season, week, r.team)
        deff = _strict_state(corrected, season, week, r.opponent)
        rows.append({
            "team": r.team,
            "opponent": r.opponent,
            "off_true_proe": off["true_proe"],
            "opp_pass_rate_faced": deff["pass_rate_faced"],
            "opp_def_pass_success_allowed": deff["def_pass_success_allowed"],
            "off_history_games": off["history_games"],
            "opp_history_games": deff["history_games"],
        })
    out = pd.DataFrame(rows)
    vals = _num(out["opp_def_pass_success_allowed"])
    league_mean = float(vals.mean()) if vals.notna().any() else np.nan
    out["league_mean_def_pass_success_allowed"] = league_mean
    return out


def _base_components(
    bundle,
    player_logs: pd.DataFrame,
    season: int,
    week: int,
    *,
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mc = build_mc_predictions(bundle, iterations=iterations, seed=seed)
    _, ml_pred = build_ml(player_logs, bundle.player_consensus, int(season), int(week))
    _, state_pred = build_state_predictions(
        player_logs, bundle.player_consensus, int(season), int(week)
    )
    base_cols = [
        "player", "player_clean_key", "team", "opponent", "season", "week",
        "position", "role", "event_id", "market", "mc_proj",
    ]
    out = mc[base_cols].copy()
    out = _attach_component_projection(out, ml_pred, "ml")
    out = _attach_component_projection(out, state_pred, "state")
    actual = build_actual_rows(player_logs, int(season), int(week))
    out = out.merge(
        actual,
        on=["team", "player_clean_key", "market"],
        how="left",
        validate="one_to_one",
    )
    out = out.loc[_num(out["actual"]).notna()].reset_index(drop=True)
    return out, mc


def _candidate_mc(
    base_metrics: pd.DataFrame,
    target_state: pd.DataFrame,
    candidate: str,
    *,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    m = base_metrics.copy()
    state = target_state.set_index("team").to_dict("index")

    if candidate == CANDIDATES[0]:
        vals = []
        for r in m.itertuples(index=False):
            st = state.get(canon_team(r.team), {})
            vals.append(
                rb_def_pass_rate_candidate(
                    float(st.get("opp_pass_rate_faced", np.nan)),
                    float(getattr(r, "rules_pass_rate", 0.57))
                    if np.isfinite(float(getattr(r, "rules_pass_rate", np.nan)))
                    else 0.57,
                )
            )
        m["rules_pass_rate"] = vals

    elif candidate == CANDIDATES[1]:
        vals = []
        for r in m.itertuples(index=False):
            st = state.get(canon_team(r.team), {})
            vals.append(
                wr_true_proe_candidate(
                    float(st.get("off_true_proe", np.nan)),
                    0.57,
                )
            )
        m["rules_pass_rate"] = vals

    elif candidate == CANDIDATES[2]:
        ypt = _num(m["rules_ypt"]).copy()
        pos = m["position"].fillna("").astype(str).str.upper().str.strip()
        mult = []
        for r in m.itertuples(index=False):
            st = state.get(canon_team(r.team), {})
            mult.append(
                te_pass_success_multiplier(
                    float(st.get("opp_def_pass_success_allowed", np.nan)),
                    float(st.get("league_mean_def_pass_success_allowed", np.nan)),
                )
            )
        mult = pd.Series(mult, index=m.index, dtype=float)
        mask = pos.eq("TE") & ypt.notna()
        m.loc[mask, "rules_ypt"] = ypt.loc[mask] * mult.loc[mask]

    else:
        raise KeyError(candidate)

    sims = simulate(m, iterations=int(iterations), seed=int(seed))
    vals = []
    for _, row in m.iterrows():
        arr = lookup(sims, row, str(row["market"]))
        if arr is not None and len(arr) and str(row["market"]) == "pass_yards":
            attempt_rate = pd.to_numeric(
                pd.Series([row.get("mc_pass_attempts_per_dropback")]),
                errors="coerce",
            ).iloc[0]
            share = pd.to_numeric(
                pd.Series([row.get("qb_pass_att_share")]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(attempt_rate):
                arr = arr * float(np.clip(attempt_rate, 0.50, 1.00))
            if pd.notna(share):
                arr = arr * float(np.clip(share, 0.0, 1.0))
        vals.append(float(np.mean(arr)) if arr is not None and len(arr) else np.nan)
    out = m[[
        "season", "week", "team", "opponent", "player_clean_key", "position", "market"
    ]].copy()
    out["candidate_mc_proj"] = vals
    return out


def _cluster_bootstrap_improvement(
    q: pd.DataFrame,
    cluster_col: str,
    *,
    reps: int,
    seed: int,
) -> dict:
    z = q[["ae_improvement", cluster_col]].dropna().copy()
    if z.empty:
        return {"ci_low": np.nan, "ci_high": np.nan, "valid_reps": 0}
    g = z.groupby(cluster_col)["ae_improvement"].agg(["sum", "count"])
    if len(g) < 2:
        return {"ci_low": np.nan, "ci_high": np.nan, "valid_reps": 0}
    a = g[["sum", "count"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    p = np.full(len(a), 1.0 / len(a), dtype=float)
    vals = []
    done = 0
    while done < reps:
        k = min(250, reps - done)
        counts = rng.multinomial(len(a), p, size=k).astype(float)
        s = counts @ a
        vals.append(np.where(s[:, 1] > 0, s[:, 0] / s[:, 1], np.nan))
        done += k
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    return {
        "ci_low": float(np.quantile(v, 0.025)) if len(v) else np.nan,
        "ci_high": float(np.quantile(v, 0.975)) if len(v) else np.nan,
        "valid_reps": int(len(v)),
    }


def _metric_cell(q: pd.DataFrame, candidate: str, cohort: str, season: int) -> dict:
    z = q.loc[q["season"].eq(season)].copy()
    z = z.loc[
        _num(z["actual"]).notna()
        & _num(z["baseline_proj"]).notna()
        & _num(z["candidate_proj"]).notna()
    ].copy()
    rows = len(z)
    games = z["game_id"].nunique()
    players = z["player_clean_key"].nunique()
    support = rows >= MIN_ROWS and games >= MIN_GAMES and players >= MIN_PLAYERS
    rec = {
        "candidate": candidate,
        "cohort": cohort,
        "season": int(season),
        "rows": int(rows),
        "games": int(games),
        "players": int(players),
        "support": bool(support),
    }
    if not support:
        return rec

    actual = _num(z["actual"])
    base = _num(z["baseline_proj"])
    cand = _num(z["candidate_proj"])
    eb = base - actual
    ec = cand - actual
    z["ae_improvement"] = eb.abs() - ec.abs()

    game = _cluster_bootstrap_improvement(
        z, "game_id", reps=BOOT_REPS, seed=BOOT_SEED + season
    )
    player = _cluster_bootstrap_improvement(
        z, "player_clean_key", reps=BOOT_REPS, seed=BOOT_SEED + 10000 + season
    )
    rec.update({
        "baseline_mae": float(eb.abs().mean()),
        "candidate_mae": float(ec.abs().mean()),
        "mae_improvement": float(z["ae_improvement"].mean()),
        "baseline_rmse": float(np.sqrt(np.mean(np.square(eb)))),
        "candidate_rmse": float(np.sqrt(np.mean(np.square(ec)))),
        "baseline_bias": float(eb.mean()),
        "candidate_bias": float(ec.mean()),
        "game_ci_low": game["ci_low"],
        "game_ci_high": game["ci_high"],
        "game_boot_valid": game["valid_reps"],
        "player_ci_low": player["ci_low"],
        "player_ci_high": player["ci_high"],
        "player_boot_valid": player["valid_reps"],
        "supported_improvement": bool(
            np.isfinite(game["ci_low"]) and game["ci_low"] > 0
            and np.isfinite(player["ci_low"]) and player["ci_low"] > 0
        ),
        "supported_harm": bool(
            np.isfinite(game["ci_high"]) and game["ci_high"] < 0
            and np.isfinite(player["ci_high"]) and player["ci_high"] < 0
        ),
        "rmse_nonworse": bool(
            float(np.sqrt(np.mean(np.square(ec))))
            <= float(np.sqrt(np.mean(np.square(eb))))
        ),
    })
    return rec


def _cohort_mask(frame: pd.DataFrame, cohort: str) -> pd.Series:
    market, positions = COLLATERAL[cohort]
    return (
        frame["market"].eq(market)
        & frame["position"].fillna("").astype(str).str.upper().isin(positions)
    )


def score_candidate(detail: pd.DataFrame, candidate: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    for cohort in COLLATERAL:
        q = detail.loc[_cohort_mask(detail, cohort)].copy()
        for season in TARGET_SEASONS:
            rows.append(_metric_cell(q, candidate, cohort, season))
    cells = pd.DataFrame(rows)
    primary = PRIMARY[candidate]
    p = cells.loc[cells["cohort"].eq(primary)].copy()
    primary_pass = (
        len(p) == 2
        and p["support"].fillna(False).all()
        and (pd.to_numeric(p["mae_improvement"], errors="coerce") > 0).all()
        and p["supported_improvement"].fillna(False).all()
        and p["rmse_nonworse"].fillna(False).all()
    )
    collateral_harm = cells.loc[
        ~cells["cohort"].eq(primary)
        & cells["support"].fillna(False)
        & cells.get("supported_harm", pd.Series(False, index=cells.index)).fillna(False)
    ]
    passed = bool(primary_pass and collateral_harm.empty)
    return cells, {
        "candidate": candidate,
        "primary_cohort": primary,
        "primary_gate_pass": bool(primary_pass),
        "supported_collateral_harm_cells": int(len(collateral_harm)),
        "collateral_harm": collateral_harm[
            ["cohort", "season", "mae_improvement", "game_ci_high", "player_ci_high"]
        ].to_dict("records") if len(collateral_harm) else [],
        "disposition": (
            "HISTORICAL_INTEGRATION_PASS_FREEZE_FORWARD_SHADOW"
            if passed
            else "HISTORICAL_INTEGRATION_FAIL_CLOSED"
        ),
        "production_change_authorized": False,
    }


def _attach_game_id(frame: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    s = schedule[["season", "week", "team", "opponent", "game_id"]].copy()
    s["season"] = _num(s["season"]).astype(int)
    s["week"] = _num(s["week"]).astype(int)
    s["team"] = s["team"].map(canon_team)
    s["opponent"] = s["opponent"].map(canon_team)
    s = s.drop_duplicates(["season", "week", "team"])
    out = frame.drop(columns=["game_id"], errors="ignore").merge(
        s,
        on=["season", "week", "team", "opponent"],
        how="left",
        validate="many_to_one",
    )
    if out["game_id"].isna().any():
        raise RuntimeError("candidate detail missing authoritative game_id")
    return out


def _verify_baseline_authority(
    baseline: pd.DataFrame,
    right_tail: pd.DataFrame,
) -> dict:
    keys = ["season", "week", "team", "opponent", "player_clean_key", "market"]
    rt = right_tail.copy()
    rt["season"] = _num(rt["season"]).astype(int)
    rt["week"] = _num(rt["week"]).astype(int)
    rt = rt.loc[
        rt["season"].isin(TARGET_SEASONS)
        & rt["week"].isin(TARGET_WEEKS)
    ].copy()
    rt["team"] = rt["team"].map(canon_team)
    rt["opponent"] = rt["opponent"].map(canon_team)
    rt["player_clean_key"] = rt["player_clean_key"].map(_key)
    rt["final_mean"] = _num(rt["final_mean"])

    b = baseline[keys + ["ensemble_proj"]].copy()
    z = rt[keys + ["final_mean"]].merge(b, on=keys, how="left", validate="one_to_one")
    if z["ensemble_proj"].isna().any():
        raise RuntimeError(
            f"baseline authority identity missing rows={int(z['ensemble_proj'].isna().sum())}"
        )
    gap = (_num(z["ensemble_proj"]) - _num(z["final_mean"])).abs()
    max_gap = float(gap.max()) if len(gap) else np.nan
    if not np.isfinite(max_gap) or max_gap > 1e-8:
        raise RuntimeError(f"baseline authority numerical drift max_gap={max_gap}")
    return {"rows": int(len(z)), "max_abs_gap": max_gap}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-weekly-base", type=Path, required=True)
    ap.add_argument("--team-weekly-corrected", type=Path, required=True)
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--universe-dir-2024", type=Path, required=True)
    ap.add_argument("--universe-dir-2025", type=Path, required=True)
    ap.add_argument("--right-tail-detail", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path("data/model_ensemble_weights.csv"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    args = ap.parse_args()

    base_team = _read(args.team_weekly_base, "base team-week history")
    corrected = _read(args.team_weekly_corrected, "corrected team-week history")
    logs = _read(args.player_logs, "player logs")
    schedule = _read(args.schedule, "schedule")
    right_tail = _read(args.right_tail_detail, "right-tail baseline detail")
    weights = _read(args.weights, "ensemble weights")
    for label, frame in [
        ("base_team", base_team), ("corrected_team", corrected),
        ("player_logs", logs), ("schedule", schedule),
        ("right_tail", right_tail),
    ]:
        _check_forbidden(label, frame)

    for frame in (base_team, corrected, logs, schedule):
        frame["season"] = _num(frame["season"]).astype("Int64")
        frame["week"] = _num(frame["week"]).astype("Int64")
    base_team["team"] = base_team["team"].map(canon_team)
    corrected["team"] = corrected["team"].map(canon_team)
    logs["team"] = logs["team"].map(canon_team)
    logs["opponent"] = logs["opponent"].map(canon_team)
    schedule["team"] = schedule["team"].map(canon_team)
    schedule["opponent"] = schedule["opponent"].map(canon_team)

    baseline_parts = []
    candidate_parts = {c: [] for c in CANDIDATES}

    for season in TARGET_SEASONS:
        prior = season - 1
        udir = args.universe_dir_2024 if season == 2024 else args.universe_dir_2025
        for week in TARGET_WEEKS:
            upath = udir / f"{season}_week_{week:02d}.csv"
            if not upath.exists():
                raise RuntimeError(f"missing pregame universe {upath}")
            universe = _read(upath, f"pregame universe {season} W{week}")
            bundle = build_historical_context_bundle(
                player_logs=logs,
                team_weekly=base_team,
                pregame_universe=universe,
                schedule=schedule,
                season=season,
                week=week,
                prior_season=prior,
            )
            seed = 42 + week
            base_components, base_metrics = _base_components(
                bundle, logs, season, week,
                iterations=int(args.iterations), seed=seed,
            )
            baseline_parts.append(base_components)
            target_state = build_target_state(corrected, schedule, season, week)

            for candidate in CANDIDATES:
                cmc = _candidate_mc(
                    base_metrics, target_state, candidate,
                    iterations=int(args.iterations), seed=seed,
                )
                c = base_components.copy()
                keys = [
                    "season", "week", "team", "opponent",
                    "player_clean_key", "position", "market",
                ]
                c = c.drop(columns=["mc_proj"]).merge(
                    cmc[keys + ["candidate_mc_proj"]],
                    on=keys,
                    how="left",
                    validate="one_to_one",
                )
                if c["candidate_mc_proj"].isna().any():
                    raise RuntimeError(f"candidate MC missing rows {candidate} {season} W{week}")
                c = c.rename(columns={"candidate_mc_proj": "mc_proj"})
                candidate_parts[candidate].append(c)

    baseline_components = pd.concat(baseline_parts, ignore_index=True)
    baseline_final = apply_ensemble(baseline_components, weights=weights)
    baseline_final = _attach_game_id(baseline_final, schedule)
    authority = _verify_baseline_authority(baseline_final, right_tail)

    all_cells = []
    candidate_results = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for candidate in CANDIDATES:
        comp = pd.concat(candidate_parts[candidate], ignore_index=True)
        final = apply_ensemble(comp, weights=weights)
        final = _attach_game_id(final, schedule)
        keys = [
            "season", "week", "team", "opponent",
            "player_clean_key", "position", "market", "game_id",
        ]
        detail = baseline_final[
            keys + ["actual", "ensemble_proj"]
        ].rename(columns={"ensemble_proj": "baseline_proj"}).merge(
            final[keys + ["ensemble_proj"]].rename(
                columns={"ensemble_proj": "candidate_proj"}
            ),
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        if len(detail) != len(baseline_final):
            raise RuntimeError(
                f"candidate identity drift {candidate}: {len(detail)} != {len(baseline_final)}"
            )
        cells, result = score_candidate(detail, candidate)
        all_cells.append(cells)
        candidate_results.append(result)
        safe = candidate.lower().replace("-", "_")
        detail.to_csv(args.out_dir / f"{safe}_detail.csv", index=False)

    cells = pd.concat(all_cells, ignore_index=True)
    cells.to_csv(args.out_dir / "integration_candidate_cells.csv", index=False)
    pd.DataFrame(candidate_results).to_csv(
        args.out_dir / "integration_candidate_summary.csv", index=False
    )

    payload = {
        "version": VERSION,
        "freeze": "docs/research/FOOTBALL_MATCHUP_TRANSMISSION_V1_INTEGRATION_CANDIDATES_FREEZE.md",
        "evaluation_seasons": list(TARGET_SEASONS),
        "evaluation_weeks": [2, 18],
        "iterations": int(args.iterations),
        "bootstrap_reps": BOOT_REPS,
        "bootstrap_seed": BOOT_SEED,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "outcomes_2026_used": 0,
        "candidate_coefficients_fit": 0,
        "threshold_searches": 0,
        "target_game_usage_used": False,
        "baseline_authority_reproduction": authority,
        "candidates": candidate_results,
        "production_change_authorized": False,
    }
    (args.out_dir / "integration_candidate_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(cells.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
