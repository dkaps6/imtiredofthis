#!/usr/bin/env python3
"""Frozen Rush Pool Evidence Guard V1 historical evaluator.

One candidate only. No parameter search.

From Week 2 onward, the current top-five rushing pool is preserved, except
players with player-specific rushing evidence are allowed to occupy the finite
five-player pool before synthetic position-prior-only fallbacks. Raw rushing
shares, team rush volume, normalization, residual semantics and efficiency are
unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.backtest.component_predictions import build_actual_rows, build_market_frame
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from scripts.modeling import simulation_rules
from scripts.simulation_v2 import _team_inputs

VERSION = "RUSH_POOL_EVIDENCE_GUARD_V1"
FALLBACK_STATE = "position_prior_only"
POOL_SIZE = 5
PLAYER_MASS_CAP = 0.95
RB_FAMILY = {"RB", "FB", "HB"}
FORBIDDEN_INPUT_TOKENS = ("sportsbook", "book", "odds", "line", "market_prob", "implied_prob")


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    return pd.read_csv(path)


def _optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _finite(value, default=0.0) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _player_key(value) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _prepare_metrics(bundle) -> pd.DataFrame:
    metrics = build_market_frame(bundle)
    bayes = build_bayesian_baseline(bundle.player_consensus)
    metrics = apply_bayesian_to_metrics(metrics, bayes)
    with patch.object(simulation_rules, "load_model_contexts", return_value=(bundle.teams, bundle.players)):
        metrics = simulation_rules.apply_rules_to_metrics(metrics)

    required = {
        "event_id", "team", "player", "player_clean_key", "position",
        "rules_plays_est", "rules_pass_rate", "rules_rush_share",
        "bayes_evidence_state",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise RuntimeError(f"prepared rushing metrics missing columns: {sorted(missing)}")

    cols = [
        "event_id", "team", "opponent", "player", "player_clean_key", "position",
        "rules_plays_est", "rules_pass_rate", "rules_rush_share",
        "bayes_evidence_state", "bayes_rush_share",
    ]
    cols = [c for c in cols if c in metrics.columns]
    out = metrics[cols].drop_duplicates(["event_id", "team", "player_clean_key"], keep="last").copy()
    out["rules_rush_share"] = pd.to_numeric(out["rules_rush_share"], errors="coerce").fillna(0.0)
    out["bayes_evidence_state"] = out["bayes_evidence_state"].fillna("").astype(str)
    positive = out["rules_rush_share"].gt(0)
    if out.loc[positive, "bayes_evidence_state"].eq("").any():
        bad = out.loc[positive & out["bayes_evidence_state"].eq(""), ["team", "player"]].head(10)
        raise RuntimeError(f"positive-share players missing bayes_evidence_state: {bad.to_dict('records')}")
    return out


def _baseline_mask(shares: np.ndarray) -> np.ndarray:
    clean = np.clip(np.nan_to_num(np.asarray(shares, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0, 0.95)
    out = np.zeros(len(clean), dtype=bool)
    positive = np.flatnonzero(clean > 0.0)
    if len(positive) == 0:
        return out
    order = positive[np.argsort(-clean[positive], kind="stable")]
    out[order[:POOL_SIZE]] = True
    return out


def _candidate_mask(shares: np.ndarray, evidence_states: np.ndarray, *, week: int) -> np.ndarray:
    base = _baseline_mask(shares)
    if int(week) <= 1:
        return base

    clean = np.clip(np.nan_to_num(np.asarray(shares, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0, 0.95)
    states = np.asarray(evidence_states, dtype=object)
    positive = np.flatnonzero(clean > 0.0)
    evidenced = positive[states[positive] != FALLBACK_STATE]
    fallback = positive[states[positive] == FALLBACK_STATE]

    evidenced = evidenced[np.argsort(-clean[evidenced], kind="stable")] if len(evidenced) else evidenced
    fallback = fallback[np.argsort(-clean[fallback], kind="stable")] if len(fallback) else fallback

    keep = list(evidenced[:POOL_SIZE])
    if len(keep) < POOL_SIZE:
        keep.extend(list(fallback[: POOL_SIZE - len(keep)]))

    out = np.zeros(len(clean), dtype=bool)
    if keep:
        out[np.asarray(keep, dtype=int)] = True
    return out


def _player_probabilities(shares: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    clean = np.clip(np.nan_to_num(np.asarray(shares, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0, 0.95)
    selected = np.where(mask, clean, 0.0)
    raw_sum = float(selected.sum())
    used = selected.copy()
    if raw_sum > PLAYER_MASS_CAP:
        used *= PLAYER_MASS_CAP / raw_sum
    residual = max(0.0, 1.0 - float(used.sum()))
    total = float(used.sum()) + residual
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("invalid rushing allocation probability total")
    used = used / total
    residual = residual / total
    return used, float(residual)


def _actual_position_map(player_logs: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    x = player_logs.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    if "position" not in x.columns:
        return pd.DataFrame(columns=["team", "player_clean_key", "actual_position"])
    s = pd.to_numeric(x.get("season"), errors="coerce")
    w = pd.to_numeric(x.get("week"), errors="coerce")
    x = x.loc[s.eq(int(season)) & w.eq(int(week))].copy()
    if x.empty:
        return pd.DataFrame(columns=["team", "player_clean_key", "actual_position"])
    if "player_clean_key" not in x.columns:
        x["player_clean_key"] = x.get("player", "").map(_player_key)
    return (
        x[["team", "player_clean_key", "position"]]
        .rename(columns={"position": "actual_position"})
        .drop_duplicates(["team", "player_clean_key"], keep="last")
    )


def _score_group(frame: pd.DataFrame, group: str) -> dict:
    if group == "ALL":
        g = frame.copy()
    elif group == "RB_FAMILY":
        g = frame.loc[frame["position_family"].eq("RB_FAMILY")].copy()
    elif group == "QB":
        g = frame.loc[frame["position_family"].eq("QB")].copy()
    else:
        g = frame.loc[frame["position_family"].eq("OTHER")].copy()

    if g.empty:
        return {
            "group": group, "n": 0, "baseline_mae": None, "candidate_mae": None,
            "delta_mae": None, "baseline_p90": None, "candidate_p90": None,
            "baseline_bias": None, "candidate_bias": None,
            "changed_rows": 0, "candidate_closer": 0, "baseline_closer": 0,
            "decided_changed_rows": 0, "candidate_closer_rate": None,
        }

    actual = pd.to_numeric(g["actual_rush_att"], errors="coerce").fillna(0.0)
    b = pd.to_numeric(g["baseline_rush_att"], errors="coerce").fillna(0.0)
    c = pd.to_numeric(g["candidate_rush_att"], errors="coerce").fillna(0.0)
    be = b - actual
    ce = c - actual
    ba = be.abs()
    ca = ce.abs()
    changed = (b - c).abs().gt(1e-12)
    cand_closer = changed & ca.lt(ba - 1e-12)
    base_closer = changed & ba.lt(ca - 1e-12)
    decided = cand_closer | base_closer
    decided_n = int(decided.sum())
    cand_n = int(cand_closer.sum())
    return {
        "group": group,
        "n": int(len(g)),
        "baseline_mae": float(ba.mean()),
        "candidate_mae": float(ca.mean()),
        "delta_mae": float(ca.mean() - ba.mean()),
        "baseline_p90": float(ba.quantile(0.90)),
        "candidate_p90": float(ca.quantile(0.90)),
        "baseline_bias": float(be.mean()),
        "candidate_bias": float(ce.mean()),
        "changed_rows": int(changed.sum()),
        "candidate_closer": cand_n,
        "baseline_closer": int(base_closer.sum()),
        "decided_changed_rows": decided_n,
        "candidate_closer_rate": float(cand_n / decided_n) if decided_n else None,
    }


def _position_family(pos: str) -> str:
    p = str(pos or "").strip().upper()
    if p in RB_FAMILY:
        return "RB_FAMILY"
    if p == "QB":
        return "QB"
    return "OTHER"


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
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    player_rows: list[dict] = []
    team_rows: list[dict] = []

    # Frozen scope invariant: Week 1 must be an exact selector no-op.
    fixture_shares = np.asarray([0.40, 0.30, 0.20, 0.10, 0.05, 0.04], dtype=float)
    fixture_states = np.asarray(["prior+current", FALLBACK_STATE, "prior_only", FALLBACK_STATE, "prior+current", "prior_only"], dtype=object)
    if not np.array_equal(_baseline_mask(fixture_shares), _candidate_mask(fixture_shares, fixture_states, week=1)):
        raise RuntimeError("Week-1 no-op selector invariant failed")

    for week in weeks:
        if int(week) <= 1:
            continue
        u = _read(universe_dir / f"{int(season)}_week_{int(week):02d}.csv", f"{season} W{week} pregame universe")
        bundle = build_historical_context_bundle(
            player_logs=player_logs,
            team_weekly=team_weekly,
            pregame_universe=u,
            schedule=schedule,
            season=int(season),
            week=int(week),
            prior_season=int(prior_season),
            injuries=_exact_week(injuries, int(season), int(week)),
            weather=_exact_week(weather, int(season), int(week)),
        )
        metrics = _prepare_metrics(bundle)

        proj_rows = []
        for (event_id, team), g in metrics.groupby(["event_id", "team"], sort=False, dropna=False):
            g = g.reset_index(drop=True)
            shares = g["rules_rush_share"].to_numpy(float)
            states = g["bayes_evidence_state"].astype(str).to_numpy(object)
            base_mask = _baseline_mask(shares)
            cand_mask = _candidate_mask(shares, states, week=int(week))
            base_prob, base_resid = _player_probabilities(shares, base_mask)
            cand_prob, cand_resid = _player_probabilities(shares, cand_mask)

            plays_mean, pass_rate_mean = _team_inputs(g)
            team_rush_mean = float(plays_mean * (1.0 - pass_rate_mean))

            if int(base_mask.sum()) > POOL_SIZE or int(cand_mask.sum()) > POOL_SIZE:
                raise RuntimeError("top-five pool-size invariant failed")
            if float(base_prob.sum()) > PLAYER_MASS_CAP + 1e-12 or float(cand_prob.sum()) > PLAYER_MASS_CAP + 1e-12:
                raise RuntimeError("player-mass cap invariant failed")
            if abs((float(base_prob.sum()) + base_resid) - 1.0) > 1e-12:
                raise RuntimeError("baseline residual normalization failed")
            if abs((float(cand_prob.sum()) + cand_resid) - 1.0) > 1e-12:
                raise RuntimeError("candidate residual normalization failed")

            evidenced_positive = (shares > 0.0) & (states != FALLBACK_STATE)
            base_omitted_evidenced = int((evidenced_positive & ~base_mask).sum())
            cand_omitted_evidenced = int((evidenced_positive & ~cand_mask).sum())
            base_fallback_mass = float(base_prob[states == FALLBACK_STATE].sum())
            cand_fallback_mass = float(cand_prob[states == FALLBACK_STATE].sum())

            team_rows.append({
                "season": int(season), "week": int(week), "event_id": event_id, "team": team,
                "team_rush_mean": team_rush_mean,
                "baseline_pool_players": int(base_mask.sum()),
                "candidate_pool_players": int(cand_mask.sum()),
                "baseline_player_mass": float(base_prob.sum()),
                "candidate_player_mass": float(cand_prob.sum()),
                "baseline_residual": base_resid,
                "candidate_residual": cand_resid,
                "baseline_omitted_evidenced_positive": base_omitted_evidenced,
                "candidate_omitted_evidenced_positive": cand_omitted_evidenced,
                "baseline_position_prior_only_mass": base_fallback_mass,
                "candidate_position_prior_only_mass": cand_fallback_mass,
                "selector_changed": int(not np.array_equal(base_mask, cand_mask)),
            })

            for j, row in g.iterrows():
                proj_rows.append({
                    "season": int(season), "week": int(week), "event_id": event_id, "team": team,
                    "player": row.get("player"), "player_clean_key": row.get("player_clean_key"),
                    "position": row.get("position"), "bayes_evidence_state": states[j],
                    "rules_rush_share": float(shares[j]),
                    "baseline_selected": int(base_mask[j]), "candidate_selected": int(cand_mask[j]),
                    "baseline_probability": float(base_prob[j]), "candidate_probability": float(cand_prob[j]),
                    "baseline_rush_att": team_rush_mean * float(base_prob[j]),
                    "candidate_rush_att": team_rush_mean * float(cand_prob[j]),
                })

        proj = pd.DataFrame(proj_rows)
        actual = build_actual_rows(player_logs, int(season), int(week))
        actual = actual.loc[actual["market"].eq("rush_att"), ["team", "player_clean_key", "actual"]].rename(columns={"actual": "actual_rush_att"})
        apos = _actual_position_map(player_logs, int(season), int(week))
        actual = actual.merge(apos, on=["team", "player_clean_key"], how="left", validate="one_to_one")

        scored = proj.merge(actual, on=["team", "player_clean_key"], how="outer", validate="one_to_one")
        scored["season"] = int(season)
        scored["week"] = int(week)
        for col in ("baseline_rush_att", "candidate_rush_att", "actual_rush_att"):
            scored[col] = pd.to_numeric(scored[col], errors="coerce").fillna(0.0)
        scored["position"] = scored.get("position", pd.Series("", index=scored.index)).fillna(scored.get("actual_position", ""))
        scored["position_family"] = scored["position"].map(_position_family)
        keep = (
            scored["actual_rush_att"].gt(0.0)
            | scored["baseline_rush_att"].gt(0.0)
            | scored["candidate_rush_att"].gt(0.0)
        )
        scored = scored.loc[keep].copy()
        player_rows.extend(scored.to_dict("records"))

    detail = pd.DataFrame(player_rows)
    teams = pd.DataFrame(team_rows)
    if detail.empty or teams.empty:
        raise RuntimeError(f"empty Rush Pool Evidence Guard evaluation for season={season}")

    metric_rows = [_score_group(detail, group) for group in ("ALL", "RB_FAMILY", "QB", "OTHER")]
    summary = {
        "version": VERSION,
        "season": int(season),
        "prior_season": int(prior_season),
        "weeks": [int(w) for w in weeks if int(w) >= 2],
        "sportsbook_inputs_used": 0,
        "parameters_fit": 0,
        "candidate_variants_scored": 1,
        "week1_noop_verified": True,
        "raw_share_transform": "NONE",
        "pool_size": POOL_SIZE,
        "player_mass_cap": PLAYER_MASS_CAP,
        "selector_changed_team_games": int(teams["selector_changed"].sum()),
        "team_games": int(len(teams)),
        "baseline_omitted_evidenced_positive": int(teams["baseline_omitted_evidenced_positive"].sum()),
        "candidate_omitted_evidenced_positive": int(teams["candidate_omitted_evidenced_positive"].sum()),
        "baseline_position_prior_only_mass_mean": float(teams["baseline_position_prior_only_mass"].mean()),
        "candidate_position_prior_only_mass_mean": float(teams["candidate_position_prior_only_mass"].mean()),
        "metrics": {row["group"]: row for row in metric_rows},
    }
    return detail, teams, summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--prior-season", type=int, required=True)
    p.add_argument("--weeks", default="2-18")
    p.add_argument("--player-logs", type=Path, required=True)
    p.add_argument("--team-weekly", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--universe-dir", type=Path, required=True)
    p.add_argument("--injuries", type=Path, required=True)
    p.add_argument("--weather", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    logs = _read(a.player_logs, "player logs")
    team = _read(a.team_weekly, "team weekly")
    sched = _read(a.schedule, "schedule")
    injuries = _optional(a.injuries)
    weather = _optional(a.weather)
    weeks = _parse_weeks(a.weeks)

    detail, teams, summary = evaluate_season(
        season=a.season,
        prior_season=a.prior_season,
        weeks=weeks,
        player_logs=logs,
        team_weekly=team,
        schedule=sched,
        universe_dir=a.universe_dir,
        injuries=injuries,
        weather=weather,
    )

    a.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(a.out_dir / "player_detail.csv", index=False)
    teams.to_csv(a.out_dir / "team_detail.csv", index=False)
    (a.out_dir / "season_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(summary["metrics"].values()).to_csv(a.out_dir / "metrics.csv", index=False)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
