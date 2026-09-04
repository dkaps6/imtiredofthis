#!/usr/bin/env python3
"""Build the promoted RB P3 Week-1 projection context without sportsbook data.

The active RB universe comes from Ourlads + the authoritative schedule.  The
script constructs internal football target rows for rush_att/rush_yards, runs
the same ML/State/Bayesian/rules/Monte-Carlo/ensemble components used by
production pricing, then applies P3.  No prop line, odds, or sportsbook file is
read or required.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.metrics_v2 import _join_optional, _join_player_form, _join_team_context
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics
from scripts.modeling.ensemble_v2 import apply_ensemble, load_weights
from scripts.modeling.ml_v2 import apply_ml_to_metrics
from scripts.modeling.rb_rush_synthesis_v1 import WEEK1_ROUTE, apply_p3
from scripts.modeling.state_v2 import apply_state_to_metrics
from scripts.modeling.simulation_rules import apply_rules_to_metrics
from scripts.runtime_context import resolve_season, resolve_week
from scripts.simulation_v2 import lookup, simulate

DATA = Path("data")
ML_DIAGNOSTICS = DATA / "model_ml_diagnostics.csv"
STATE_DIAGNOSTICS = DATA / "model_state_diagnostics.csv"
OUT = DATA / "rb_rush_synthesis_context.csv"
RB_POS = {"RB", "HB", "FB"}
RB_GROUPS = {"RB", "RUNNING BACK", "BACKFIELD"}


def _key(value) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} missing/empty: {path}")
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    if out.empty:
        raise RuntimeError(f"{label} has 0 rows: {path}")
    return out


def build_internal_rb_metrics(season: int, week: int) -> pd.DataFrame:
    roles = _read(DATA / "roles_ourlads.csv", "Ourlads roles")
    schedule = _read(DATA / "team_week_map.csv", "team-week map")

    schedule["season"] = pd.to_numeric(schedule["season"], errors="coerce")
    schedule["week"] = pd.to_numeric(schedule["week"], errors="coerce")
    schedule["team"] = schedule["team"].map(canon_team)
    schedule["opponent"] = schedule["opponent"].map(canon_team)
    cur = schedule.loc[
        schedule["season"].eq(int(season)) & schedule["week"].eq(int(week)),
        ["team", "opponent"],
    ].drop_duplicates("team")
    if cur.empty:
        raise RuntimeError(f"no schedule rows for season={season} week={week}")

    roles["team"] = roles["team"].map(canon_team)
    # Always re-normalize the Ourlads display name into the same punctuation-free
    # key used by PlayerForm/ML.  Ourlads can preserve apostrophes/hyphens in its
    # own player_clean_key (for example D'Andre Swift, De'Von Achane, and
    # Jacory Croskey-Merritt), which otherwise creates false identity misses.
    roles["player_clean_key"] = roles["player"].map(_key)
    pos = roles.get("position", pd.Series("", index=roles.index)).fillna("").astype(str).str.upper().str.strip()
    group = roles.get("position_group", pd.Series("", index=roles.index)).fillna("").astype(str).str.upper().str.strip()
    model_role = roles.get("model_role", roles.get("role", pd.Series("", index=roles.index))).fillna("").astype(str).str.upper()
    rb = roles.loc[pos.isin(RB_POS) | group.isin(RB_GROUPS) | model_role.str.startswith(("RB", "HB", "FB"))].copy()
    rb = rb.merge(cur, on="team", how="inner", validate="many_to_one")
    rb = rb.drop_duplicates(["team", "player_clean_key"], keep="first")
    if rb.empty:
        raise RuntimeError("Ourlads + schedule produced zero Week-1 RB/HB/FB players")

    base = rb[["player", "player_clean_key", "team", "opponent"]].copy()
    base["event_id"] = base.apply(lambda r: "|".join(sorted([str(r["team"]), str(r["opponent"])])), axis=1)
    base["season"] = int(season)
    base["week"] = int(week)

    # Internal target definitions, not sportsbook offers.  Deliberately no line
    # or odds columns exist in this frame.
    expanded = pd.concat(
        [base.assign(market="rush_att"), base.assign(market="rush_yards")],
        ignore_index=True,
    )
    out = _join_player_form(expanded, int(season), int(week))
    out = _join_team_context(out, int(season))
    out = _join_optional(out, int(week))
    out["season"] = int(season)
    out["week"] = int(week)
    out["team_abbr"] = out["team"]
    out["opponent_abbr"] = out["opponent"]
    out["player_canonical"] = out["player"]
    if "tgt_share" in out.columns and "target_share" not in out.columns:
        out["target_share"] = out["tgt_share"]
    if "yprr" in out.columns and "yprr_proxy" not in out.columns:
        out["yprr_proxy"] = out["yprr"]
    return out.loc[:, ~out.columns.duplicated()].copy()


def project_week1(season: int, week: int, *, iterations: int | None = None) -> pd.DataFrame:
    if int(week) != 1:
        raise RuntimeError("run_rb_week1_no_odds.py is intentionally Week-1 only")
    metrics = build_internal_rb_metrics(season, week)

    ml = _read(ML_DIAGNOSTICS, "ML diagnostics")
    state = _read(STATE_DIAGNOSTICS, "State diagnostics")
    metrics = apply_ml_to_metrics(metrics, ml)
    metrics = apply_state_to_metrics(metrics, state)
    metrics = apply_bayesian_to_metrics(metrics)
    metrics = apply_rules_to_metrics(metrics)

    if int(pd.to_numeric(metrics.get("ml_applied", 0), errors="coerce").fillna(0).sum()) == 0:
        raise RuntimeError("RB Week-1 dry run matched zero ML rows")
    if int(pd.to_numeric(metrics.get("state_applied", 0), errors="coerce").fillna(0).sum()) == 0:
        raise RuntimeError("RB Week-1 dry run matched zero State rows")
    if int(pd.to_numeric(metrics.get("bayes_applied", 0), errors="coerce").fillna(0).sum()) == 0:
        raise RuntimeError("RB Week-1 dry run matched zero Bayesian rows")
    if int(pd.to_numeric(metrics.get("rules_applied", 0), errors="coerce").fillna(0).sum()) == 0:
        raise RuntimeError("RB Week-1 dry run matched zero rules rows")

    sims = simulate(metrics, iterations=iterations)
    weights = load_weights()
    projected = []
    for _, row in metrics.iterrows():
        market = str(row["market"]).lower()
        outcomes = lookup(sims, row, market)
        if outcomes is None or len(outcomes) == 0:
            raise RuntimeError(f"RB Week-1 simulation missing {row.get('player')} {market}")
        mc_proj = float(np.mean(np.asarray(outcomes, dtype=float)))
        ens = apply_ensemble(
            pd.DataFrame([{
                "market": market,
                "mc_proj": mc_proj,
                "ml_proj": row.get("ml_proj"),
                "state_proj": row.get("state_proj"),
            }]),
            weights=weights,
        ).iloc[0]
        if str(ens["ensemble_status"]) != "calibrated":
            raise RuntimeError(
                f"RB Week-1 {market} is not using calibrated STACK1 weights: "
                f"status={ens['ensemble_status']} player={row.get('player')}"
            )
        projected.append({
            "season": int(season),
            "week": int(week),
            "event_id": row.get("event_id"),
            "player": row.get("player"),
            "player_clean_key": row.get("player_clean_key"),
            "team": row.get("team"),
            "opponent": row.get("opponent"),
            "position": row.get("position"),
            "market": market,
            "mc_proj": mc_proj,
            "ml_proj": row.get("ml_proj"),
            "state_proj": row.get("state_proj"),
            "ensemble_proj": float(ens["ensemble_proj"]),
            "ensemble_status": ens["ensemble_status"],
            "ensemble_method": ens["ensemble_method"],
            "ensemble_weight_mc": ens["ensemble_weight_mc"],
            "ensemble_weight_ml": ens["ensemble_weight_ml"],
            "ensemble_weight_state": ens["ensemble_weight_state"],
            "ensemble_calibration_rows": ens["ensemble_calibration_rows"],
        })

    comp = pd.DataFrame(projected)
    keys = ["season", "week", "event_id", "player", "player_clean_key", "team", "opponent", "position"]
    wide = comp.pivot_table(index=keys, columns="market", values="ensemble_proj", aggfunc="first").reset_index()
    wide = wide.rename(columns={"rush_att": "stack_att", "rush_yards": "stack_yards"})
    if wide[["stack_att", "stack_yards"]].isna().any().any():
        raise RuntimeError("RB Week-1 component pivot has missing stack_att/stack_yards")

    p3 = apply_p3(wide)
    if not p3["rb_synthesis_route"].astype(str).eq(WEEK1_ROUTE).all():
        raise RuntimeError("RB Week-1 dry run routed a row outside WEEK1_STACK_OVERRIDE")
    diff = (pd.to_numeric(p3["rb_synthesis_proj"], errors="coerce") - pd.to_numeric(p3["stack_yards"], errors="coerce")).abs()
    if float(diff.max()) > 1e-12:
        raise RuntimeError(f"RB Week-1 P3 != full-stack parent max_diff={float(diff.max())}")

    p3["football_only_no_odds"] = 1
    p3["sportsbook_inputs_used"] = 0
    p3["simulation_iterations"] = int(sims.iterations)
    return p3.sort_values(["team", "player_clean_key"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    season = int(args.season if args.season is not None else resolve_season())
    week = int(args.week if args.week is not None else resolve_week(season=season))
    out = project_week1(season, week, iterations=args.iterations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(
        f"RB P3 WEEK1 NO-ODDS DRY RUN: PASS rows={len(out)} teams={out['team'].nunique()} "
        f"min={out['rb_synthesis_proj'].min():.3f} max={out['rb_synthesis_proj'].max():.3f} out={args.out}"
    )
    print(out[["team", "player", "stack_att", "stack_yards", "rb_synthesis_proj", "rb_synthesis_route"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
