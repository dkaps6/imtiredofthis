#!/usr/bin/env python3
"""Run production pricing with RB rush+receiving conservation.

The promoted Week-1 RB P3 rushing mean is authoritative for standalone RB rushing.
The legacy joint simulator also emits a rush+receiving distribution, but its rushing
component is the pre-P3 Monte Carlo component. That can make the same player's final
rush+receiving mean contradict final rush_yards + receiving_yards.

This wrapper changes no sportsbook inputs and promotes no new model. It preserves
the joint simulated receiving distribution and rescales only the simulated RB/FB
rushing component to the already-promoted P3 football mean before recombining the
two correlated components. The downstream pricing implementation remains unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_pricing_v2 as pricing
from scripts.modeling.rb_pricing_adapter_v1 import load_rb_context, lookup_rb_projection
from scripts.simulation_v2 import (
    MARKET_MAP,
    _player_key,
    lookup,
    simulate as canonical_simulate,
)

DATA = Path("data")
AUDIT_CSV = DATA / "rb_rush_rec_conservation_input_audit.csv"
AUDIT_JSON = DATA / "rb_rush_rec_conservation_input_audit.json"
RB_POSITIONS = {"RB", "FB", "HB", "TB"}


def _position(row: pd.Series) -> str:
    for col in ("position_group", "position", "alignment_position"):
        value = row.get(col)
        if value is None or pd.isna(value):
            continue
        pos = str(value).upper().strip()
        if pos:
            if pos in {"HB", "TB"} or pos.startswith("RB"):
                return "RB"
            if pos.startswith("FB"):
                return "FB"
            return pos
    return ""


def _game_key(row: pd.Series) -> str:
    game = row.get("event_id")
    if game is not None and not pd.isna(game) and str(game).strip():
        return str(game)
    return "|".join(sorted([str(row.get("team", "")), str(row.get("opponent", ""))]))


def _conserved_simulate(metrics: pd.DataFrame, *, iterations=None, seed=None, allocation_trace=None):
    result = canonical_simulate(
        metrics,
        iterations=iterations,
        seed=seed,
        allocation_trace=allocation_trace,
    )
    if metrics is None or metrics.empty:
        return result

    frame = metrics.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    canonical_market = frame.get("market", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower().map(
        lambda value: MARKET_MAP.get(value, value)
    )
    frame["_canonical_market"] = canonical_market
    frame["_position_family"] = frame.apply(_position, axis=1)
    eligible = frame.loc[
        frame["_canonical_market"].eq("rush_rec_yards")
        & frame["_position_family"].isin({"RB", "FB"})
    ].copy()
    if eligible.empty:
        AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["player", "team"]).to_csv(AUDIT_CSV, index=False)
        AUDIT_JSON.write_text(json.dumps({
            "disposition": "NO_ELIGIBLE_RB_RUSH_REC_ROWS",
            "players": 0,
            "sportsbook_inputs_used": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    identity_cols = [c for c in ("event_id", "team", "player_clean_key", "player") if c in eligible.columns]
    eligible = eligible.drop_duplicates(identity_cols, keep="first")
    rb_context = load_rb_context()
    audit_rows: list[dict] = []

    for _, row in eligible.iterrows():
        rush = lookup(result, row, "rush_yards")
        rec = lookup(result, row, "rec_yards")
        combo_before = lookup(result, row, "rush_rec_yards")
        if rush is None or rec is None or combo_before is None:
            raise RuntimeError(
                f"RB rush+receiving conservation missing simulation component "
                f"player={row.get('player')} team={row.get('team')}"
            )
        rush = np.asarray(rush, dtype=float)
        rec = np.asarray(rec, dtype=float)
        combo_before = np.asarray(combo_before, dtype=float)
        if len(rush) != len(rec) or len(rush) != len(combo_before):
            raise RuntimeError(f"RB rush+receiving simulation length mismatch player={row.get('player')}")
        if not np.isfinite(rush).all() or not np.isfinite(rec).all():
            raise RuntimeError(f"RB rush+receiving simulation contains non-finite component player={row.get('player')}")

        meta = lookup_rb_projection(row, rb_context)
        p3_mean = float(meta["rb_synthesis_proj"])
        raw_rush_mean = float(np.mean(rush))
        rec_mean = float(np.mean(rec))
        if raw_rush_mean > 0:
            scaled_rush = rush * (p3_mean / raw_rush_mean)
        elif abs(p3_mean) <= 1e-12:
            scaled_rush = np.zeros_like(rush)
        else:
            raise RuntimeError(
                f"cannot conserve positive P3 mean from zero rush distribution "
                f"player={row.get('player')} p3={p3_mean}"
            )
        conserved = scaled_rush + rec
        game = _game_key(row)
        pkey = _player_key(row)
        result.values[(game, pkey, "rush_rec_yards")] = conserved
        conserved_mean = float(np.mean(conserved))
        gap = conserved_mean - (p3_mean + rec_mean)
        audit_rows.append({
            "event_id": game,
            "player": row.get("player"),
            "team": row.get("team"),
            "opponent": row.get("opponent"),
            "p3_rush_mean": p3_mean,
            "raw_mc_rush_mean": raw_rush_mean,
            "rec_mc_mean": rec_mean,
            "legacy_combo_mean": float(np.mean(combo_before)),
            "conserved_combo_mean": conserved_mean,
            "conservation_gap": gap,
            "rb_synthesis_version": meta.get("rb_synthesis_version"),
            "rb_synthesis_route": meta.get("rb_synthesis_route"),
            "sportsbook_inputs_used": False,
        })

    audit = pd.DataFrame(audit_rows)
    max_gap = float(pd.to_numeric(audit["conservation_gap"], errors="coerce").abs().max())
    if not np.isfinite(max_gap) or max_gap > 1e-8:
        raise RuntimeError(f"RB rush+receiving conservation arithmetic failed max_gap={max_gap}")
    if not audit["rb_synthesis_version"].astype(str).eq("RB_P3_SYNTHESIS_V1").all():
        raise RuntimeError("RB rush+receiving conservation consumed non-promoted RB synthesis version")
    if not audit["rb_synthesis_route"].astype(str).eq("WEEK1_STACK_OVERRIDE").all():
        raise RuntimeError("RB rush+receiving conservation consumed non-Week1 RB route")

    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_CSV, index=False)
    payload = {
        "disposition": "RB_RUSH_REC_DISTRIBUTION_CONSERVED_WITH_PROMOTED_P3",
        "players": int(len(audit)),
        "max_arithmetic_gap": max_gap,
        "sportsbook_inputs_used": False,
        "audit": str(AUDIT_CSV),
    }
    AUDIT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[rb_rush_rec_conservation] " + json.dumps(payload, sort_keys=True))
    return result


def main() -> int:
    pricing.simulate = _conserved_simulate
    return int(pricing.main())


if __name__ == "__main__":
    raise SystemExit(main())
