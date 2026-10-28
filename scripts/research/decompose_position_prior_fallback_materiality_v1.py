#!/usr/bin/env python3
"""Decompose position-prior fallback materiality. Outcome-blind, descriptive.

Required by GPT audit 6425282770, "MATERIALITY CORRECTION": the headline
`5,422 / 10,960 = 49.5%` is card-level authority coverage pooled across target
weeks 1-7 and across opportunity kinds, and must NOT be characterised as the
live production fallback burden. Two reasons given, both correct:

  1. it pools target weeks, and early weeks naturally carry less named-player
     current evidence than Week 7;
  2. production priors are not equally material by position/metric, so a raw
     count mixes harmless ZERO-prior fallback rows with allocation-moving ones.

A QB with no target history is `position_prior_only`, but his production
target-share prior is 0.0, so that fallback cannot move a single target. The
same is true of a TE's rush share. Counting those beside an RB's 0.32 rush
share overstates the architecture problem.

This script therefore reports the decomposition the audit asked for, using
PRODUCTION's own prior table and position mapping imported directly from
`scripts/modeling/bayesian_v2` rather than values retyped from the audit text,
so the numbers cannot drift from what production would actually apply.

It is not a new predictive experiment. It reads the committed descriptive card
and the prior table. It inspects no outcome, no market and no result.

IMPORTANT -- WHAT THIS SCRIPT DOES NOT ESTABLISH (re-audit 6427742545).

Every quantity here is derived from the STATIC `DEFAULTS` table, and
production does not normally hand a zero-evidence player that constant.
`_weighted_group_mean()` first takes a `prior_games`-weighted mean of the
position group's own priors, then a median, and reaches `DEFAULTS` only when
no usable prior data exists at all. So a `position_prior_only` player's actual
posterior is the group mean -- measured at 0.359568 for zero-history RB/FB
rush share against a 0.32 default, and 0.165842 for WR target share against
0.16.

Consequently the fields here are named as the PROXIES they are:
`nonzero_default_prior_candidate` and `default_prior_mass_proxy`. They do NOT
mean "this row enters production's finite allocator", and the mass sums are
NOT production fallback mass. Production routing admits only a team's top five
`rules_rush_share` rows to the rush allocator and conserves target entitlement
to the finite room; a nonzero default on a broad card row proves neither.

The production-faithful counterpart lives in
`reconcile_production_fallback_admission_v1.py`, which reconciles against the
durable priced board rather than inferring from constants.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.sweep_player_state_threshold_v1 import (
    _market_shaped_columns,
)


def _load_production_priors():
    """Load the prior table and position mapping from production's own file.

    Loaded by path rather than `from scripts.modeling.bayesian_v2 import ...`
    because `scripts/modeling/__init__.py` pulls in ensemble_v2 and therefore
    scikit-learn. A descriptive decomposition should not need the ML stack to
    read a dict of constants, and dragging it in would make this script fail
    for reasons unrelated to its subject. bayesian_v2 itself imports nothing
    from its own package, so loading the module directly is safe.

    Reading the real file, rather than retyping the values from the audit
    text, is the point: these numbers cannot drift from what production
    applies.
    """
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "modeling" / "bayesian_v2.py"
    spec = importlib.util.spec_from_file_location(
        "_prism_bayesian_v2_priors", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load production priors from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DEFAULTS, mod._position_group


DEFAULTS, _position_group = _load_production_priors()

# A card's opportunity_kind maps onto the production share metric it would be
# allocated from.
_KIND_TO_METRIC = {"carry": "rush_share", "target": "tgt_share"}

FALLBACK = "position_prior_only"


def _prior_value(kind: str, position: str) -> float:
    metric = _KIND_TO_METRIC.get(str(kind))
    if metric is None:
        return float("nan")
    return float(DEFAULTS[metric][_position_group(position)])


def annotate(cards: pd.DataFrame) -> pd.DataFrame:
    out = cards.copy()
    out["production_position_group"] = out["roster_position"].map(_position_group)
    out["production_share_metric"] = out["opportunity_kind"].map(_KIND_TO_METRIC)
    out["production_prior_value"] = [
        _prior_value(k, p) for k, p in
        zip(out["opportunity_kind"], out["roster_position"])]
    out["is_position_prior_only"] = out["evidence_authority"].eq(FALLBACK)
    # The distinction the audit asked for: a fallback whose prior is 0.0
    # cannot move finite allocation at all.
    out["nonzero_default_prior_candidate"] = (
        out["is_position_prior_only"]
        & pd.to_numeric(out["production_prior_value"],
                        errors="coerce").fillna(0).gt(0))
    out["default_prior_mass_proxy"] = np.where(
        out["is_position_prior_only"],
        pd.to_numeric(out["production_prior_value"],
                      errors="coerce").fillna(0.0), 0.0)
    return out


def _by(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = df.groupby(keys, dropna=False)
    t = g.agg(
        cards=("gsis_id", "size"),
        position_prior_only=("is_position_prior_only", "sum"),
        nonzero_default_prior_candidate=("nonzero_default_prior_candidate", "sum"),
        default_prior_mass_proxy=("default_prior_mass_proxy", "sum"),
        production_prior_value=("production_prior_value", "max"),
    ).reset_index()
    t["pct_position_prior_only"] = (t["position_prior_only"]
                                    / t["cards"].replace(0, np.nan)).round(4)
    t["pct_nonzero_default_candidate"] = (t["nonzero_default_prior_candidate"]
                                  / t["cards"].replace(0, np.nan)).round(4)
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=Path, default=Path(
        "data/research/player_identity/player_identity_card_v1_2026.csv"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/research/player_identity"))
    ap.add_argument("--focus-week", type=int, default=7)
    a = ap.parse_args()

    cards = pd.read_csv(a.cards, low_memory=False)
    bad = _market_shaped_columns(cards)
    if bad:
        raise RuntimeError(f"card carries market-shaped columns {bad}")
    if not (pd.to_numeric(cards["sportsbook_inputs_used"],
                          errors="coerce").fillna(1) == 0).all():
        raise RuntimeError("card reports sportsbook input use")

    ann = annotate(cards)
    elig = ann.loc[ann["can_receive_opportunity"].astype(bool)].copy()

    # The audit's requested decomposition.
    by_wkp = _by(elig, ["target_week", "opportunity_kind", "roster_position"])
    by_week = _by(elig, ["target_week"])
    by_kind_pos = _by(elig, ["opportunity_kind", "roster_position"])
    focus = elig.loc[elig["target_week"].eq(a.focus_week)]
    by_focus = _by(focus, ["opportunity_kind", "roster_position"])

    # Implied fallback opportunity mass entering each finite room, in the same
    # framing as the Week-7 zero-history audit: a team's (week, kind) room is
    # normalised, so the share mass injected by evidence-free players is what
    # competes against named players.
    room = (focus.groupby(["team_target_week", "opportunity_kind"],
                          dropna=False)
            .agg(players=("gsis_id", "size"),
                 fallback_players=("is_position_prior_only", "sum"),
                 nonzero_default_candidate_players=("nonzero_default_prior_candidate",
                                            "sum"),
                 default_prior_mass_proxy=("default_prior_mass_proxy", "sum"))
            .reset_index()
            .sort_values("default_prior_mass_proxy", ascending=False))

    pooled_n = int(elig["is_position_prior_only"].sum())
    pooled_pct = pooled_n / max(1, len(elig))
    moving_n = int(elig["nonzero_default_prior_candidate"].sum())
    moving_pct = moving_n / max(1, len(elig))
    f_pooled = int(focus["is_position_prior_only"].sum())
    f_moving = int(focus["nonzero_default_prior_candidate"].sum())

    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "position_prior_fallback_materiality_v1"
    by_wkp.to_csv(a.out_dir / f"{stem}_by_week_kind_position.csv", index=False)
    by_week.to_csv(a.out_dir / f"{stem}_by_week.csv", index=False)
    by_kind_pos.to_csv(a.out_dir / f"{stem}_by_kind_position.csv", index=False)
    by_focus.to_csv(a.out_dir / f"{stem}_week{a.focus_week}.csv", index=False)
    room.to_csv(a.out_dir / f"{stem}_week{a.focus_week}_rooms.csv", index=False)

    status = {
        "version": "POSITION_PRIOR_FALLBACK_MATERIALITY_V1",
        "authorized_by": "GPT_AUDIT_6425282770_MATERIALITY_CORRECTION",
        "scientific_boundary": "DESCRIPTIVE_DECOMPOSITION_ONLY_NO_FITTING",
        "outcomes_inspected": False,
        "sportsbook_inputs_used": 0,
        "prior_table_source": "scripts/modeling/bayesian_v2.DEFAULTS",
        "position_mapping_source": "scripts/modeling/bayesian_v2._position_group",
        "kind_to_metric": _KIND_TO_METRIC,
        "opportunity_eligible_cards": int(len(elig)),
        "pooled_position_prior_only": pooled_n,
        "pooled_position_prior_only_pct": round(pooled_pct, 4),
        "pooled_nonzero_default_candidate": moving_n,
        "pooled_nonzero_default_candidate_pct": round(moving_pct, 4),
        "zero_prior_fallback_cards": pooled_n - moving_n,
        "focus_week": a.focus_week,
        "focus_week_eligible_cards": int(len(focus)),
        "focus_week_position_prior_only": f_pooled,
        "focus_week_position_prior_only_pct": round(
            f_pooled / max(1, len(focus)), 4),
        "focus_week_nonzero_default_candidate": f_moving,
        "focus_week_nonzero_default_candidate_pct": round(
            f_moving / max(1, len(focus)), 4),
        "by_week": by_week.to_dict("records"),
        "by_kind_position": by_kind_pos.to_dict("records"),
    }
    (a.out_dir / f"{stem}_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")

    print("=== POSITION-PRIOR FALLBACK MATERIALITY V1 ===")
    print(f"opportunity-eligible cards: {len(elig)}")
    print(f"pooled position_prior_only : {pooled_n} ({pooled_pct:.1%})")
    print(f"  of which ZERO-prior, cannot move allocation: "
          f"{pooled_n - moving_n}")
    print(f"  allocation-moving        : {moving_n} ({moving_pct:.1%})")
    print(f"\nWeek {a.focus_week} slice: eligible={len(focus)} "
          f"position_prior_only={f_pooled} "
          f"({f_pooled / max(1, len(focus)):.1%}) "
          f"allocation_moving={f_moving} "
          f"({f_moving / max(1, len(focus)):.1%})")
    print("\nby target week:")
    print(by_week.to_string(index=False))
    print("\nby opportunity kind x roster position (all weeks):")
    print(by_kind_pos.to_string(index=False))
    print(f"\nWeek {a.focus_week} rooms with the most fallback prior mass:")
    print(room.head(12).to_string(index=False))
    print("\nNo threshold, weight or coefficient is produced here. The pooled "
          "figure alone overstates severity; the allocation-moving figure is "
          "the one that bears on the finite room.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
