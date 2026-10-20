#!/usr/bin/env python3
"""Grade a preregistered alt-line card against final actuals.

A card is a frozen set of OVER / "+ yardage" legs with thresholds chosen
before kickoff, recorded under ``data/market_track_record/cards/``. This grades
each leg at its own threshold rather than at the sportsbook's main line, which
is what the alt-line construction process actually commits to.

Losses are classified, not pooled, because the card declares which layer it
skipped:

``MODEL_MISS``
    the player participated in his stated role and the projection was wrong.
``AVAILABILITY_MISS``
    no snap evidence for the event -- inactive, or a role that never
    materialised. Attributable to the availability layer, not the projection.
``STATE_UNRESOLVED``
    identity or outcome could not be proven. Per the anomaly gate this fails
    closed into its own bucket rather than being assigned to either side.

Snap participation cannot see an early exit: a quarterback who takes the field
and leaves in the first quarter is ``snap_participated=True`` with a near-zero
stat line. Those rows are therefore flagged for investigation rather than
silently counted as projection failures.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.name_clean import canonical_key  # noqa: E402
from scripts.operations.grade_market_track_record_gsis_v1 import (  # noqa: E402
    _snap_participation_sets,
    _suffix_strip,
    build_alias_index,
    load_actual_stats_unfiltered,
    load_roster_identity,
    load_snap_participation,
    resolve_gsis,
)

# An actual this far below the projection is a discontinuity to explain before
# it counts as evidence about the model, per the player-state anomaly gate.
LOW_OUTPUT_RATIO = 0.25


def grade(cards_path: Path, season: int, week: int) -> pd.DataFrame:
    legs = pd.read_csv(cards_path)
    actual = load_actual_stats_unfiltered(season, [week])
    roster = load_roster_identity(season, [week])
    snaps = load_snap_participation(season, [week])
    idx = build_alias_index(actual, roster)
    snap_exact, snap_base = _snap_participation_sets(snaps)

    stats = actual.loc[actual["week"].astype("Int64") == week]
    by_gsis = {str(r.gsis_id): r for r in stats.itertuples(index=False)}

    rows = []
    for leg in legs.itertuples(index=False):
        key = canonical_key(leg.player)
        gsis, id_status = resolve_gsis(key, leg.team, idx)
        stat = by_gsis.get(str(gsis)) if gsis else None
        participated = (
            (season, week, leg.team, key) in snap_exact
            or (season, week, leg.team, _suffix_strip(key)) in snap_base
        )

        got = getattr(stat, leg.market, None) if stat is not None else None
        if got is None or not gsis:
            outcome, bucket = "UNRESOLVED", "STATE_UNRESOLVED"
        elif float(got) >= float(leg.threshold):
            outcome, bucket = "HIT", ""
        elif not participated:
            outcome, bucket = "MISS", "AVAILABILITY_MISS"
        else:
            outcome, bucket = "MISS", "MODEL_MISS"

        proj = float(leg.model_proj)
        rows.append({
            "card": leg.card, "tier": leg.tier, "player": leg.player,
            "team": leg.team, "opponent": leg.opponent, "market": leg.market,
            "threshold": float(leg.threshold), "model_proj": proj,
            "stated_p": float(leg.stated_p), "rank_used": leg.rank_used,
            "actual": (float(got) if got is not None else None),
            "outcome": outcome, "loss_bucket": bucket,
            "margin": (float(got) - float(leg.threshold)) if got is not None else None,
            "snap_participated": participated,
            "identity_status": id_status,
            "investigate_low_output": bool(
                got is not None and proj > 0 and float(got) < LOW_OUTPUT_RATIO * proj
            ),
        })
    return pd.DataFrame(rows)


def report(detail: pd.DataFrame) -> None:
    print("\n=== LEG-LEVEL ===")
    for card, grp in detail.groupby("card", sort=True):
        hits = int((grp["outcome"] == "HIT").sum())
        print(f"\n{card}: {hits}-{len(grp) - hits}  ({hits / len(grp) * 100:.1f}%)")
        for tier in ("SAFE", "MODERATE", "HIGH", "LOTTERY"):
            t = grp.loc[grp["tier"] == tier]
            if t.empty:
                continue
            th = int((t["outcome"] == "HIT").sum())
            print(f"  {tier:<9}{th}/{len(t)}")
            for r in t.itertuples(index=False):
                mark = "HIT " if r.outcome == "HIT" else ("MISS" if r.outcome == "MISS" else "UNRS")
                act = f"{r.actual:.0f}" if r.actual is not None and pd.notna(r.actual) else "--"
                flag = "  <<< INVESTIGATE" if r.investigate_low_output else ""
                print(f"      {mark} {r.player[:20]:<21}{r.rank_used:<4}"
                      f"{int(r.threshold):>4}+ {r.market.replace('_yards',''):<5}"
                      f" proj {r.model_proj:>6.1f}  actual {act:>5}"
                      f"  p={r.stated_p:.3f} {r.loss_bucket}{flag}")

    print("\n=== TICKET-LEVEL (all legs must hit) ===")
    for (card, tier), grp in detail.groupby(["card", "tier"], sort=False):
        won = bool((grp["outcome"] == "HIT").all())
        print(f"  {card:<18}{tier:<9}{'WON' if won else 'LOST':<5}"
              f"  {int((grp['outcome'] == 'HIT').sum())}/{len(grp)} legs")

    print("\n=== LOSS ATTRIBUTION (the declared experiment) ===")
    miss = detail.loc[detail["outcome"] != "HIT"]
    if miss.empty:
        print("  no losses")
    else:
        for b, n in miss["loss_bucket"].value_counts().items():
            print(f"  {b or 'UNCLASSIFIED':<20}{n}")

    print("\n=== CALIBRATION: stated probability vs realised ===")
    for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)):
        b = detail.loc[(detail["stated_p"] >= lo) & (detail["stated_p"] < hi)]
        if b.empty:
            continue
        hits = int((b["outcome"] == "HIT").sum())
        print(f"  stated {lo:.2f}-{hi:.2f}  n={len(b):<3} mean stated {b['stated_p'].mean():.3f}"
              f"  realised {hits / len(b):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=Path, required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    detail = grade(a.cards, a.season, a.week)
    report(detail)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
