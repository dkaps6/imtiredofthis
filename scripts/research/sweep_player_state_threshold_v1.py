#!/usr/bin/env python3
"""Descriptive sensitivity sweep for PARTIAL_PARTICIPATION_RATIO.

Authorized by GPT audit 6425282770 §5, with explicit constraints that are part
of the design rather than a preamble:

  "This is NOT parameter fitting and must not inspect outcomes or choose a
   winner. [...] Do not promote/tune the threshold from this sweep. If results
   are sensitive, carry that uncertainty forward."

So this script:

  * reads NO outcome, result, grade, market or sportsbook column -- it cannot,
    because it only ever loads rosters, snap counts and the injury report, and
    play-by-play for opportunity counting;
  * emits NO recommendation, ranking, score or "best" threshold. There is no
    code path that selects one. A reader wanting to know which threshold is
    preferable will not find an answer here, by construction;
  * reports only what the audit asked for: state-label counts, how many
    player-weeks change label, opportunity-eligible subset counts, how many
    card rows gain or lose usable trajectory/stability/efficiency evidence,
    named examples of the largest classification changes, and whether the
    qualitative picture is stable or threshold-sensitive.

Stability is reported as a measured quantity, not a verdict on the threshold:
it says how much the card moves across the swept range, which is the
uncertainty to carry forward.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.build_player_identity_card_v1 import (
    DISPOSITION_USABLE,
    _first_col,
    PARTIAL_PARTICIPATION_RATIO,
    build_cards,
    build_participation,
    label_player_week_states,
    load_injuries,
    load_pbp,
    load_snaps,
    load_weekly_rosters,
)

SWEEP = (0.30, 0.40, 0.50, 0.60, 0.70)

# Columns whose availability defines "usable evidence" for a card row. These
# are exactly the fields the anomaly gate withholds.
_EVIDENCE_COLS = (
    "opp_per_participation_week",
    "opp_week_to_week_sd",
    "yards_per_opp",
)

# Guard against this script being pointed at market/priced data. Matching is
# TOKEN-wise, not substring: a substring test flags `yardline_100` for "line"
# and `qb_hit` for "hit", both legitimate play-by-play fields, and a guard that
# fires on real data gets loosened until it is useless.
_FORBIDDEN_TOKENS = frozenset({
    "odds", "implied", "payout", "clv", "bookmaker", "sportsbook", "edge",
    "vig", "juice", "devig", "novig",
})
_FORBIDDEN_EXACT = frozenset({
    "line", "price", "market", "closing_line", "opening_line", "graded",
    "profit", "roi", "model_proj", "mc_proj", "ml_proj", "state_proj",
    "ensemble_proj", "qb_synthesis_proj", "rb_synthesis_proj",
})


# Negative attestations -- columns whose job is to record that a boundary was
# NOT crossed -- necessarily contain the words they attest about.
# `sportsbook_inputs_used = 0` is the provenance proving no sportsbook input
# reached the card, so flagging it inverts its meaning. Exempt the class, not
# the one instance.
_ATTESTATION_SUFFIXES = ("_inputs_used", "_source_state")


def _market_shaped_columns(df: pd.DataFrame) -> list[str]:
    bad = set()
    for c in df.columns:
        n = str(c).strip().lower()
        if n.endswith(_ATTESTATION_SUFFIXES):
            continue
        if n in _FORBIDDEN_EXACT or n.endswith("_proj"):
            bad.add(str(c))
        elif _FORBIDDEN_TOKENS & set(n.split("_")):
            bad.add(str(c))
    return sorted(bad)


def _assert_no_outcome_columns(df: pd.DataFrame, name: str) -> None:
    """Refuse market/priced inputs.

    Deliberately NOT applied to play-by-play. Real nflverse PBP legitimately
    carries game outcomes (`result`, `home_score`, `total`) because it is a
    record of games that happened; the card builder never reads them, and
    banning the frame for containing them would be theatre. What matters is
    that no market/priced column reaches this descriptive sweep, and that the
    emitted cards carry none -- both checked.
    """
    bad = _market_shaped_columns(df)
    if bad:
        raise RuntimeError(
            f"{name} carries market-shaped columns {bad}; this sweep is "
            "descriptive and must not see them")


def _evidence_available(cards: pd.DataFrame) -> pd.Series:
    have = pd.Series(True, index=cards.index)
    for c in _EVIDENCE_COLS:
        have &= pd.to_numeric(cards[c], errors="coerce").notna()
    return have


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--weeks", default="1-7")
    ap.add_argument("--prior-seasons", default="2025")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/research/player_identity"))
    a = ap.parse_args()

    lo, _, hi = a.weeks.partition("-")
    weeks = list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    priors = [int(x) for x in a.prior_seasons.split(",") if x.strip()]
    seasons = sorted({a.season, *priors})

    rosters = load_weekly_rosters(seasons)
    snaps = load_snaps(seasons)
    injuries = load_injuries(seasons)
    pbp = load_pbp(seasons)
    # PBP is exempt by design -- see _assert_no_outcome_columns. The builder
    # reads an explicit column list from it and never touches scores.
    for nm, df in (("rosters", rosters), ("snaps", snaps),
                   ("injuries", injuries)):
        _assert_no_outcome_columns(df, nm)

    participation, bridge = build_participation(rosters, snaps)
    if bridge.get("bridge_state") != "ok":
        raise RuntimeError(f"participation bridge not ok: {bridge}")
    untrusted = set(bridge.get("partial_bridge_gsis_ids", []))

    rows, state_tables, per_pw = [], {}, {}
    for r in SWEEP:
        pw = label_player_week_states(participation, injuries, partial_ratio=r)
        cards, diag = build_cards(pbp, rosters, pw, a.season, weeks, priors,
                                  untrusted_denominator_ids=untrusted)
        _assert_no_outcome_columns(cards, f"cards@ratio={r:.2f}")
        if not (pd.to_numeric(cards["sportsbook_inputs_used"],
                              errors="coerce").fillna(1) == 0).all():
            raise RuntimeError("cards report sportsbook input use")
        elig = cards["can_receive_opportunity"].astype(bool)
        have = _evidence_available(cards)
        counts = pw["player_state"].value_counts().to_dict()
        state_tables[f"{r:.2f}"] = {k: int(v) for k, v in counts.items()}
        per_pw[f"{r:.2f}"] = dict(zip(
            zip(pw["season"].astype(int), pw["week"].astype(int), pw["gsis_id"]),
            pw["player_state"]))
        rows.append({
            "partial_participation_ratio": r,
            "is_current_default": bool(
                abs(r - PARTIAL_PARTICIPATION_RATIO) < 1e-12),
            "participation_weeks": int(len(pw)),
            "weeks_usable": int((pw["gate_disposition"]
                                 == DISPOSITION_USABLE).sum()),
            "weeks_withheld": int((pw["gate_disposition"]
                                   != DISPOSITION_USABLE).sum()),
            **{f"state_{k}": int(v) for k, v in counts.items()},
            "cards": int(len(cards)),
            "cards_opportunity_eligible": int(elig.sum()),
            "cards_with_usable_evidence": int(have.sum()),
            "eligible_cards_with_usable_evidence": int((have & elig).sum()),
            "gate_weeks_withheld_total": int(
                diag.get("gate_weeks_withheld_total", 0)),
        })
        print(f"[sweep] ratio={r:.2f} withheld_weeks={rows[-1]['weeks_withheld']} "
              f"eligible_cards_with_evidence="
              f"{rows[-1]['eligible_cards_with_usable_evidence']}")

    table = pd.DataFrame(rows)

    # Monotonicity is a correctness check on the sweep itself: a looser
    # threshold must flag at least as many weeks abnormal. A violation means
    # the labeller is not doing what it claims.
    wh = table["weeks_withheld"].tolist()
    if any(b < x for x, b in zip(wh, wh[1:])):
        raise RuntimeError(
            f"withheld weeks not monotonic across the sweep: {wh}; a looser "
            "threshold cannot flag fewer weeks")

    # Label churn between adjacent thresholds, and named examples of it.
    # The audit asked for NAMED examples. Weekly rosters carry `full_name`,
    # not `player_name`, so a strict check for the latter silently produced a
    # blank column -- a deliverable that passes every gate and answers nothing.
    r_name = _first_col(rosters, ["player_name", "full_name",
                                  "player_display_name", "football_name"])
    r_pos = _first_col(rosters, ["position", "depth_chart_position"])
    if not r_name:
        raise RuntimeError(
            f"roster authority exposes no name column (has: "
            f"{sorted(c for c in rosters.columns if 'name' in str(c))}); the "
            "audit asked for named examples and this must not emit blanks")
    name_by = (rosters.dropna(subset=["gsis_id"])
               .drop_duplicates("gsis_id")
               .set_index("gsis_id")
               .rename(columns={r_name: "name", r_pos: "position"})
               [["name"] + (["position"] if r_pos else [])]
               .to_dict("index"))

    keys = [f"{r:.2f}" for r in SWEEP]
    churn, examples = [], []
    for k0, k1 in zip(keys, keys[1:]):
        a0, a1 = per_pw[k0], per_pw[k1]
        changed = [k for k in a0 if a0[k] != a1.get(k)]
        churn.append({"from_ratio": k0, "to_ratio": k1,
                      "player_weeks_changed": len(changed)})
        # "Largest classification changes" means the players whose state moves
        # most across the pair, not the first fifteen in id order. Rank by how
        # many of a player's weeks flip, ties broken by id for reproducibility.
        by_player: dict[str, list] = {}
        for key in changed:
            by_player.setdefault(key[2], []).append(key)
        ranked = sorted(by_player.items(),
                        key=lambda kv: (-len(kv[1]), str(kv[0])))[:10]
        for pid, keys in ranked:
            info = name_by.get(pid, {})
            for key in sorted(keys, key=lambda t: (t[0], t[1])):
                examples.append({
                    "from_ratio": k0, "to_ratio": k1,
                    "season": key[0], "week": key[1], "gsis_id": pid,
                    "player": info.get("name", ""),
                    "position": info.get("position", ""),
                    "weeks_changed_for_this_player": len(keys),
                    "state_at_from": a0[key], "state_at_to": a1.get(key, ""),
                })

    base = table["eligible_cards_with_usable_evidence"]
    spread = int(base.max() - base.min())
    denom = int(base.max()) or 1
    rel = spread / denom
    stability = ("STABLE_ACROSS_SWEPT_RANGE" if rel < 0.05
                 else "MODERATELY_THRESHOLD_SENSITIVE" if rel < 0.20
                 else "THRESHOLD_SENSITIVE")

    ex = pd.DataFrame(examples)
    if not ex.empty and not ex["player"].astype(str).str.strip().any():
        raise RuntimeError(
            "every named example came out blank; the name lookup is not "
            "resolving against the roster authority")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out_dir / "player_state_threshold_sweep_v1.csv", index=False)
    pd.DataFrame(churn).to_csv(
        a.out_dir / "player_state_threshold_sweep_v1_churn.csv", index=False)
    ex.to_csv(a.out_dir / "player_state_threshold_sweep_v1_examples.csv",
              index=False)

    status = {
        "version": "PLAYER_STATE_THRESHOLD_SWEEP_V1",
        "authorized_by": "GPT_AUDIT_6425282770_SECTION_5",
        "scientific_boundary": "DESCRIPTIVE_SENSITIVITY_ONLY_NO_FITTING",
        "outcomes_inspected": False,
        "threshold_recommended": None,
        "no_winner_selected": True,
        "season": a.season,
        "target_weeks": weeks,
        "prior_seasons": priors,
        "swept_ratios": list(SWEEP),
        "current_default_ratio": PARTIAL_PARTICIPATION_RATIO,
        "state_counts_by_ratio": state_tables,
        "label_churn_between_adjacent_ratios": churn,
        "eligible_cards_with_usable_evidence_by_ratio": {
            f"{r:.2f}": int(v) for r, v in zip(
                table["partial_participation_ratio"], base)},
        "eligible_evidence_spread_cards": spread,
        "eligible_evidence_spread_fraction": round(rel, 4),
        "qualitative_stability": stability,
        "sportsbook_inputs_used": 0,
    }
    (a.out_dir / "player_state_threshold_sweep_v1_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n=== PLAYER-STATE THRESHOLD SENSITIVITY SWEEP V1 ===")
    print(table.to_string(index=False))
    print("\nlabel churn between adjacent ratios:")
    print(pd.DataFrame(churn).to_string(index=False))
    print(f"\neligible cards with usable evidence: spread={spread} "
          f"({rel:.2%} of max) -> {stability}")
    print("\nNo threshold is recommended and none was selected. If the state "
          "above is not STABLE_ACROSS_SWEPT_RANGE, that uncertainty is the "
          "result and must be carried forward rather than resolved here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
