#!/usr/bin/env python3
"""The five-part Week-7 fallback materiality report required by `6428042413`.

The auditor asked for five things reported DISTINCTLY, because conflating them
is what produced two wrong conclusions already:

  1. full football-universe population fallback
  2. nonzero POSTERIOR fallback
  3. actual rush allocator admission / materiality
  4. actual target-room entitlement / materiality
  5. downstream priced-board subset, as a SEPARATE sportsbook-availability fact

Everything is read from the durable trace in
`data/research/week7_opportunity_trace/`, extracted from paid artifact
12456298985. Two values are taken from production's own output rather than
re-derived, because re-deriving them is precisely where I went wrong:

  * the rush route is reproduced by applying production's own `_top_n_shares`
    and room-normalisation block from `simulation_v2.py` to the preserved
    `rules_rush_share`. That reproduces the auditor's 1.824446 to six decimals.

  * the target route is NOT re-derived. Production's final entitlement is
    `entitlement_tgt_share`, the end of a pipeline (`rules_tgt_share` ->
    `m38_explicit_entitlement_tgt_share` -> `post_m38_tgt_share` -> team_scale)
    that cannot be reconstructed from the sharpening function alone. I tried
    four derivations and none matched; reading production's column matches
    exactly. Where production records its own answer, read it.

Part 5 is kept strictly separate and explicitly labelled a sportsbook
availability fact. Using the priced board to bound a football population is
the §3 violation the auditor caught, and it is not repeated here.

Outcome-blind: no line, odds, price, edge, probability or result is read.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

TRACE = pathlib.Path("data/research/week7_opportunity_trace")
FALLBACK = "position_prior_only"


def _load_top_n_shares():
    """Production's own rush allocator, loaded by path.

    `scripts/simulation_v2.py` pulls heavy modeling dependencies through its
    package, and only the allocator is needed. Executing the function's own
    source keeps this honest: it is production's logic, not a paraphrase.
    """
    src = pathlib.Path("scripts/simulation_v2.py").read_text()
    ns: dict = {"np": np, "pd": pd}
    start = src.index("def _top_n_shares")
    end = src.index("def _sharpen_wr_target_shares")
    exec(src[start:end], ns)  # noqa: S102 - production source, by design
    return ns["_top_n_shares"]


def _first_finite(row, *cols) -> float:
    for c in cols:
        v = pd.to_numeric(row.get(c), errors="coerce")
        if pd.notna(v) and np.isfinite(v):
            return float(v)
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=pathlib.Path, default=TRACE)
    ap.add_argument("--out-dir", type=pathlib.Path, default=TRACE)
    ap.add_argument("--week", type=int, default=7)
    a = ap.parse_args()

    uni = pd.read_csv(
        a.trace_dir / "week7_trace__data__football_simulation_universe.csv",
        low_memory=False)
    ent = pd.read_csv(
        a.trace_dir / "week7_trace__data__target_entitlement_v1_trace.csv",
        low_memory=False)
    board = pd.read_csv(
        a.trace_dir / "week7_trace__outputs__props_priced_clean.csv",
        low_memory=False)

    pg = pd.to_numeric(uni["prior_games"], errors="coerce").fillna(-1)
    cg = pd.to_numeric(uni["current_games"], errors="coerce").fillna(-1)
    uni["zero_history"] = (pg == 0) & (cg == 0)
    uni["fallback_state"] = uni["bayes_evidence_state"].eq(FALLBACK)

    # ---- (1) full football-universe population fallback -------------------
    part1 = {
        "football_universe_rows": int(len(uni)),
        "teams": int(uni["team"].nunique()),
        "evidence_state_counts": {
            k: int(v) for k, v in uni["bayes_evidence_state"]
            .value_counts().items()},
        "position_prior_only_rows": int(uni["fallback_state"].sum()),
        "true_zero_history_rows": int(uni["zero_history"].sum()),
        "true_zero_history_teams": int(
            uni.loc[uni["zero_history"], "team"].nunique()),
        "current_games_zero_rows": int((cg == 0).sum()),
        "zero_history_equals_position_prior_only": bool(
            int(uni["zero_history"].sum()) == int(uni["fallback_state"].sum())),
    }

    # ---- (2) nonzero POSTERIOR fallback -----------------------------------
    # The correction that mattered: DEFAULTS has exact zeros for TE rush and
    # QB target, but production's posterior for those cells is small and
    # NONZERO, so the "zero-prior, cannot move allocation" bucket was an
    # artifact of the constant table.
    fb = uni.loc[uni["fallback_state"]].copy()
    post = (fb.groupby("bayes_position_group")[
        ["bayes_rush_share", "bayes_tgt_share"]].max().round(6))
    part2 = {
        "posterior_by_position_group": {
            g: {"bayes_rush_share": float(r["bayes_rush_share"]),
                "bayes_tgt_share": float(r["bayes_tgt_share"])}
            for g, r in post.iterrows()},
        "rows_with_nonzero_rush_posterior": int(
            (pd.to_numeric(fb["bayes_rush_share"], errors="coerce")
             .fillna(0) > 0).sum()),
        "rows_with_nonzero_tgt_posterior": int(
            (pd.to_numeric(fb["bayes_tgt_share"], errors="coerce")
             .fillna(0) > 0).sum()),
        "rows_with_exactly_zero_rush_posterior": int(
            (pd.to_numeric(fb["bayes_rush_share"], errors="coerce")
             .fillna(0) == 0).sum()),
        "rows_with_exactly_zero_tgt_posterior": int(
            (pd.to_numeric(fb["bayes_tgt_share"], errors="coerce")
             .fillna(0) == 0).sum()),
        "note": ("DEFAULTS lists rush_share[TE]=0.00 and tgt_share[QB]=0.0, "
                 "but production's posterior for those cells is nonzero; the "
                 "earlier zero-prior count was a constant-table artifact"),
    }

    # ---- (3) actual rush allocator admission ------------------------------
    top_n = _load_top_n_shares()
    admitted_rows = 0
    admitted_teams: set[str] = set()
    prob_sum = 0.0
    per_team = []
    for team, td in uni.groupby("team", dropna=False):
        raw = np.array([
            _first_finite(r, "rules_rush_share", "bayes_rush_share",
                          "rush_share") for _, r in td.iterrows()])
        clean = np.clip(np.nan_to_num(top_n(raw, 5), nan=0.0, posinf=0.0,
                                      neginf=0.0), 0.0, 0.95)
        raw_sum = float(clean.sum())
        used = clean.copy()
        if raw_sum > 0.95:
            used *= 0.95 / raw_sum
        probs = np.append(used, max(0.0, 1.0 - float(used.sum())))
        probs = probs / probs.sum()
        zh = uni.loc[td.index, "zero_history"].to_numpy()
        sel = zh & (clean > 0)
        if sel.any():
            admitted_rows += int(sel.sum())
            admitted_teams.add(str(team))
            tp = float(probs[:-1][sel].sum())
            prob_sum += tp
            per_team.append({"team": str(team),
                             "zero_history_admitted": int(sel.sum()),
                             "allocation_probability": round(tp, 6)})
    part3 = {
        "method": ("production's own _top_n_shares + room normalisation from "
                   "simulation_v2.py applied to preserved rules_rush_share"),
        "zero_history_rows_admitted_to_top_five": admitted_rows,
        "teams_with_admission": len(admitted_teams),
        "final_rush_allocation_probability": round(prob_sum, 6),
        "per_team": sorted(per_team, key=lambda r: -r["allocation_probability"]),
        "reproduces_audit_1_824446": bool(abs(prob_sum - 1.824446) < 5e-6),
    }

    # ---- (4) actual target-room entitlement -------------------------------
    zh_keys = set(uni.loc[uni["zero_history"], "player_clean_key"])
    ent["zero_history"] = ent["player_clean_key"].isin(zh_keys)
    ev = pd.to_numeric(ent["entitlement_tgt_share"], errors="coerce").fillna(0)
    total = float(ev.sum())
    zh_total = float(ev[ent["zero_history"]].sum())
    part4 = {
        "method": ("production's own entitlement_tgt_share, READ not "
                   "re-derived: it is the end of rules_tgt_share -> "
                   "m38_explicit_entitlement_tgt_share -> post_m38_tgt_share "
                   "-> team_scale, which the sharpening function alone cannot "
                   "reconstruct"),
        "zero_history_final_target_entitlement": round(zh_total, 6),
        "total_modeled_player_target_mass": round(total, 6),
        "zero_history_share_of_total": round(zh_total / total, 6) if total else None,
        "zero_history_players_entitlement_ge_0_05": int(
            (ev[ent["zero_history"]] >= 0.05).sum()),
        "reproduces_audit_1_264686": bool(abs(zh_total - 1.264686) < 5e-6),
    }

    # ---- (5) priced-board subset: a SPORTSBOOK AVAILABILITY FACT ----------
    bfb = board.loc[board["bayes_evidence_state"].eq(FALLBACK)]
    part5 = {
        "WARNING": ("this is sportsbook AVAILABILITY, not a football "
                    "population. A player can move the finite football room "
                    "and never receive a prop, so this must never bound "
                    "parts 1-4. Using it that way is the §3 error the "
                    "auditor caught in 6427742545."),
        "priced_rows": int(len(board)),
        "priced_players": int(board["player"].nunique()),
        "position_prior_only_priced_rows": int(len(bfb)),
        "position_prior_only_priced_players": int(bfb["player"].nunique()),
        "football_universe_fallback_rows_for_comparison":
            part1["position_prior_only_rows"],
        "ratio_football_to_priced_fallback_players": (
            round(part1["true_zero_history_rows"]
                  / max(1, bfb["player"].nunique()), 2)),
    }

    status = {
        "version": "WEEK7_FALLBACK_MATERIALITY_REPORT_V1",
        "authorized_by": "GPT_AUDIT_RESPONSE_6428042413",
        "scientific_boundary": "DESCRIPTIVE_REPORT_ONLY_NO_FITTING",
        "outcomes_inspected": False,
        "source_artifact": "12456298985",
        "source_run": "39849371821",
        "week": a.week,
        "part1_full_football_universe_population_fallback": part1,
        "part2_nonzero_posterior_fallback": part2,
        "part3_rush_allocator_materiality": part3,
        "part4_target_room_materiality": part4,
        "part5_priced_board_subset_sportsbook_availability_only": part5,
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"week{a.week}_fallback_materiality_report_v1.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"=== WEEK {a.week} FALLBACK MATERIALITY — FIVE PARTS ===\n")
    print("(1) FULL FOOTBALL UNIVERSE")
    print(f"    rows {part1['football_universe_rows']} across "
          f"{part1['teams']} teams")
    print(f"    position_prior_only {part1['position_prior_only_rows']} | "
          f"true zero-history {part1['true_zero_history_rows']} across "
          f"{part1['true_zero_history_teams']} teams | "
          f"current_games=0 {part1['current_games_zero_rows']}")
    print("\n(2) NONZERO POSTERIOR")
    for g, v in part2["posterior_by_position_group"].items():
        print(f"    {g:4s} rush {v['bayes_rush_share']:.6f}  "
              f"tgt {v['bayes_tgt_share']:.6f}")
    print(f"    exactly-zero rush posteriors: "
          f"{part2['rows_with_exactly_zero_rush_posterior']} | "
          f"exactly-zero tgt: {part2['rows_with_exactly_zero_tgt_posterior']}")
    print("\n(3) RUSH ALLOCATOR")
    print(f"    admitted {part3['zero_history_rows_admitted_to_top_five']} rows "
          f"across {part3['teams_with_admission']} teams")
    print(f"    final allocation probability "
          f"{part3['final_rush_allocation_probability']}  "
          f"(reproduces audit: {part3['reproduces_audit_1_824446']})")
    print("\n(4) TARGET ROOM")
    print(f"    zero-history entitlement "
          f"{part4['zero_history_final_target_entitlement']} of "
          f"{part4['total_modeled_player_target_mass']} = "
          f"{part4['zero_history_share_of_total']:.4%}")
    print(f"    players >= 0.05: "
          f"{part4['zero_history_players_entitlement_ge_0_05']}  "
          f"(reproduces audit: {part4['reproduces_audit_1_264686']})")
    print("\n(5) PRICED BOARD — SPORTSBOOK AVAILABILITY ONLY")
    print(f"    priced {part5['priced_rows']} rows / "
          f"{part5['priced_players']} players")
    print(f"    position_prior_only priced: "
          f"{part5['position_prior_only_priced_rows']} rows / "
          f"{part5['position_prior_only_priced_players']} players")
    print(f"    football-universe fallback is "
          f"{part5['ratio_football_to_priced_fallback_players']}x the priced "
          f"fallback player count -- which is why the priced board must not "
          f"bound the football population")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
