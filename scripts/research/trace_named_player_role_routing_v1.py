#!/usr/bin/env python3
"""Compact named-player role-routing trace: JSN / Kupp / Barner.

Authorized by GPT audit `6441088661`, disposition
`OPPONENT_TRANSMISSION_GENERALIZATION_V2_CLEARED__COMPACT_NAMED_PLAYER_ROLE_ROUTING_TRACE_AUTHORIZED`,
reconfirmed `6443421541`. Answers the mechanism question:

    How does canonical role assignment determine which opponent-transmission
    path a named player is actually eligible to receive?

Read-only. No outcome, market, line, odds or result is read. Nothing is fitted,
scored, ranked or promoted; no coefficient, sign or threshold is selected.

BASIS. The durable Week-7 pregame artifact at
`data/research/week7_opportunity_trace/`, which is the same pregame basis the
cleared Walker decomposition used. Week 7 is used to IDENTIFY routing, never to
score it.

POPULATION. In 2026 Week 7 the three authorized names are all Seattle players
facing Kansas City, which is the same-offense / same-opponent contrast the audit
asked for: identity and role vary while the defensive environment is held fixed.
(Kenneth Walker III is a Kansas City back facing the Seattle defense in that
same game, so the cleared Walker case and this trace are opposite sides of one
matchup. Walker is not redone here.)

WHAT CAN AND CANNOT BE CERTIFIED FROM THE ARCHIVE. The preservation filter that
built the Week-7 trace kept opponent column NAMES and dropped their VALUES, and
did not keep `rules_*_mult` or `rules_ypt`. So:

  * role label, eligibility, pre/post opportunity share and the
    team-normalization effect are read directly and are certified;
  * whether the man-coverage gate fired is certified NEGATIVE by an exact
    argument: WR1 would take x0.95 and WR1_5 x1.15, so if the gate had fired
    their rules/bayes ratios would differ. They are identical, so it did not;
  * whether the TE and RB coverage/pressure gates fired is NOT ESTABLISHED,
    because every member of a position group receives the same multiplier and a
    uniform within-group multiplier is indistinguishable from that group's
    normalization scale. This is reported as unresolved rather than guessed;
  * shared `pass_eff_mult` movement is NOT ESTABLISHED: `rules_ypt` was not
    preserved.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

TRACE = Path("data/research/week7_opportunity_trace")
UNIVERSE = TRACE / "week7_trace__data__football_simulation_universe.csv"

NAMED = ("Jaxon Smith-Njigba", "Cooper Kupp", "AJ Barner")

# The role -> gate routing, read off the if/elif chain in
# simulation_rules.py. The asymmetry is the finding: WR branches key on the
# COMPUTED role label, TE and RB branches key on POSITION.
ROUTING = {
    "WR1":   ("wr1_target_mult",    ("coverage_man_rate >= 0.50 -> x0.95",)),
    "WR1_5": ("wr1_5_target_mult",  ("coverage_man_rate >= 0.50 -> x1.15",)),
    "SLOT":  ("slot_target_mult",   ("coverage_man_rate >= 0.50 -> x1.05",
                                     "middle_open_rate >= 0.50 -> x1.10")),
    "_TE":   ("te_target_mult",     ("coverage_zone_rate >= 0.60 -> x1.15",
                                     "middle_open_rate >= 0.50 -> x1.10")),
    "_RB":   ("rb_rec_target_mult", ("coverage_zone_rate >= 0.60 -> x1.20",
                                     "pressure mismatch > 0.075 -> x1.10")),
}

# Role-specific opponent conversion signals that exist upstream and are
# referenced by no production module at all.
STRANDED_EFFICIENCY = {
    "WR": "ypt_allowed_wr", "SWR": "ypt_allowed_slot",
    "TE": "ypt_allowed_te", "RB": "ypt_allowed_rb",
}

FORBIDDEN = ("line", "odds", "price", "edge", "prob", "actual", "result",
             "win", "roi", "grade")


def assert_routing_matches_source() -> dict:
    """Fail closed if the if/elif chain this trace describes has changed."""
    src = Path("scripts/modeling/simulation_rules.py").read_text(encoding="utf-8")
    block = src.split("tgt_mult = 1.0", 1)[1].split("base_tgt *= tgt_mult", 1)[0]
    checks = {}
    for role, (mult, _) in ROUTING.items():
        if mult not in block:
            raise RuntimeError(f"{mult} no longer in the target-mult chain")
    # WR branches must be keyed on the role label, TE/RB on position.
    if not re.search(r'if role == "WR1"', block):
        raise RuntimeError("WR1 branch is no longer keyed on the role label")
    if not re.search(r'elif pos == "TE"', block):
        raise RuntimeError("TE branch is no longer keyed on position")
    if not re.search(r'elif pos in \{"RB", "FB"\}', block):
        raise RuntimeError("RB branch is no longer keyed on position")
    checks["wr_keyed_on_role_label"] = True
    checks["te_rb_keyed_on_position"] = True
    # An unlabelled WR must still fall through to 1.0.
    checks["unlabelled_wr_falls_through_to_1.0"] = (
        'else' not in block.split('elif pos in {"RB", "FB"}')[-1])
    lab = src.split("def _wr_role_labels(", 1)[1].split("\ndef ", 1)[0]
    if "perim[2]" in lab:
        raise RuntimeError("a third perimeter WR label now exists")
    checks["only_two_perimeter_wrs_labelled"] = True
    return checks


def eligible_gates(role: str, position: str) -> tuple[str, tuple[str, ...]]:
    """Resolve the if/elif chain exactly as production does."""
    r, pos = str(role or "").strip(), str(position or "").upper()
    if r in ("WR1", "WR1_5", "SLOT"):
        return ROUTING[r]
    if pos == "TE":
        return ROUTING["_TE"]
    if pos in ("RB", "FB"):
        return ROUTING["_RB"]
    return ("", ())


def build(u: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    sea = u.loc[u["team"].eq("SEA")
                & u["position"].isin(["WR", "SWR", "TE", "RB", "FB"])].copy()
    sea["ratio_rules_over_bayes"] = (
        pd.to_numeric(sea["rules_tgt_share"], errors="coerce")
        / pd.to_numeric(sea["bayes_tgt_share"], errors="coerce"))

    rows = []
    for _, p in sea.iterrows():
        role = "" if pd.isna(p.get("rules_role")) else str(p["rules_role"])
        mult, gates = eligible_gates(role, p["position"])
        rows.append({
            # 1 player / team / position
            "player": p["player"], "team": p["team"],
            "opponent": p["opponent"], "position": p["position"],
            "depth_role": p["role"],
            # 2 canonical role label actually assigned
            "canonical_role_label": role or "NONE_ASSIGNED",
            "role_label_source": ("computed from tgt_share rank by "
                                  "_wr_role_labels" if p["position"] in
                                  ("WR", "SWR") else "position branch"),
            # 3 + 4 eligible gates
            "routed_to_multiplier": mult or "NONE",
            "eligible_gates": " | ".join(gates) or "NONE",
            "receives_any_opponent_target_channel": bool(mult),
            # 6 pre-gate -> post-gate opportunity share
            "role_ranking_key_raw_tgt_share": p["tgt_share"],
            "pre_gate_bayes_tgt_share": p["bayes_tgt_share"],
            "post_gate_rules_tgt_share": p["rules_tgt_share"],
            "ratio_rules_over_bayes": p["ratio_rules_over_bayes"],
            # 9 role-specific efficiency signal, present upstream, stranded
            "role_specific_efficiency_signal": STRANDED_EFFICIENCY.get(
                str(p["position"]).upper(), ""),
            "that_signal_consumed_in_production": False,
            "bayes_evidence_state": p["bayes_evidence_state"],
            "is_named_target": p["player"] in NAMED,
        })
    out = pd.DataFrame(rows).sort_values(
        ["position", "pre_gate_bayes_tgt_share"], ascending=[True, False])

    # 5 which gates actually fired -- certified only where an exact argument
    # exists.
    wr = out.loc[out["position"].isin(["WR", "SWR"])]
    wr_ratios = sorted(set(wr["ratio_rules_over_bayes"].round(9).dropna()))
    labelled = wr.loc[wr["canonical_role_label"].isin(["WR1", "WR1_5"])]
    man_gate_fired = None
    if len(labelled) >= 2 and len(wr_ratios) == 1:
        man_gate_fired = False  # x0.95 vs x1.15 would have split the ratios

    # 7 finite-room / team normalization
    grp = (out.groupby("position")["ratio_rules_over_bayes"]
           .agg(lambda s: sorted(set(s.round(9).dropna()))).to_dict())
    room = {
        "team_pre_gate_bayes_tgt_share_sum": float(
            pd.to_numeric(out["pre_gate_bayes_tgt_share"],
                          errors="coerce").sum()),
        "team_post_gate_rules_tgt_share_sum": float(
            pd.to_numeric(out["post_gate_rules_tgt_share"],
                          errors="coerce").sum()),
        "per_position_group_scale": {k: v for k, v in grp.items()},
        "interpretation": ("the ratio is uniform WITHIN each position group, "
                           "so the group scale and any fired within-group "
                           "multiplier are not separable for TE/RB"),
    }

    kupp = out.loc[out["player"].eq("Cooper Kupp")]
    sha = out.loc[out["player"].eq("Rashid Shaheed")]
    # The label is decided by the RAW tgt_share rank, not by the bayes
    # posterior that is actually scaled, so the exclusion margin must be
    # measured on the ranking key. Reporting it on bayes_tgt_share would quote
    # a number the decision never used.
    margin = margin_bayes = None
    if len(kupp) and len(sha):
        margin = float(
            pd.to_numeric(sha["role_ranking_key_raw_tgt_share"].iloc[0])
            - pd.to_numeric(kupp["role_ranking_key_raw_tgt_share"].iloc[0]))
        margin_bayes = float(sha["pre_gate_bayes_tgt_share"].iloc[0]
                             - kupp["pre_gate_bayes_tgt_share"].iloc[0])

    status = {
        "version": "NAMED_PLAYER_ROLE_ROUTING_TRACE_V1",
        "authorized_by": "GPT_AUDIT_6441088661",
        "reconfirmed_by": "GPT_CONTROL_6443421541",
        "mechanism_question": ("how does canonical role assignment determine "
                              "which opponent-transmission path a named player "
                              "is eligible to receive?"),
        "scientific_boundary": "READ_ONLY_ROLE_ROUTING_TRACE_NO_FITTING_NO_SCORING",
        "outcomes_inspected": False,
        "sportsbook_inputs_used": 0,
        "week7_used_to_score_anything": False,
        "walker_redone": False,
        "basis": str(UNIVERSE),
        "population": ("2026 week 7, SEA offense vs KC -- same offense, same "
                       "opponent, roles vary. Walker is the KC back facing SEA "
                       "in the same game and is not re-examined."),
        "routing_source_checks": assert_routing_matches_source(),
        "THE_ROUTING_ASYMMETRY": (
            "in simulation_rules.py the WR branches key on the COMPUTED role "
            "label while the TE and RB branches key on POSITION. So every TE "
            "and every RB receives an opponent-coverage target channel by "
            "position, but a WR receives one only if _wr_role_labels assigned "
            "him SLOT, WR1 or WR1_5 -- that is, slot plus the top two "
            "perimeter WRs by target share. Any other WR falls through the "
            "whole chain to tgt_mult = 1.0, and no diagnostic records it."),
        "named_player_finding": {
            "Jaxon Smith-Njigba": ("labelled WR1 -> eligible for "
                                   "wr1_target_mult"),
            "Cooper Kupp": ("NO label -> structurally ineligible for every "
                            "opponent-coverage target gate, while SEA's third "
                            "tight end is eligible by position"),
            "AJ Barner": ("no label, but eligible via the TE position branch "
                          "-> te_target_mult"),
        },
        "kupp_exclusion_margin_on_ranking_key_raw_tgt_share": margin,
        "kupp_exclusion_margin_on_bayes_posterior_for_reference": margin_bayes,
        "RANKING_KEY_MISMATCH": (
            "_wr_role_labels sorts perimeter WRs by the RAW tgt_share with a "
            "0.0 default, while the quantity production actually scales is "
            "bayes_tgt_share. The two disagree. Montorie Foster is the clean "
            "case: his raw tgt_share is NaN -> 0.0, so he sorts last and gets "
            "no label, yet his bayes posterior (0.165842) is the second "
            "largest in the SEA receiving room -- larger than both the "
            "labelled WR1_5 and Cooper Kupp. So the role ranking and the "
            "allocated share are keyed on different quantities."),
        "kupp_exclusion_is_structural_not_circumstantial": True,
        "gate_firing": {
            "man_coverage_gate_fired_for_KC": man_gate_fired,
            "man_gate_argument": ("WR1 takes x0.95 and WR1_5 x1.15, so a fired "
                                  "gate would split their rules/bayes ratios. "
                                  f"observed WR ratios: {wr_ratios}. identical "
                                  "=> not fired."),
            "te_and_rb_gates_fired": "NOT_ESTABLISHED",
            "te_rb_blocker": ("a uniform within-group multiplier is "
                              "indistinguishable from the group normalization "
                              "scale in the preserved artifact; rules_*_mult "
                              "was not preserved"),
            "shared_pass_eff_mult_movement": "NOT_ESTABLISHED",
            "pass_eff_blocker": "rules_ypt was not preserved",
        },
        "finite_room": room,
        "final_opponent_attributable_movement": {
            "all_SEA_WRs_week7": 0.0,
            "distinction": ("Kupp's zero is STRUCTURAL -- he is ineligible "
                            "whatever the defense does. JSN's zero is "
                            "CIRCUMSTANTIAL -- he was eligible and the gate "
                            "did not fire. These are different failures and "
                            "must not be pooled."),
        },
        "separate_defect_not_merged": (
            "Montorie Foster is position_prior_only (zero history) and carries "
            "a larger rules_tgt_share than Cooper Kupp. That is the "
            "zero-history entitlement defect, which the audit requires be kept "
            "a distinct cause. It is recorded here only as an observation."),
        "not_established": [
            "whether any TE/RB coverage or pressure gate fired",
            "shared pass_eff_mult movement",
            "how often any gate fires league-wide",
            "whether consuming the stranded ypt_allowed_* family would improve "
            "any projection",
        ],
    }
    return out, status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=Path, default=UNIVERSE)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/research/named_player_routing"))
    a = ap.parse_args()

    u = pd.read_csv(a.universe, low_memory=False)
    bad = [c for c in u.columns
           if any(k in c.lower() for k in FORBIDDEN)]
    if bad:
        raise RuntimeError(f"universe carries outcome/decision columns {bad}")

    out, status = build(u)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "named_player_role_routing_v1"
    out.to_csv(a.out_dir / f"{stem}.csv", index=False)
    (a.out_dir / f"{stem}_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")

    print("=== NAMED-PLAYER ROLE ROUTING TRACE V1 — SEA vs KC, 2026 W7 ===")
    cols = ["player", "position", "depth_role", "canonical_role_label",
            "routed_to_multiplier", "receives_any_opponent_target_channel",
            "role_ranking_key_raw_tgt_share", "pre_gate_bayes_tgt_share",
            "ratio_rules_over_bayes"]
    print(out[cols].to_string(index=False))
    print("\nROUTING ASYMMETRY: WR branches key on the computed role label;")
    print("TE and RB branches key on position. An unlabelled WR falls through")
    print("the entire chain to tgt_mult = 1.0, silently.\n")
    print(f"Cooper Kupp misses the WR1_5 label by "
          f"{status['kupp_exclusion_margin_on_ranking_key_raw_tgt_share']:.6f} "
          "on the ranking key and is")
    print("therefore ineligible for EVERY opponent-coverage target gate, while")
    print("SEA's third-string tight end is eligible by position.\n")
    print(f"man-coverage gate fired for KC: "
          f"{status['gate_firing']['man_coverage_gate_fired_for_KC']}")
    print(f"TE/RB gates: {status['gate_firing']['te_and_rb_gates_fired']}")
    print("\nNo coefficient, threshold or sign is selected here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
