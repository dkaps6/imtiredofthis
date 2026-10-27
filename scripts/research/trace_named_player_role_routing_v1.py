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
    # An unlabelled WR must still fall through to 1.0. Audit 6445134492
    # correctly objected that recording this as a boolean without raising is
    # not a guard at all.
    tail = block.split('elif pos in {"RB", "FB"}')[-1]
    if re.search(r"^\s*else\s*:", tail, re.M):
        raise RuntimeError(
            "an else branch now follows the RB/FB arm, so an unlabelled WR no "
            "longer falls through to 1.0; the published finding is stale")
    checks["unlabelled_wr_falls_through_to_1.0"] = True
    # Prove only two perimeter labels are assigned by parsing the assignment
    # statements, not by the absence of a `perim[2]` string, which a rewrite
    # could trivially evade.
    lab = src.split("def _wr_role_labels(", 1)[1].split("\ndef ", 1)[0]
    assigned = set(re.findall(r'out\[\(team, _key\([^)]*\)\)\] = "(\w+)"', lab))
    if assigned != {"SLOT", "WR1", "WR1_5"}:
        raise RuntimeError(
            f"_wr_role_labels now assigns {sorted(assigned)}; the published "
            "slot-plus-top-two finding must be revised")
    idx = set(re.findall(r"perim\[(\d+)\]", lab))
    if idx - {"0", "1"}:
        raise RuntimeError(
            f"_wr_role_labels now indexes perim{sorted(idx)}; more than two "
            "perimeter WRs are labelled")
    checks["only_two_perimeter_wrs_labelled"] = True
    checks["labels_assigned_exactly"] = sorted(assigned)
    return checks


# Every gate multiplier, so the observed rule-layer ratio can be DECODED
# rather than guessed. Audit 6445134492 noticed that 1.265 is exactly
# 1.15 * 1.10 and 1.200 is exactly the RB zone multiplier; because
# apply_rules_to_metrics() performs no normalization, the ratio IS the
# multiplier, so the gate state is recoverable.
GATE_SPECS = {
    "te_target_mult":     (("coverage_zone_rate >= 0.60", 1.15),
                           ("middle_open_rate >= 0.50", 1.10)),
    "rb_rec_target_mult": (("coverage_zone_rate >= 0.60", 1.20),
                           ("pressure_mismatch > 0.075", 1.10)),
    "wr1_target_mult":    (("coverage_man_rate >= 0.50", 0.95),),
    "wr1_5_target_mult":  (("coverage_man_rate >= 0.50", 1.15),),
    "slot_target_mult":   (("coverage_man_rate >= 0.50", 1.05),
                           ("middle_open_rate >= 0.50", 1.10)),
}


def decode_gates(observed: dict[str, float], tol: float = 1e-9) -> dict:
    """Recover the gate state from observed rule-layer multipliers.

    Sound only because the rules layer applies no normalization, so a row's
    rules/bayes ratio is the multiplier itself. Each role's achievable products
    are enumerated and checked for distinctness; a decode is reported only when
    the observed value matches exactly one product. Shared booleans are then
    cross-checked across position groups, which is independent corroboration
    rather than a single-group inference.

    RESIDUAL CONDITION, stated rather than assumed away: injury redistribution
    replaces `base_tgt` outright and injury limitation halves it, so either
    would break the exact match. The decode therefore holds for rows carrying
    neither, which the uniform exact ratios within each group evidence but do
    not prove.
    """
    from itertools import product as _p
    per_role, booleans = {}, {}
    for mult, gates in GATE_SPECS.items():
        rows = []
        for combo in _p([False, True], repeat=len(gates)):
            v, fired = 1.0, []
            for (g, m), on in zip(gates, combo):
                if on:
                    v *= m
                    fired.append(g)
            rows.append((round(v, 9), fired, dict(zip([g for g, _ in gates],
                                                      combo))))
        vals = [r[0] for r in rows]
        entry = {"achievable_products": sorted(set(vals)),
                 "products_all_distinct": len(set(vals)) == len(vals)}
        if mult in observed and entry["products_all_distinct"]:
            hit = [r for r in rows if abs(r[0] - observed[mult]) <= tol]
            if len(hit) == 1:
                entry["observed"] = observed[mult]
                entry["decoded_fired"] = hit[0][1]
                entry["decoded_state"] = hit[0][2]
                for g, on in hit[0][2].items():
                    booleans.setdefault(g, set()).add(bool(on))
        per_role[mult] = entry
    consistent = {g: (list(v)[0] if len(v) == 1 else "CONTRADICTION")
                  for g, v in booleans.items()}
    return {
        "per_role": per_role,
        "gate_state": consistent,
        "cross_group_consistent": all(v != "CONTRADICTION"
                                      for v in consistent.values()),
        "corroboration": ("coverage_zone_rate is decoded independently by the "
                          "TE and RB groups and coverage_man_rate independently "
                          "by WR1 and WR1_5; agreement is corroboration, not a "
                          "single-group inference"),
        "derived_consequences": {
            "volatility_mult": (
                1.0 if consistent.get("pressure_mismatch > 0.075") is False
                else "NOT_ESTABLISHED"),
            "pass_eff_mult_bound": (
                ["1.00", "1.01"]
                if consistent.get("pressure_mismatch > 0.075") is False
                else "NOT_ESTABLISHED"),
            "pass_eff_note": (
                "pressure is not > 0.075, so pass_eff_mult is not 0.98. It is "
                "1.00, or 1.01 if pressure < -0.075, which the preserved "
                "artifact cannot separate because rules_pass_eff_mult -- which "
                "apply_rules_to_metrics DOES emit -- was not preserved. Field 8 "
                "is therefore BOUNDED, not resolved."),
        },
    }


def allocator_chain(team: str = "SEA") -> dict:
    """Field 7 / 10 from the REAL allocator, not inferred from a ratio.

    Audit 6445134492 blocker 1: apply_rules_to_metrics() performs no team-room
    normalization, so calling the rule-layer ratio "normalization" was wrong,
    and summing selected rows was not a finite-room measurement. The actual
    chain is rules_tgt_share -> post_m38_tgt_share -> entitlement_tgt_share
    with a residual, and it is preserved.
    """
    path = TRACE / "week7_trace__data__target_entitlement_v1_trace.csv"
    audit = TRACE / "week7_trace__data__team_target_pool_audit.csv"
    if not path.exists():
        return {"available": False,
                "blocker": f"{path} not preserved; fields 7/10 NOT ESTABLISHED"}
    e = pd.read_csv(path, low_memory=False)
    t = e.loc[e["team"].eq(team)].copy()
    for c in ("rules_tgt_share", "post_m38_tgt_share",
              "entitlement_tgt_share"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["m38_delta_pct"] = ((t["post_m38_tgt_share"] / t["rules_tgt_share"]) - 1.0)
    t["net_rules_to_entitlement"] = (t["entitlement_tgt_share"]
                                     / t["rules_tgt_share"])
    room = {}
    if audit.exists():
        a = pd.read_csv(audit, low_memory=False)
        r = a.loc[a["team"].eq(team)]
        if len(r):
            room = {k: (float(r.iloc[0][k])
                        if pd.api.types.is_number(r.iloc[0][k]) else
                        r.iloc[0][k])
                    for k in ("entitlement_sum", "residual_share",
                              "entitlement_players", "entitlement_gap")
                    if k in a.columns}
    return {
        "available": True,
        "stages": ["rules_tgt_share", "post_m38_tgt_share",
                   "entitlement_tgt_share"],
        "team_sums": {c: float(t[c].sum()) for c in
                      ("rules_tgt_share", "post_m38_tgt_share",
                       "entitlement_tgt_share")},
        "m38_conserves_team_mass": bool(
            abs(t["rules_tgt_share"].sum()
                - t["post_m38_tgt_share"].sum()) < 1e-9),
        "team_room_audit": room,
        "rows": t[["player", "position", "rules_tgt_share",
                   "post_m38_tgt_share", "entitlement_tgt_share",
                   "m38_delta_pct", "net_rules_to_entitlement"]]
        .sort_values("entitlement_tgt_share", ascending=False)
        .to_dict("records"),
    }


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
    # NOT normalization. apply_rules_to_metrics() performs no team-room
    # normalization, so this ratio is the rule-layer matchup multiplier itself
    # (audit 6445134492 blocker 1). Calling it a group scale was wrong, and it
    # is what made the gate state look unrecoverable.
    rule_layer = {
        "quantity": "rules_tgt_share / bayes_tgt_share",
        "what_it_is": ("the rule-layer matchup multiplier, applied to the "
                       "posterior inside apply_rules_to_metrics, which performs "
                       "NO team-room normalization"),
        "what_it_is_not": ("a per-position-group room normalization, and not a "
                           "final allocated share"),
        "per_position_group_observed_multiplier": {k: v for k, v in grp.items()},
    }
    observed_mult = {}
    for pos, vals in grp.items():
        if len(vals) == 1:
            key = {"TE": "te_target_mult", "RB": "rb_rec_target_mult",
                   "FB": "rb_rec_target_mult"}.get(str(pos))
            if key:
                observed_mult[key] = float(vals[0])
    for role_name, mult_name in (("WR1", "wr1_target_mult"),
                                 ("WR1_5", "wr1_5_target_mult"),
                                 ("SLOT", "slot_target_mult")):
        r = out.loc[out["canonical_role_label"].eq(role_name),
                    "ratio_rules_over_bayes"].dropna()
        if len(r) == 1:
            observed_mult[mult_name] = float(round(r.iloc[0], 9))
    decoded = decode_gates(observed_mult)
    chain = allocator_chain("SEA")

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
            "te_and_rb_gates_DECODED": decoded["gate_state"],
            "decode_basis": ("the rules layer applies no normalization, so the "
                             "rules/bayes ratio IS the multiplier. Each role's "
                             "achievable products are distinct, so an exact "
                             "match identifies the gate state uniquely."),
            "decode_cross_group_consistent": decoded["cross_group_consistent"],
            "decode_residual_condition": (
                "holds for rows with no injury redistribution and no injury "
                "limitation, either of which would break the exact match"),
            "shared_pass_eff_mult_movement": "BOUNDED_NOT_RESOLVED",
            "pass_eff_bound": decoded["derived_consequences"],
        },
        "rule_layer_multiplier": rule_layer,
        "observed_multipliers": observed_mult,
        "gate_decode": decoded,
        "allocator_chain_fields_7_and_10": chain,
        "direct_wr_coverage_target_rule_layer_delta": {
            "all_SEA_WRs_week7": 0.0,
            "SCOPE": ("this is ONLY the direct WR coverage-target rule-layer "
                      "delta. It is NOT total opponent attribution and does "
                      "NOT make any WR globally opponent-insensitive (audit "
                      "6445134492 blocker 3). Shared pass_eff_mult can still "
                      "move YPT, and the M38/entitlement stages move a WR's "
                      "share for teammate-structural reasons, which is "
                      "indirect opponent exposure."),
            "distinction": ("Kupp's zero is STRUCTURAL -- as an unlabelled WR "
                            "he is ineligible whatever the defense does. "
                            "JSN's zero is CIRCUMSTANTIAL -- he was eligible "
                            "and the man gate did not fire."),
            "WHY_THIS_MATTERS_MORE_THAN_V1_SAID": (
                "the zone gate DID fire for KC. No WR role has a zone route "
                "at all -- wr1 and wr1_5 key only on coverage_man_rate -- so "
                "zone coverage, the most common NFL coverage family, moved "
                "every TE (x1.265) and every RB/FB (x1.200) and could not move "
                "any WR. V1 reported the WR zero without noting that the "
                "defense was in a configuration which paid out to every "
                "position-routed player."),
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
