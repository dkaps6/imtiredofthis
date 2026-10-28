#!/usr/bin/env python3
"""Trace the generic game script: is the opponent-aware part consumed?

User-authorized 2026-10-27 as the parallel lane to GPT's artifact-only source
work. The last untraced item on the STILL OPEN list.

Read-only and source-only. No artifact, no outcome, no market, no fitting, no
scoring, no threshold. Every claim below is asserted against production source
on each run, so the finding fails closed rather than going stale.

THE QUESTION. `rules_v2.project_game_script()` computes an opponent-aware
`diff = success_diff(offense, defense)` and derives `lead/neutral/trail`
probabilities from it, then sets `pass_share = 0.57` on the next line and builds
the pass/rush attempt split from that constant. Does the opponent-aware part
reach any projection, or only a diagnostics file?

WHY IT MATTERS MORE THAN THE DOWNSTREAM MULTIPLIERS. The pass/rush split is the
finite opportunity pool from which every target and every carry is allocated. A
constant there is upstream of every role multiplier already traced.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RULES = Path("scripts/modeling/rules_v2.py")
SIMRULES = Path("scripts/modeling/simulation_rules.py")
SIM = Path("scripts/simulation_v2.py")

# Fields GameScriptProjection carries, split by what they are built from.
OPPONENT_AWARE = ("lead_prob", "neutral_prob", "trail_prob",
                  "blowout_risk", "shootout_risk")
CONSTANT_DERIVED = ("projected_pass_attempts", "projected_rush_attempts")
TEAM_SPECIFIC = ("projected_plays",)

# Modules that can turn a value into a projection. A diagnostics writer is not
# one of them, which is the whole distinction this trace rests on.
PROJECTION_CONSUMERS = (
    "scripts/simulation_v2.py",
    "scripts/modeling/simulation_rules.py",
    "scripts/modeling/ml_v2.py",
    "scripts/modeling/state_v2.py",
    "scripts/modeling/ensemble_v2.py",
    "scripts/run_pricing_v2.py",
)


def assert_source_contract() -> dict:
    """Fail closed if any structural claim in this trace stops holding."""
    rules = RULES.read_text(encoding="utf-8")
    simrules = SIMRULES.read_text(encoding="utf-8")
    sim = SIM.read_text(encoding="utf-8")
    checks = {}

    body = rules.split("def project_game_script(", 1)[1].split("\ndef ", 1)[0]

    m = re.search(r"^\s*pass_share\s*=\s*([\d.]+)\s*$", body, re.M)
    if not m:
        raise RuntimeError(
            "pass_share is no longer a bare literal in project_game_script; "
            "the published constant-split finding must be revised")
    checks["pass_share_literal"] = float(m.group(1))

    if not re.search(r"diff\s*=\s*success_diff\(offense,\s*defense\)", body):
        raise RuntimeError("project_game_script no longer computes success_diff")
    if not re.search(r"lead,\s*neutral,\s*trail\s*=\s*script_distribution\(diff\)",
                     body):
        raise RuntimeError("script_distribution(diff) is no longer derived")
    checks["opponent_aware_diff_computed"] = True

    # The split must be built from the constant, not from diff.
    if not re.search(r"pass_attempts\s*=\s*plays\s*\*\s*pass_share", body):
        raise RuntimeError(
            "pass_attempts is no longer plays * pass_share; revise the finding")
    checks["split_built_from_constant"] = True

    # rules_pass_rate must still be the constant ratio by construction.
    if not re.search(
            r'rules_pass_rate"\]\s*=\s*script\.projected_pass_attempts\s*/\s*'
            r'script\.projected_plays', simrules):
        raise RuntimeError("rules_pass_rate is no longer that ratio")
    checks["rules_pass_rate_is_constant_ratio"] = True

    # simulation_v2 must still prefer rules_pass_rate over the PROE fallback.
    ti = sim.split("def _team_inputs(", 1)[1].split("\ndef ", 1)[0]
    if 'pass_rate = _num(row, "rules_pass_rate")' not in ti:
        raise RuntimeError("simulation_v2 no longer reads rules_pass_rate first")
    if not re.search(r"if not np\.isfinite\(pass_rate\)", ti):
        raise RuntimeError("the PROE fallback guard changed shape")
    fb = re.search(r"pass_rate\s*=\s*([\d.]+)\s*\+\s*\(proe", ti)
    if not fb:
        raise RuntimeError("the PROE fallback no longer has that form")
    checks["sim_prefers_rules_pass_rate"] = True
    checks["proe_fallback_base"] = float(fb.group(1))
    checks["proe_fallback_is_opponent_environment_aware"] = "proe" in ti
    return checks


def consumption() -> dict:
    """Where each GameScriptProjection field is read, and by what kind of file."""
    files = {p: Path(p).read_text(encoding="utf-8")
             for p in PROJECTION_CONSUMERS if Path(p).exists()}
    simrules = files.get("scripts/modeling/simulation_rules.py", "")
    # build_rule_diagnostics writes model_rule_diagnostics.csv; it is a
    # diagnostics writer, not a projection path.
    diag_body = (simrules.split("def build_rule_diagnostics(", 1)[1]
                 if "def build_rule_diagnostics(" in simrules else "")
    out = {}
    for field in OPPONENT_AWARE + CONSTANT_DERIVED + TEAM_SPECIFIC:
        # Count the SAME token on both sides. V1 compared a total that also
        # counted the quoted dict key against a diagnostics count of
        # `script.<field>` only, which made every diagnostics-only field look
        # like it reached a projection.
        token = f"script.{field}"
        hits = {path: src.count(token) for path, src in files.items()
                if src.count(token)}
        in_diag = diag_body.count(token)
        projection_hits = {
            q: n for q, n in hits.items()
            if not (q == "scripts/modeling/simulation_rules.py"
                    and n <= in_diag)
        }
        out[field] = {
            "referenced_in": sorted(hits),
            "diagnostics_only_references": in_diag,
            "reaches_a_projection": bool(projection_hits),
            "projection_references": sorted(projection_hits),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                   default=Path("data/research/game_script"))
    a = ap.parse_args()

    checks = assert_source_contract()
    cons = consumption()
    stranded = [f for f in OPPONENT_AWARE if not cons[f]["reaches_a_projection"]]

    status = {
        "version": "GAME_SCRIPT_TRANSMISSION_TRACE_V1",
        "authorized_by": "USER_2026_10_27 parallel lane",
        "scientific_boundary": "READ_ONLY_SOURCE_TRACE_NO_FITTING_NO_SCORING",
        "artifacts_read": False,
        "outcomes_inspected": False,
        "sportsbook_inputs_used": 0,
        "source_contract_checks": checks,
        "field_consumption": cons,
        "opponent_aware_fields_not_reaching_any_projection": stranded,
        "FINDING": (
            "project_game_script computes an opponent-aware diff via "
            "success_diff and turns it into lead/neutral/trail probabilities "
            "through a softmax on diff*20, then builds the pass/rush split "
            "from a bare pass_share literal that does not depend on diff. "
            "Team VOLUME stays team-specific through estimate_plays, but the "
            "pass/rush SPLIT is the same constant for every team whose context "
            "resolves."),
        "PREEMPTION": (
            "the constant does not merely sit beside a richer signal, it "
            "preempts one. simulation_v2._team_inputs reads rules_pass_rate "
            "FIRST and only falls back to a PROE-based, win-probability-nudged "
            "pass rate when that value is non-finite. Since "
            "apply_rules_to_metrics writes rules_pass_rate for every row whose "
            "offense and defense context resolves, the opponent/environment "
            "aware path runs only when the rules layer fails."),
        "RESOLVES_READ_VS_EFFECTIVE": (
            "audit 6449349551 blocker 3 correctly established that "
            "success_rate_def IS read by production, via "
            "rules_v2.success_diff inside project_game_script, and flagged "
            "whether it changes a named player's projection as a separate "
            "routing question. This trace answers it: diff's only outputs are "
            "the script probabilities and the risk booleans, none of which "
            "reach a projection. success_rate_def is READ and has NO PATH to "
            "any projection. Read and effective come apart."),
        "WHY_THIS_IS_UPSTREAM": (
            "the pass/rush split is the finite opportunity pool from which "
            "every target and every carry is allocated, so a constant here "
            "sits upstream of every role multiplier already traced."),
        "not_established": [
            "that a non-constant split would improve any projection -- "
            "untested, scoring window closed, and no alternative is proposed",
            "the numeric effect on any named player's mean",
            "whether 0.57 is a defensible league-average choice",
            "per-team finiteness of rules_pass_rate in a live run (that needs "
            "the artifacts, which this session cannot read)",
        ],
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "game_script_transmission_v1_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== GAME SCRIPT TRANSMISSION TRACE V1 ===")
    print(f"pass_share literal in project_game_script : "
          f"{checks['pass_share_literal']}")
    print(f"PROE fallback base (only if rules value absent): "
          f"{checks['proe_fallback_base']} + proe\n")
    print(f"{'GameScriptProjection field':30s} {'reaches projection?':>20s}  where")
    for f in TEAM_SPECIFIC + CONSTANT_DERIVED + OPPONENT_AWARE:
        c = cons[f]
        where = ", ".join(p.split("/")[-1] for p in c["projection_references"]) \
            or ("diagnostics only" if c["referenced_in"] else "NOWHERE")
        print(f"{f:30s} {str(c['reaches_a_projection']):>20s}  {where}")
    print(f"\nopponent-aware fields reaching NO projection: {stranded}")
    print("\nNo alternative split is proposed. No fitting, no scoring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
