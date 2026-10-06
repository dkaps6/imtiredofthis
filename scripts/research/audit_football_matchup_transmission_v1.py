#!/usr/bin/env python3
"""Football Matchup Transmission V1 — Phase A deterministic production audit.

Reads an already-frozen Full Slate source artifact and current repository code.
No outcomes, sportsbook selection logic, fitting, or production mutation.

The goal is to distinguish:
- football context that exists,
- football context that reaches canonical contexts,
- football context that actually changes generic simulation,
- context used only by specialist paths,
- context that is present but effectively inert.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    ("plays_est", "team_volume", "USED_BY_GENERIC_SIMULATION"),
    ("neutral_pace", "team_volume", "USED_BY_GENERIC_SIMULATION"),
    ("true_proe", "team_pass_tendency", "QB_SPECIALIST_PRESENT_GENERIC_RULE_BYPASS"),
    ("proe", "team_pass_tendency", "GENERIC_SIM_FALLBACK_BYPASSED_BY_RULES_PASS_RATE"),
    ("neutral_pass_rate", "team_pass_tendency", "AVAILABLE_BUT_NOT_CONSUMED_GENERIC"),
    ("pass_rate_off", "team_pass_tendency", "QB_SPECIALIST_PRESENT_NOT_GENERIC_RULE"),
    ("pass_rate_faced", "defense_tendency", "QB_SPECIALIST_PRESENT_NOT_GENERIC_RULE"),
    ("success_rate_def", "defense_quality", "USED_TO_COMPUTE_SCRIPT_METADATA_ONLY"),
    ("def_rush_epa", "run_defense", "TEAM_CONTEXT_PRESENT_BUT_NOT_USED_BY_RULES"),
    ("explosive_play_rate_allowed", "defense_quality", "TEAM_CONTEXT_PRESENT_BUT_NOT_USED_BY_RULES"),
    ("light_box_rate", "run_defense", "USED_BY_GENERIC_THRESHOLD_RULE"),
    ("heavy_box_rate", "run_defense", "USED_BY_GENERIC_THRESHOLD_RULE"),
    ("yards_before_contact_per_rb_rush_x", "run_defense", "AVAILABLE_BUT_DROPPED_BEFORE_TEAM_CONTEXT"),
    ("yards_before_contact_per_rb_rush_y", "run_defense", "AVAILABLE_BUT_DROPPED_BEFORE_TEAM_CONTEXT"),
    ("rush_stuff_rate_x", "run_defense", "AVAILABLE_BUT_DROPPED_BEFORE_TEAM_CONTEXT"),
    ("rush_stuff_rate_y", "run_defense", "AVAILABLE_BUT_DROPPED_BEFORE_TEAM_CONTEXT"),
    ("ypt_allowed_wr", "position_receiving_defense", "AVAILABLE_BUT_DROPPED_BEFORE_TEAM_CONTEXT"),
    ("ypt_allowed_te", "position_receiving_defense", "AVAILABLE_BUT_DROPPED_BEFORE_TEAM_CONTEXT"),
    ("ypt_allowed_rb", "position_receiving_defense", "AVAILABLE_BUT_DROPPED_BEFORE_TEAM_CONTEXT"),
    ("ypt_allowed_outside", "alignment_receiving_defense", "AVAILABLE_BUT_DROPPED_BEFORE_TEAM_CONTEXT"),
    ("ypt_allowed_slot", "alignment_receiving_defense", "AVAILABLE_BUT_DROPPED_BEFORE_TEAM_CONTEXT"),
    ("coverage_man_rate", "coverage", "USED_BY_GENERIC_TARGET_RULE"),
    ("coverage_zone_rate", "coverage", "USED_BY_GENERIC_TARGET_RULE"),
    ("middle_open_rate", "coverage", "USED_BY_GENERIC_TARGET_RULE"),
    ("pressure_rate_generated", "pass_rush", "USED_BY_GENERIC_PRESSURE_RULE"),
    ("pressure_rate_allowed", "pass_protection", "USED_BY_GENERIC_PRESSURE_RULE"),
    ("def_pass_epa_allowed", "pass_defense", "QB_SPECIALIST_PRESENT_NOT_GENERIC_RULE"),
    ("def_pass_success_allowed", "pass_defense", "QB_SPECIALIST_PRESENT_NOT_GENERIC_RULE"),
    ("def_ypa_allowed", "pass_defense", "QB_SPECIALIST_PRESENT_NOT_GENERIC_RULE"),
]


def finite_stats(df: pd.DataFrame, col: str) -> dict:
    if col not in df.columns:
        return {"present": False, "finite_n": 0, "unique_n": 0, "min": None, "max": None}
    s = pd.to_numeric(df[col], errors="coerce")
    finite = s[np.isfinite(s)]
    return {
        "present": True,
        "finite_n": int(len(finite)),
        "unique_n": int(finite.nunique()),
        "min": float(finite.min()) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
    }


def code_has(path: Path, token: str) -> bool:
    if not path.exists():
        return False
    return token in path.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    root = args.artifact_root
    data = root / "data"
    tf_path = data / "team_form.csv"
    sim_path = data / "model_rule_simulation_inputs.csv"
    diag_path = data / "model_rule_diagnostics.csv"
    if not tf_path.exists() or not sim_path.exists() or not diag_path.exists():
        raise RuntimeError("source artifact missing required TeamForm/rule input/diagnostic files")

    tf = pd.read_csv(tf_path, low_memory=False)
    sim = pd.read_csv(sim_path, low_memory=False)
    diag = pd.read_csv(diag_path, low_memory=False)
    for d in (tf, sim, diag):
        d.columns = [str(c).strip().lower() for c in d.columns]

    if tf["team"].nunique() != 32:
        raise RuntimeError(f"TeamForm expected 32 teams; got {tf['team'].nunique()}")

    team_rates = sim[["team", "rules_pass_rate"]].drop_duplicates().copy()
    team_rates["rules_pass_rate"] = pd.to_numeric(team_rates["rules_pass_rate"], errors="coerce")
    if team_rates["team"].nunique() != 32:
        raise RuntimeError(f"simulation inputs expected 32 teams; got {team_rates['team'].nunique()}")
    fixed57 = bool(np.allclose(team_rates["rules_pass_rate"].to_numpy(float), 0.57, atol=1e-12, rtol=0))

    repo = Path(".")
    rules = repo / "scripts/modeling/rules_v2.py"
    sim_rules = repo / "scripts/modeling/simulation_rules.py"
    sim_v2 = repo / "scripts/simulation_v2.py"
    context = repo / "scripts/modeling/context_bridge.py"
    contracts = repo / "scripts/modeling/contracts.py"

    source_assertions = {
        "project_game_script_hardcodes_057": code_has(rules, "pass_share = 0.57"),
        "generic_sim_prefers_rules_pass_rate": code_has(sim_v2, 'pass_rate = _num(row, "rules_pass_rate")'),
        "generic_sim_proe_is_fallback": code_has(sim_v2, 'proe = _num(row, "proe", "pass_rate_over_expected", default=0.0)'),
        "def_rush_epa_in_team_context_contract": code_has(contracts, "def_rush_epa"),
        "def_rush_epa_not_in_matchup_rule_body": (
            code_has(rules, "def_rush_epa") is False
        ),
        "explosive_allowed_not_in_matchup_rule_body": (
            code_has(rules, "explosive_play_rate_allowed") is False
        ),
        "lead_prob_not_consumed_by_sim_v2": code_has(sim_v2, "lead_prob") is False,
        "trail_prob_not_consumed_by_sim_v2": code_has(sim_v2, "trail_prob") is False,
    }
    if not all(source_assertions.values()):
        raise RuntimeError(f"source architecture assertions changed: {source_assertions}")

    rows = []
    for feature, family, classification in FEATURES:
        stats = finite_stats(tf, feature)
        rows.append({
            "feature": feature,
            "family": family,
            "classification": classification,
            **stats,
        })
    inventory = pd.DataFrame(rows)

    # Exact pregame Bijan trace. No target-game outcome is read.
    bdiag = diag.loc[diag["player"].astype(str).str.contains("Bijan Robinson", case=False, na=False)]
    bsim = sim.loc[sim["player"].astype(str).str.contains("Bijan Robinson", case=False, na=False)]
    no = tf.loc[tf["team"].astype(str).str.upper().eq("NO")]
    atl = tf.loc[tf["team"].astype(str).str.upper().eq("ATL")]
    if len(bdiag) != 1 or bsim.empty or len(no) != 1 or len(atl) != 1:
        raise RuntimeError("unable to isolate exact Week-4 ATL/NO/Bijan pregame state")

    d = bdiag.iloc[0]
    s = bsim.iloc[0]
    no = no.iloc[0]
    atl = atl.iloc[0]

    plays = float(d["projected_plays"])
    fixed_rush_total = float(d["projected_rush_attempts"])
    playerform_share = float(pd.to_numeric(s.get("rush_share"), errors="coerce"))
    bayes_share = float(pd.to_numeric(s.get("bayes_rush_share"), errors="coerce"))
    ypc = float(pd.to_numeric(s.get("bayes_ypc"), errors="coerce"))
    neutral_pass = float(pd.to_numeric(atl.get("neutral_pass_rate"), errors="coerce")) / 100.0
    if neutral_pass > 1:
        raise RuntimeError("neutral_pass_rate normalization unexpected")
    alt_team_rush = plays * (1.0 - neutral_pass)
    alt_carries = alt_team_rush * playerform_share
    alt_yards_same_ypc = alt_carries * ypc

    bijan = {
        "season": 2026,
        "week": 4,
        "player": "Bijan Robinson",
        "team": "ATL",
        "opponent": "NO",
        "atl_true_proe": float(pd.to_numeric(atl.get("true_proe"), errors="coerce")),
        "atl_neutral_pass_rate": neutral_pass,
        "rules_pass_rate": float(d["projected_pass_attempts"] / d["projected_plays"]),
        "projected_plays": plays,
        "rules_team_rush_attempts": fixed_rush_total,
        "playerform_rush_share": playerform_share,
        "bayes_rules_rush_share": bayes_share,
        "bayes_ypc": ypc,
        "no_light_box_rate": float(pd.to_numeric(no.get("light_box_rate"), errors="coerce")),
        "no_heavy_box_rate": float(pd.to_numeric(no.get("heavy_box_rate"), errors="coerce")),
        "rules_rush_eff_mult": float(d["rush_eff_mult"]),
        "mechanical_alt_team_rushes_using_existing_neutral_pass_rate": alt_team_rush,
        "mechanical_alt_carries_using_existing_playerform_share": alt_carries,
        "mechanical_alt_rush_yards_same_bayes_ypc": alt_yards_same_ypc,
        "note": "architecture illustration only; not a candidate and no target-game outcome used",
    }

    payload = {
        "version": "FOOTBALL_MATCHUP_TRANSMISSION_V1_PHASE_A",
        "status": "DETERMINISTIC_TRANSMISSION_AUDIT_COMPLETE",
        "outcomes_used": 0,
        "sportsbook_refetch_used": False,
        "team_rules_pass_rate_fixed_057": fixed57,
        "team_rules_pass_rate_unique_values": sorted({round(float(v), 12) for v in team_rates["rules_pass_rate"].dropna()}),
        "source_assertions": source_assertions,
        "feature_classification_counts": inventory["classification"].value_counts().to_dict(),
        "bijan_pregame_trace": bijan,
        "production_change_authorized": False,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(args.out_dir / "phase_a_feature_transmission_inventory.csv", index=False)
    pd.DataFrame([bijan]).to_csv(args.out_dir / "phase_a_bijan_pregame_trace.csv", index=False)
    (args.out_dir / "phase_a_result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(inventory.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
