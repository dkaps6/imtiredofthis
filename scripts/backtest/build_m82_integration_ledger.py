#!/usr/bin/env python3
"""M82: freeze the M40-M81 QB research integration-eligibility ledger.

This is provenance/decision hygiene, not a predictive model. It prevents a dead
information family from being reopened merely by changing algorithms while also
preserving genuinely source-blocked frontiers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROWS = [
    ("canonical_role_opportunity_identity", "M47 and supporting M39+", "PROMOTED_FOUNDATION", "stable-primary/attempt-share identity and canonical opportunity decomposition are foundational", "none; maintain only if data contract changes"),
    ("57_43_pass_rush_opportunity", "M16-M21", "PROMOTED_FOUNDATION", "full walk-forward calibration selected stable pass opportunity architecture", "new independent evidence only"),
    ("gentle_pressure_rule", "M22-M23", "PROMOTED_FOUNDATION", "full canonical calibration retained pressure at materially reduced strength", "new independent pressure information, not coefficient retuning"),
    ("rushing_allocation_pool", "M24-M30", "PROMOTED_FOUNDATION", "full simulation traced and promoted player-specific/top-5 carry allocation", "not a QB frontier"),
    ("receiving_target_pool_and_wr_hierarchy", "M31-M38", "PROMOTED_FOUNDATION", "target-pool breadth and WR hierarchy were calibrated in integrated simulation", "new receiving information only"),
    ("team_pass_rate_and_gamescript_repackaging", "M40-M42 plus earlier M16-M21", "FULL_STACK_TESTED_CLOSED", "fixed architecture outperformed more dynamic pass-rate/game-state formulations", "materially new pregame intent observable"),
    ("market_context_gamescript_attempts_joint_ypa", "M50-M55", "FULL_STACK_TESTED_CLOSED", "some low-50s historical results were real but the old frontier used market variables and/or market-narrowed cohorts; not valid football-only foundation", "football-only rebuild with genuinely new information, not reuse of spreads/totals"),
    ("richer_static_defensive_matchup", "M56", "SIGNAL_SCREEN_FAILED", "lagged pressure/pass EPA/funnel/coverage/box and QB-defense interactions did not clear the precommitted incremental gate", "conditional defensive adaptation would be a different observable"),
    ("raw_range_decompression", "M57-M59", "FULL_STACK_TESTED_CLOSED", "range signal existed but unacceptable high-side catastrophic/collateral behavior prevented promotion", "new pregame selector that predicts when decompression is correct"),
    ("attempt_selective_trust_stack", "M61", "FULL_STACK_TESTED_CLOSED", "second-stage trust models were tested inside canonical simulation and failed frozen gates", "materially new trust information"),
    ("extreme_error_regime_classifier", "M62", "SIGNAL_SCREEN_FAILED", "pregame extreme-error classification did not produce deployable residual signal", "new information family"),
    ("directional_high_low_attempt_surprise", "M63", "SIGNAL_SCREEN_FAILED", "high-volume surprise remained poorly predictable; low-side partial structure was insufficient", "new week-specific intent/opportunity observable"),
    ("possession_dropback_generative_state", "M64-M65", "FULL_STACK_TESTED_CLOSED", "more detailed possession/dropback/state occupancy did not beat the simpler architecture prospectively", "new state-transition information"),
    ("existing_model_combination_and_diversity", "M66", "FULL_STACK_TESTED_CLOSED", "representative QB systems made highly correlated residual errors; static/learned combinations did not unlock a breakthrough", "new independent information, not another ensemble form"),
    ("offensive_intent_opening_playcaller", "M67-M69", "SIGNAL_SCREEN_FAILED", "opening/playcaller/tendency signals showed limited structure but no actionable canonical gain", "direct pregame gameplan information materially distinct from historical tendency"),
    ("qb_efficiency_volatility_uncertainty", "M70-M71", "SIGNAL_SCREEN_FAILED", "pregame efficiency/volatility families did not predict residual direction strongly enough", "new decision/process observable"),
    ("explosive_weapon_aggregate_matchup", "M72", "SIGNAL_SCREEN_FAILED", "aggregate explosive-weapon x defense matchup failed to replicate a QB residual bridge", "true player-level responsibility/route matchup would be materially new"),
    ("attempt_opportunity_oracle_recoverability", "M73", "FULL_STACK_TESTED_CLOSED", "diagnostic established large structural headroom but oracle information is not pregame deployable", "identify a new pregame predictor of opportunity surprise"),
    ("limited_attempt_dropback_context", "M74", "SIGNAL_SCREEN_FAILED", "focused context did not provide a robust new predictive bridge", "new information only"),
    ("ngs_receiver_tracking_pfr_secondary", "M75", "SIGNAL_SCREEN_FAILED", "exact player-level tracking/secondary sources qualified but Ridge/HGB and interactions worsened canonical-v3", "materially new player-level matchup information"),
    ("exact_personnel_discontinuity", "M76-M77", "SIGNAL_SCREEN_FAILED", "qualified exact-personnel source did not improve frozen 2024/2025 predictive test", "new mechanism beyond discontinuity counts"),
    ("official_inactives", "M78-M79", "SIGNAL_SCREEN_FAILED", "exact pregame inactives qualified but worsened passing yards, attempts and YPA", "new injury consequence information rather than availability identity"),
    ("ftn_tactical_pressure_decision_drop_families", "M80-M81", "SIGNAL_SCREEN_FAILED", "all four qualified FTN families worsened late-2024 MAE and both attempts/YPA component MAE under frozen development", "materially new FTN observable; no same-data model-zoo rescue"),
    ("route_x_coverage_shell", "M80", "SOURCE_BLOCKED_NEW_INFORMATION_REQUIRED", "historical interaction is scientifically feasible but a trustworthy deployable in-season route source was not established", "obtain complete historical + live pregame/deployable route/shell contract"),
    ("true_blocker_x_true_rusher_assignment", "M80", "SOURCE_BLOCKED_NEW_INFORMATION_REQUIRED", "genuinely new individual assignment information exists in limited public competition data but lacks complete historical/live contract", "obtain exact weekly blocker-rusher assignments historically and live"),
    ("top_weapon_escape_hatch", "concept overlaps M72/M75; exact hypothesis not directly tested", "SOURCE_BLOCKED_NEW_INFORMATION_REQUIRED", "realized WR and QB big games are mechanically linked; valid prediction requires a pregame best-single-weapon edge conditional on macro QB matchup. Existing aggregate/tracking proxies were negative", "materially new route/responsibility-level receiver-defender exposure, role-specific defensive replacement/injury, or equivalent"),
    ("defensive_adaptive_gameplan", "overlaps M56/M67-M69/M80 but exact conditional response not tested", "SOURCE_BLOCKED_NEW_INFORMATION_REQUIRED", "static defense and offensive intent were tested; defense-specific tactical adaptation to opponent archetype remains a distinct hypothesis only if predicted from prior games", "pregame-predictable conditional blitz/shell/man-zone/bracket/pressure response from strictly prior comparable-opponent games"),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["family", "migrations", "disposition", "evidence", "reopen_condition"]
    df = pd.DataFrame(ROWS, columns=cols)
    allowed = {
        "PROMOTED_FOUNDATION", "FULL_STACK_TESTED_CLOSED", "SIGNAL_SCREEN_FAILED",
        "PARTIAL_SIGNAL_INTEGRATION_CANDIDATE", "SOURCE_BLOCKED_NEW_INFORMATION_REQUIRED",
    }
    bad = sorted(set(df["disposition"]) - allowed)
    if bad:
        raise RuntimeError(f"invalid M82 ledger dispositions: {bad}")
    df.to_csv(args.out_dir / "m82_m40_m81_integration_ledger.csv", index=False)
    counts = df["disposition"].value_counts().to_dict()
    summary = {
        "migration": "M82", "rows": len(df), "disposition_counts": counts,
        "same_data_model_zoo_rescue_allowed": False,
        "note": "No family is labeled PARTIAL_SIGNAL_INTEGRATION_CANDIDATE unless prior evidence leaves a genuine architecture question open. Source-blocked rows require materially new information before predictive testing.",
    }
    (args.out_dir / "m82_integration_ledger_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(df.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
