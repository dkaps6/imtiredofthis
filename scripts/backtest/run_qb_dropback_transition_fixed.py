#!/usr/bin/env python3
"""Authoritative football-only M74 entrypoint.

Integrity-only corrections to the frozen M74 audit:
1. The legacy `m64_pred_dropback_rate_gamescript` diagnostic is market-derived,
   so it is removed both from model families and from the exact canonical safe
   copier before the audit runs.
2. For state+new attribution, the supporting second model must show its own
   same-model incremental improvement over the M65 state control in addition to
   the frozen support gate. Old state signal cannot masquerade as replication of
   the new transition information.

No M74 new family, model, threshold, target, or frozen numerical gate changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import audit_qb_dropback_transition as m

MARKET_DERIVED_LEGACY_CONTROL = "m64_pred_dropback_rate_gamescript"

# Remove the market-derived diagnostic from every feature family.
m.STATE_CONTROL = [c for c in m.STATE_CONTROL if c != MARKET_DERIVED_LEGACY_CONTROL]
m.FAMILIES["m65_state_control"] = list(m.STATE_CONTROL)
m.FAMILIES["state_plus_new_transition"] = list(m.STATE_CONTROL) + list(m.COMBINED_NEW)

if MARKET_DERIVED_LEGACY_CONTROL in m.FAMILIES["m65_state_control"]:
    raise RuntimeError("M74 market-derived legacy control was not removed")
if MARKET_DERIVED_LEGACY_CONTROL in m.FAMILIES["state_plus_new_transition"]:
    raise RuntimeError("M74 market-derived legacy control leaked into combined family")

# Exact football-only safe canonical allow-list. This supersedes the original
# copier solely to make the prohibited market-derived diagnostic unreachable.
def add_safe_canonical_columns_football_only(rec, r):
    for c in [
        "opening_first15_dbr_mean8", "opening_q1_dbr_mean8",
        "playcaller_opening_first15_dbr_mean8", "playcaller_opening_q1_dbr_mean8",
        "playcaller_changed_since_last_game", "playcaller_prior_games_allteams",
        "playcaller_prior_games_team", "playcaller_new_to_team",
        "m64_pred_dropback_rate_neutral",
        "m65_pred_neutral_share", "m65_pred_trailing_share", "m65_pred_leading_share",
        "m65_pred_neutral_dropback_rate", "m65_pred_trailing_dropback_rate",
        "m65_pred_leading_dropback_rate", "m65_pred_dropback_rate",
    ]:
        rec[c] = float(r[c]) if c in r.index and pd.notna(r[c]) else np.nan


m.add_safe_canonical_columns = add_safe_canonical_columns_football_only


def corrected_interpretation(out_dir: Path):
    """Re-apply the frozen verdict with incremental support required on both models."""
    result_path = out_dir / "m74_model_results.csv"
    interp_path = out_dir / "m74_precommitted_interpretation.csv"
    results = pd.read_csv(result_path)
    interp = pd.read_csv(interp_path)

    supported_new = []
    for family in m.NEW_STANDALONE_FAMILIES:
        q = results[results.family.eq(family)]
        if len(q) != 2:
            continue
        for _, winner in q.iterrows():
            other = q[q.model.ne(winner.model)]
            if bool(winner.full_gate) and len(other) and bool(other.iloc[0].support_gate):
                supported_new.append(family)
                break

    combo_supported = False
    combo = results[results.family.eq("state_plus_new_transition")]
    if len(combo) == 2:
        for _, winner in combo.iterrows():
            other = combo[combo.model.ne(winner.model)]
            if (
                bool(winner.full_gate)
                and bool(winner.incremental_control_gate)
                and len(other)
                and bool(other.iloc[0].support_gate)
                and bool(other.iloc[0].incremental_control_gate)
            ):
                combo_supported = True
                break

    state = results[results.family.eq("m65_state_control")]
    state_supported = False
    if len(state) == 2:
        for _, winner in state.iterrows():
            other = state[state.model.ne(winner.model)]
            if bool(winner.full_gate) and len(other) and bool(other.iloc[0].support_gate):
                state_supported = True
                break

    if supported_new or combo_supported:
        verdict = "m74_dbr_transition_signal_followup"
    elif state_supported:
        verdict = "m74_existing_state_signal_only_no_new_transition_breakthrough"
    else:
        verdict = "m74_dbr_shift_not_predictable_with_current_opening_inducement_information"

    interp.loc[0, "supported_new_families"] = "|".join(sorted(set(supported_new)))
    interp.loc[0, "state_plus_new_incremental_supported"] = bool(combo_supported)
    interp.loc[0, "m65_state_control_supported"] = bool(state_supported)
    interp.loc[0, "m74_interpretation"] = verdict
    interp.loc[0, "production_actionable"] = False
    interp.to_csv(interp_path, index=False)


if __name__ == "__main__":
    rc = m.main()
    # argparse inside m.main has already resolved --out-dir. Parse it directly
    # from argv without changing the frozen audit's argument contract.
    import sys
    args = sys.argv[1:]
    out_dir = None
    for i, v in enumerate(args):
        if v == "--out-dir" and i + 1 < len(args):
            out_dir = Path(args[i + 1])
            break
    if out_dir is None:
        raise RuntimeError("M74 authoritative wrapper requires --out-dir")
    corrected_interpretation(out_dir)
    raise SystemExit(rc)
