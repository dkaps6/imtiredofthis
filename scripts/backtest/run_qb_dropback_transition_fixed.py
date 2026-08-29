#!/usr/bin/env python3
"""Authoritative M74 entrypoint.

The original M74 module declared `m64_pred_dropback_rate_gamescript` inside the
existing-state control. That legacy M64 diagnostic used a spread-implied game
state mixture, which violates the current football-only M74 feature boundary.
This entrypoint removes that field before executing the frozen audit.

No new M74 family, model, threshold, target, or gate is changed.
"""
from scripts.backtest import audit_qb_dropback_transition as m

MARKET_DERIVED_LEGACY_CONTROL = "m64_pred_dropback_rate_gamescript"

m.STATE_CONTROL = [
    c for c in m.STATE_CONTROL if c != MARKET_DERIVED_LEGACY_CONTROL
]
m.FAMILIES["m65_state_control"] = list(m.STATE_CONTROL)
m.FAMILIES["state_plus_new_transition"] = list(m.STATE_CONTROL) + list(m.COMBINED_NEW)

if MARKET_DERIVED_LEGACY_CONTROL in m.FAMILIES["m65_state_control"]:
    raise RuntimeError("M74 market-derived legacy control was not removed")
if MARKET_DERIVED_LEGACY_CONTROL in m.FAMILIES["state_plus_new_transition"]:
    raise RuntimeError("M74 market-derived legacy control leaked into combined family")

raise SystemExit(m.main())
