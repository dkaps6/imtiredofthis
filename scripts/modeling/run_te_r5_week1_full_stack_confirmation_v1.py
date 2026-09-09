#!/usr/bin/env python3
"""Mechanical launcher for the frozen TE-R5 confirmation.

The current Full Slate PlayerForm join carries a second `position` field, so a
roster-seeded frame can emerge with `position_x` / `position_y`.  Production
Ourlads roster identity is authoritative for the current slate.  This launcher
normalizes that join artifact before the frozen confirmation consumes the frame.
No model feature, coefficient, cap, target-mass equation, or science gate changes.
"""
from __future__ import annotations

import pandas as pd

from scripts.modeling import te_r5_week1_full_stack_confirmation_v1 as frozen


def _family(value: object) -> str:
    s = str(value or "").upper().strip()
    if s in {"LWR", "RWR", "SWR", "WR"}:
        return "WR"
    if s in {"HB", "TB", "RB"}:
        return "RB"
    return s


_original_build = frozen.build_live_metrics


def _build_live_metrics_position_safe(root):
    out = _original_build(root)
    if "position" in out.columns:
        return out
    if "position_x" not in out.columns:
        raise RuntimeError("Full Slate frame has no authoritative roster position column")

    roster_pos = out["position_x"].fillna("").astype(str).str.upper().str.strip()
    form_pos = out["position_y"].fillna("").astype(str).str.upper().str.strip() if "position_y" in out.columns else pd.Series("", index=out.index)
    conflicts = [
        i for i in out.index
        if form_pos.loc[i] and _family(roster_pos.loc[i]) != _family(form_pos.loc[i])
    ]
    if conflicts:
        sample = out.loc[conflicts[:20], [c for c in ["team", "player", "player_clean_key", "position_x", "position_y"] if c in out.columns]]
        raise RuntimeError(f"roster/PlayerForm position-family conflict rows={len(conflicts)} sample={sample.to_dict('records')}")

    out["player_form_position"] = form_pos
    out["position"] = roster_pos
    return out


frozen.build_live_metrics = _build_live_metrics_position_safe

if __name__ == "__main__":
    raise SystemExit(frozen.main())
