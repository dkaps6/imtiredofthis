#!/usr/bin/env python3
"""Suffix-safe identity contract for the full-roster football simulation wrapper.

Full Slate providers legitimately disagree on suffixes (Jr/Sr/II/III/etc.).  The
football roster and pricing request set therefore use the same deterministic
Player Identity v3 base-name key that already protects the canonical metrics
join. Team remains part of every identity comparison, and same-team collisions
remain fatal. No football feature, simulation parameter, or sportsbook value is
changed here.
"""
from __future__ import annotations

import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v1 as base
from scripts.utils.player_identity_v3 import player_name_key


def _suffix_safe_key(value) -> str:
    try:
        return str(player_name_key(value, strip_suffix=True) or "").strip()
    except Exception:
        return ""


def _canonical_identity_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["team"] = out["team"].map(base.canon_team)
    if "player" not in out.columns:
        raise RuntimeError("full-roster identity frame requires player display name for suffix-safe identity")
    # Do not trust provider-specific player_clean_key representation here. Build
    # the identity key from the display name exactly as metrics_v2 does.
    out["player_clean_key"] = out["player"].map(_suffix_safe_key)
    blank = out["player_clean_key"].astype("string").fillna("").str.strip().eq("")
    if blank.any():
        sample = out.loc[blank, [c for c in ("player", "team") if c in out.columns]].head(20).to_dict("records")
        raise RuntimeError(f"suffix-safe full-roster identity key unresolved: {sample}")
    collisions = out.duplicated(["team", "player_clean_key"], keep=False)
    if collisions.any():
        sample = out.loc[collisions, ["player", "team", "player_clean_key"]].sort_values(
            ["team", "player_clean_key", "player"], kind="mergesort"
        ).head(20).to_dict("records")
        raise RuntimeError(f"suffix-safe full-roster identity is ambiguous within current team: {sample}")
    return out


def main() -> int:
    base._identity_frame = _canonical_identity_frame
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
