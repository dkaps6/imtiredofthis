#!/usr/bin/env python3
"""Identity-key hotfix for the full-roster football simulation wrapper.

V1 compared PlayerForm canonical keys with model-context display names.  This
wrapper installs the same canonical name-key contract before invoking V1; it does
not change any football feature, simulation parameter, or sportsbook input.
"""
from __future__ import annotations

import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v1 as base
from scripts.utils.canonical_names import canonicalize_player_name_safe


def _name_key(value) -> str:
    try:
        _, key = canonicalize_player_name_safe(value)
        if key:
            return str(key).strip()
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _canonical_identity_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["team"] = out["team"].map(base.canon_team)
    if "player_clean_key" in out.columns:
        # Existing PlayerForm/metrics keys have already passed the production
        # identity contract; preserve them, only normalizing representation.
        out["player_clean_key"] = (
            out["player_clean_key"].astype("string").fillna("").str.strip()
            .map(lambda v: "".join(ch.lower() for ch in str(v) if ch.isalnum()))
        )
    else:
        if "player" not in out.columns:
            raise RuntimeError("identity frame has neither player_clean_key nor player")
        out["player_clean_key"] = out["player"].map(_name_key)
    if out["player_clean_key"].eq("").any():
        sample = out.loc[out["player_clean_key"].eq(""), [c for c in ("player", "team") if c in out.columns]].head(20).to_dict("records")
        raise RuntimeError(f"canonical full-roster identity key unresolved: {sample}")
    return out


def main() -> int:
    base._identity_frame = _canonical_identity_frame
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
