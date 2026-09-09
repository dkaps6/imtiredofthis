"""Production adapter from frozen RB P3 football projections into pricing.

This module is deliberately sportsbook-blind. It only loads the precomputed
football-only RB synthesis context and resolves the authoritative rushing-yard
mean for a player/week. Sportsbook line/odds information remains downstream
inside ``run_pricing_v2.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.utils.player_identity_v3 import player_name_key

RB_CONTEXT_PATH = Path("data/rb_rush_synthesis_context.csv")
WEEK1_ROUTE = "WEEK1_STACK_OVERRIDE"
RB_VERSION = "RB_P3_SYNTHESIS_V1"


def _base_key(value) -> str:
    """Use the same suffix-insensitive person-name contract as Player Identity v3."""
    return player_name_key(value, strip_suffix=True)


def load_rb_context(path: Path = RB_CONTEXT_PATH) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"promoted RB synthesis context missing/empty: {path}")
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    required = {
        "season", "week", "player", "team", "opponent", "rb_synthesis_proj",
        "rb_synthesis_route", "rb_synthesis_version", "rb_synthesis_applied",
        "football_only_no_odds", "sportsbook_inputs_used",
    }
    missing = sorted(required - set(out.columns))
    if missing:
        raise RuntimeError(f"promoted RB synthesis context missing columns: {missing}")
    if out.empty:
        raise RuntimeError("promoted RB synthesis context has zero rows")

    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["rb_synthesis_proj"] = pd.to_numeric(out["rb_synthesis_proj"], errors="coerce")
    if out[["season", "week", "rb_synthesis_proj"]].isna().any().any():
        raise RuntimeError("promoted RB synthesis context contains non-numeric contract fields")
    if not pd.to_numeric(out["rb_synthesis_applied"], errors="coerce").eq(1).all():
        raise RuntimeError("promoted RB synthesis context contains unapplied rows")
    if not pd.to_numeric(out["football_only_no_odds"], errors="coerce").eq(1).all():
        raise RuntimeError("promoted RB synthesis context violated football-only contract")
    if not pd.to_numeric(out["sportsbook_inputs_used"], errors="coerce").eq(0).all():
        raise RuntimeError("promoted RB synthesis context reports sportsbook leakage")
    if (out["rb_synthesis_proj"] < 0).any():
        raise RuntimeError("promoted RB synthesis context contains negative projection")

    out["team"] = out["team"].fillna("").astype(str).str.upper().str.strip()
    out["opponent"] = out["opponent"].fillna("").astype(str).str.upper().str.strip()
    out["player_base_key"] = out["player"].map(_base_key)
    if out["player_base_key"].astype(str).str.len().eq(0).any():
        raise RuntimeError("promoted RB synthesis context contains blank normalized player identity")

    # The currently promoted live production contract is Week 1. Do not allow
    # a future week to silently use the Week-1 parent while W2-18 source safety
    # remains unresolved.
    wk1 = out["week"].eq(1)
    if wk1.any():
        if not out.loc[wk1, "rb_synthesis_route"].astype(str).eq(WEEK1_ROUTE).all():
            raise RuntimeError("Week-1 RB context contains non-P3 Week-1 route")
        if not out.loc[wk1, "rb_synthesis_version"].astype(str).eq(RB_VERSION).all():
            raise RuntimeError("Week-1 RB context version drift")

    dup = out.duplicated(["season", "week", "team", "player_base_key"], keep=False)
    if dup.any():
        sample = out.loc[dup, ["season", "week", "team", "player", "player_base_key"]].head(10)
        raise RuntimeError(f"duplicate promoted RB context identities after suffix normalization:\n{sample.to_string(index=False)}")
    return out


def lookup_rb_projection(row: pd.Series, context: pd.DataFrame) -> dict[str, object]:
    season = int(float(row.get("season")))
    week = int(float(row.get("week")))
    team = str(row.get("team") or "").upper().strip()
    opponent = str(row.get("opponent") or "").upper().strip()
    player_key = _base_key(row.get("player"))
    if not player_key:
        raise RuntimeError(f"promoted RB synthesis pricing row has blank player identity: {row.get('player')!r}")

    q = context.loc[
        context["season"].eq(season)
        & context["week"].eq(week)
        & context["team"].eq(team)
        & context["player_base_key"].eq(player_key)
    ].copy()
    if len(q) != 1:
        raise RuntimeError(
            f"promoted RB synthesis identity mismatch player={row.get('player')} "
            f"team={team} season={season} week={week} matches={len(q)}"
        )
    rec = q.iloc[0]
    ctx_opp = str(rec.get("opponent") or "").upper().strip()
    if opponent and ctx_opp and opponent != ctx_opp:
        raise RuntimeError(
            f"promoted RB synthesis opponent mismatch player={row.get('player')} "
            f"pricing={opponent} context={ctx_opp}"
        )
    proj = float(rec["rb_synthesis_proj"])
    if not np.isfinite(proj) or proj < 0:
        raise RuntimeError("promoted RB synthesis produced invalid pricing mean")
    return {
        "rb_synthesis_proj": proj,
        "rb_synthesis_route": str(rec["rb_synthesis_route"]),
        "rb_synthesis_version": str(rec["rb_synthesis_version"]),
        "rb_synthesis_applied": 1,
        "rb_stack_implied_ypc": rec.get("rb_stack_implied_ypc", np.nan),
        "rb_ypc_fallback_used": rec.get("rb_ypc_fallback_used", 0),
    }
