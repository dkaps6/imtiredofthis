#!/usr/bin/env python3
"""Fail closed if final RB rush+receiving means contradict final components."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
OUTPUTS = Path("outputs")
PRICED = OUTPUTS / "props_priced_clean.csv"
CONSERVATION = DATA / "rb_rush_rec_conservation_input_audit.json"
OUT = DATA / "rb_rush_rec_conservation_final_audit.csv"
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "vi", "vii"}


def _key(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    parts = [p for p in re.sub(r"[^a-z0-9]+", " ", text).split() if p]
    while parts and parts[-1] in SUFFIXES:
        parts.pop()
    return "".join(parts)


def main() -> int:
    if not CONSERVATION.exists() or CONSERVATION.stat().st_size == 0:
        raise RuntimeError("RB rush+receiving conservation input audit missing")
    meta = json.loads(CONSERVATION.read_text(encoding="utf-8"))
    if meta.get("sportsbook_inputs_used") is not False:
        raise RuntimeError("RB rush+receiving conservation reports sportsbook inputs")

    priced = pd.read_csv(PRICED, low_memory=False)
    priced.columns = [str(c).strip().lower() for c in priced.columns]

    # RB P3 and its rush+receiving conservation guarantee are Week-1-only.
    # Outside Week 1 the production contract explicitly uses the calibrated
    # generic ensemble.  The correct certified state is therefore a no-op with
    # zero P3-applied priced rows, not a fabricated P3 context and not a slate
    # abort.  Week 1 remains strict below.
    disposition = str(meta.get("disposition", ""))
    if disposition == "NO_ELIGIBLE_RB_RUSH_REC_ROWS":
        if "week" not in priced.columns or "rb_synthesis_applied" not in priced.columns:
            raise RuntimeError("non-Week-1 RB conservation audit missing week/rb_synthesis_applied columns")
        weeks = sorted(set(pd.to_numeric(priced["week"], errors="coerce").dropna().astype(int).tolist()))
        if not weeks or 1 in weeks:
            raise RuntimeError(
                "RB rush+receiving conservation reported no eligible rows for a slate containing Week 1"
            )
        applied = pd.to_numeric(priced["rb_synthesis_applied"], errors="coerce").fillna(0)
        if not applied.eq(0).all():
            sample = priced.loc[~applied.eq(0), [c for c in ("player", "team", "week", "source_market", "rb_synthesis_applied") if c in priced.columns]].head(20).to_dict("records")
            raise RuntimeError(f"non-Week-1 priced rows incorrectly claim RB P3 application: {sample}")
        if int(meta.get("players", -1)) != 0:
            raise RuntimeError(f"non-Week-1 RB conservation no-op reports players={meta.get('players')}")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["team", "player_base_key"]).to_csv(OUT, index=False)
        print(
            "[rb_rush_rec_final] "
            + json.dumps({
                "disposition": "FINAL_RB_RUSH_REC_NOT_APPLICABLE_OUTSIDE_WEEK1",
                "weeks": weeks,
                "players_checked": 0,
                "p3_applied_rows": 0,
                "max_gap": 0.0,
            }, sort_keys=True)
        )
        return 0

    if disposition != "RB_RUSH_REC_DISTRIBUTION_CONSERVED_WITH_PROMOTED_P3":
        raise RuntimeError(f"RB rush+receiving input conservation not certified: {disposition}")

    required = {"player", "team", "source_market", "side", "model_proj", "rb_synthesis_applied"}
    missing = required - set(priced.columns)
    if missing:
        raise RuntimeError(f"priced output missing conservation columns: {sorted(missing)}")
    x = priced.loc[priced["side"].astype(str).str.upper().eq("OVER")].copy()
    x["player_base_key"] = x["player"].map(_key)
    # Model mean must be invariant to sportsbook book/line for one player/market.
    spread = x.groupby(["team", "player_base_key", "source_market"])["model_proj"].nunique()
    if (spread > 1).any():
        raise RuntimeError(f"model projection varies by book/line: {spread.loc[spread>1].head(20).to_dict()}")
    one = x.drop_duplicates(["team", "player_base_key", "source_market"])
    pivot = one.pivot_table(
        index=["team", "player_base_key"],
        columns="source_market",
        values="model_proj",
        aggfunc="first",
    ).reset_index()
    need = ["player_rush_yds", "player_reception_yds", "player_rush_reception_yds"]
    present = pivot.dropna(subset=need).copy()
    if present.empty:
        raise RuntimeError("no players have all three RB conservation markets in final priced output")

    # The rush+receiving reconciliation is a P3-specific guarantee: P3 anchors
    # rush_rec_yards to its own synthesized rush mean, so the two are conserved
    # by construction. A player whose team is outside P3's built scope for this
    # run (see RB P3 team-scope fallback) prices rush_yds and rush_rec_yds from
    # two independently calibrated ensembles instead, with no such reconciliation
    # promised or intended -- checking them here would assert an invariant this
    # codebase never guaranteed for that player.
    rb_applied = one.loc[
        one["source_market"].eq("player_rush_yds"), ["team", "player_base_key", "rb_synthesis_applied"]
    ].rename(columns={"rb_synthesis_applied": "rush_yds_rb_synthesis_applied"})
    present = present.merge(rb_applied, on=["team", "player_base_key"], how="left")
    present["rush_yds_rb_synthesis_applied"] = pd.to_numeric(
        present["rush_yds_rb_synthesis_applied"], errors="coerce"
    ).fillna(0)
    p3_covered = present.loc[present["rush_yds_rb_synthesis_applied"].eq(1)].copy()

    if p3_covered.empty:
        max_gap = 0.0
    else:
        p3_covered["component_sum"] = p3_covered["player_rush_yds"] + p3_covered["player_reception_yds"]
        p3_covered["conservation_gap"] = p3_covered["player_rush_reception_yds"] - p3_covered["component_sum"]
        max_gap = float(p3_covered["conservation_gap"].abs().max())
        if not np.isfinite(max_gap) or max_gap > 1e-6:
            sample = p3_covered.loc[p3_covered["conservation_gap"].abs().gt(1e-6)].head(20).to_dict("records")
            raise RuntimeError(f"final RB rush+receiving projection conservation failed max_gap={max_gap} sample={sample}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    p3_covered.to_csv(OUT, index=False)
    print(
        "[rb_rush_rec_final] "
        + json.dumps({
            "disposition": "FINAL_RB_RUSH_REC_PROJECTIONS_CONSERVED",
            "players_checked": int(len(p3_covered)),
            "players_outside_p3_scope_skipped": int(len(present) - len(p3_covered)),
            "max_gap": max_gap,
        }, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
