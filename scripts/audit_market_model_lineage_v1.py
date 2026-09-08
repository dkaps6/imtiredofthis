#!/usr/bin/env python3
"""Materialize the actual model lineage consumed by the current priced Full Slate.

This is a governance/audit artifact, not a projection model.  Its purpose is to
make the distinction between promoted specialist models, generic canonical
components, and research that exists but is not yet consumed impossible to hide
behind a single 'production' label.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA = Path("data")
OUTPUTS = Path("outputs")
PRICED = OUTPUTS / "props_priced_clean.csv"
OUT_CSV = DATA / "market_model_lineage_current.csv"
OUT_JSON = DATA / "market_model_lineage_current.json"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"lineage required artifact missing/empty: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if df.empty:
        raise RuntimeError(f"lineage required artifact has zero rows: {path}")
    return df


def _pos(value) -> str:
    p = str(value or "").upper().strip()
    if p in {"RB", "HB", "TB", "FB"}: return "RB/FB"
    if p in {"WR", "LWR", "RWR", "SWR"}: return "WR"
    if p == "TE": return "TE"
    if p == "QB": return "QB"
    return p or "OTHER"


def main() -> int:
    priced = _read(PRICED)
    metrics = _read(DATA / "metrics_ready.csv")
    pos_col = next((c for c in ("position_group", "position", "alignment_position") if c in metrics.columns), None)
    if pos_col is None:
        raise RuntimeError("metrics_ready has no position column for lineage audit")
    positions = metrics[["player", "team", pos_col]].drop_duplicates(["player", "team"]).copy()
    positions["position_family"] = positions[pos_col].map(_pos)
    p = priced.merge(positions[["player", "team", "position_family"]], on=["player", "team"], how="left", validate="many_to_one")
    if p["position_family"].isna().any():
        raise RuntimeError("lineage audit could not attach position to every priced player")

    rows: list[dict] = []
    def add(market, position, owner, specialist, science, active_research, limitation):
        part = p.loc[p["source_market"].astype(str).eq(market)]
        if position != "ALL":
            part = part.loc[part["position_family"].eq(position)]
        rows.append({
            "market": market,
            "position_family": position,
            "priced_side_rows": int(len(part)),
            "final_mean_owner": owner,
            "specialist_model_active": int(bool(specialist)),
            "scientific_status": science,
            "active_or_next_research": active_research,
            "known_limitation": limitation,
        })

    add(
        "player_pass_yds", "QB",
        "QB_PASS_SYNTHESIS_V1 / M89-M90 over canonical joint MC components",
        True, "PROMOTED_SPECIALIST_ACTIVE",
        "C2 shared pass/receiving conservation supported but full-stack integration still pending",
        "QB mean is specialist-promoted; joint receiver distribution is not yet C2-integrated",
    )
    add(
        "player_rush_yds", "RB/FB",
        "RB_P3_SYNTHESIS_V1 / WEEK1_STACK_OVERRIDE",
        True, "PROMOTED_SPECIALIST_ACTIVE",
        "multiseason shared-room RB entitlement remains research",
        "Week-1 route only; room entitlement beyond P3 still pending",
    )
    for position in ("QB", "WR", "TE"):
        add(
            "player_rush_yds", position,
            "canonical calibrated rush-yards ensemble + joint MC",
            False, "GENERIC_CANONICAL_ACTIVE",
            "position-specific rushing specialist not promoted for this family",
            "RB P3 is correctly gated off for non-RB/FB rushing",
        )
    for market in ("player_reception_yds", "player_receptions"):
        add(
            market, "WR",
            "canonical Bayes/ML/State/rules/joint MC with WR M38 relative target hierarchy",
            True, "PARTIAL_SPECIALIST_ACTIVE_MODEL_QUALITY_BLOCKED",
            "finite current-team WR room pool -> residual entitlement around M38",
            "current team target pool is over-entitled before simulation; receiving ensemble is not calibrated",
        )
        add(
            market, "TE",
            "canonical Bayes/ML/State/rules/joint MC",
            False, "GENERIC_CANONICAL_ACTIVE_TE_R5P_NOT_CONSUMED",
            "TE-R5/TE-R5P passed research gates and awaits full-stack integration after finite pool layer",
            "TE specialist entitlement is not yet wired into Full Slate",
        )
        add(
            market, "RB/FB",
            "canonical Bayes/ML/State/rules/joint MC",
            False, "GENERIC_CANONICAL_ACTIVE",
            "finite RB receiving-room entitlement",
            "current team target pool is over-entitled before simulation",
        )
    add(
        "player_rush_reception_yds", "RB/FB",
        "RB P3-conserved rushing component + canonical receiving joint MC",
        True, "PARTIAL_SPECIALIST_ACTIVE_MODEL_QUALITY_BLOCKED",
        "finite RB receiving-room entitlement; dedicated joint-market calibration later",
        "rushing component is P3-consistent but receiving component still inherits target-pool blocker",
    )
    add(
        "player_anytime_td", "ALL",
        "generic joint MC offensive_td_rate + red-zone/script modifiers",
        False, "ATD_GENERIC_ACTIVE_NOT_DEDICATED_SCIENCE_CERTIFIED",
        "dedicated football-only anytime-TD opportunity/entitlement calibration",
        "old sportsbook-assisted TD scorer is retired; current ATD lane has no dedicated walk-forward certification",
    )

    out = pd.DataFrame(rows)
    observed = set(p["source_market"].astype(str).unique())
    represented = set(out.loc[out["priced_side_rows"].gt(0), "market"].astype(str))
    missing = sorted(observed - represented)
    if missing:
        raise RuntimeError(f"priced markets absent from lineage registry: {missing}")

    qb = p.loc[p["source_market"].astype(str).eq("player_pass_yds")]
    if not qb.empty and not pd.to_numeric(qb.get("qb_synthesis_applied", 0), errors="coerce").fillna(0).eq(1).all():
        raise RuntimeError("lineage claims QB specialist active but priced rows disagree")
    rb = p.loc[p["source_market"].astype(str).eq("player_rush_yds") & p["position_family"].eq("RB/FB")]
    if not rb.empty and not pd.to_numeric(rb.get("rb_synthesis_applied", 0), errors="coerce").fillna(0).eq(1).all():
        raise RuntimeError("lineage claims RB P3 active but priced rows disagree")
    atd = p.loc[p["source_market"].astype(str).eq("player_anytime_td")]
    if not atd.empty:
        if int(pd.to_numeric(atd.get("ml_applied", 0), errors="coerce").fillna(0).sum()) != 0:
            raise RuntimeError("ATD lineage expected no dedicated ML consumption but priced rows disagree")
        if int(pd.to_numeric(atd.get("state_applied", 0), errors="coerce").fillna(0).sum()) != 0:
            raise RuntimeError("ATD lineage expected no State consumption but priced rows disagree")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    payload = {
        "disposition": "MARKET_MODEL_LINEAGE_EXPLICIT",
        "priced_markets": sorted(observed),
        "lineage_rows": int(len(out)),
        "old_alternate_engine_active": False,
        "all_specialist_research_consumed": False,
        "te_r5p_consumed": False,
        "c2_full_stack_consumed": False,
        "anytime_td_dedicated_science_certified": False,
        "sportsbook_inputs_used_to_define_lineage": False,
        "audit": str(OUT_CSV),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[market_model_lineage] " + json.dumps(payload, sort_keys=True))
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
