#!/usr/bin/env python3
"""End-to-end semantic and consumption certification for a priced Full Slate.

A green process is insufficient. This gate reconciles the canonical compact odds
layer to exact bookmaker lines, then to priced side rows, validates model
component consumption, promoted QB/RB synthesis, quarantine preservation, and
carries forward unresolved provider/data-quality blockers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
OUTPUTS = Path("outputs")
PRICING_INPUT = OUTPUTS / "props_pricing_offers.csv"
PRICED = OUTPUTS / "props_priced_clean.csv"
METRICS = DATA / "metrics_ready.csv"
MODEL_CONTEXT = DATA / "model_context_bridge.csv"
OUT = OUTPUTS / "paid_full_slate_replay_result.json"
AUDIT_CSV = DATA / "full_slate_post_pricing_audit.csv"

ANYTIME = "player_anytime_td"
NUMERIC_MARKETS = {
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_rush_reception_yds",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"post-pricing required artifact missing/empty: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise RuntimeError(f"post-pricing required artifact has zero rows: {path}")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"post-pricing required JSON missing/empty: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _norm_line(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round(8)


def _position_family(value) -> str:
    if value is None or pd.isna(value):
        return ""
    pos = str(value).upper().strip()
    if pos in {"HB", "TB"} or pos.startswith("RB"):
        return "RB"
    if pos.startswith("FB"):
        return "FB"
    if pos.startswith("QB"):
        return "QB"
    if pos.startswith("WR") or pos in {"LWR", "RWR", "SWR"}:
        return "WR"
    if pos.startswith("TE"):
        return "TE"
    return pos


def _key_tuples(df: pd.DataFrame, *, output: bool = False) -> pd.Series:
    market_col = "source_market" if output else "market"
    line_col = "vegas_line" if output else "line"
    need = ["event_id", "player", market_col, "book", line_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"offer reconciliation missing columns: {missing}")
    return pd.Series(
        list(zip(
            df["event_id"].astype(str),
            df["player"].astype(str),
            df[market_col].astype(str),
            df["book"].astype(str),
            _norm_line(df[line_col]),
        )),
        index=df.index,
    )


def _attach_pricing_positions(priced: pd.DataFrame) -> pd.DataFrame:
    metrics = _read(METRICS)
    pos_col = next((c for c in ("position_group", "position", "alignment_position") if c in metrics.columns), None)
    if pos_col is None:
        raise RuntimeError("metrics_ready has no position column for final promoted-routing audit")
    need = {"player", "team", pos_col}
    if not need.issubset(metrics.columns):
        raise RuntimeError(f"metrics_ready missing position routing columns: {sorted(need-set(metrics.columns))}")
    positions = metrics[["player", "team", pos_col]].copy()
    positions["position_family_audit"] = positions[pos_col].map(_position_family)
    if positions["position_family_audit"].eq("").any():
        sample = positions.loc[positions["position_family_audit"].eq(""), ["player", "team", pos_col]].head(20).to_dict("records")
        raise RuntimeError(f"blank current position in pricing routing audit: {sample}")
    ambiguity = positions.groupby(["player", "team"])["position_family_audit"].nunique()
    bad = ambiguity.loc[ambiguity.ne(1)]
    if not bad.empty:
        raise RuntimeError(f"ambiguous current position at player/team grain: {bad.head(20).to_dict()}")
    positions = positions.drop_duplicates(["player", "team"])[["player", "team", "position_family_audit"]]
    out = priced.merge(positions, on=["player", "team"], how="left", validate="many_to_one")
    if out["position_family_audit"].isna().any():
        sample = out.loc[out["position_family_audit"].isna(), ["player", "team", "source_market"]].drop_duplicates().head(20).to_dict("records")
        raise RuntimeError(f"priced output missing current position mapping: {sample}")
    return out


def audit() -> dict:
    rows: list[dict] = []
    pricing_audit = _json(DATA / "live_pricing_offer_audit.json")
    if pricing_audit.get("disposition") != "PRICING_OFFERS_MATERIALIZED_EXACTLY_ONCE":
        raise RuntimeError(f"pricing offer adapter not certified: {pricing_audit.get('disposition')}")
    if bool(pricing_audit.get("consensus_line_created")):
        raise RuntimeError("pricing adapter created a consensus line; sportsbook provenance contract violated")

    inp = _read(PRICING_INPUT)
    priced = _read(PRICED)
    input_key = _key_tuples(inp, output=False)
    if input_key.duplicated().any():
        raise RuntimeError(f"pricing input has duplicate book-line keys={int(input_key.duplicated().sum())}")

    required_output = {
        "event_id", "player", "team", "opponent", "source_market", "book", "vegas_line",
        "side", "model_proj", "fair_prob", "ml_applied", "state_applied", "bayes_applied",
        "rules_applied", "ensemble_status",
    }
    missing = required_output - set(priced.columns)
    if missing:
        raise RuntimeError(f"priced output missing audit columns: {sorted(missing)}")
    out_key = _key_tuples(priced, output=True)
    priced = priced.assign(offer_key=out_key)
    inp = inp.assign(offer_key=input_key)

    side_counts = priced.groupby("offer_key")["side"].agg(list)
    bad_sides = side_counts.loc[side_counts.map(lambda values: sorted(map(str, values)) != ["OVER", "UNDER"])]
    if not bad_sides.empty:
        raise RuntimeError(f"priced offers do not contain exactly OVER+UNDER: {bad_sides.head(20).to_dict()}")
    if priced.duplicated(["offer_key", "side"]).any():
        raise RuntimeError("priced output contains duplicate offer/side rows")

    in_keys = set(inp["offer_key"])
    out_keys = set(priced["offer_key"])
    missing_keys = in_keys - out_keys
    extra_keys = out_keys - in_keys
    if missing_keys or extra_keys:
        raise RuntimeError(
            "pricing input/output offer keys do not reconcile; "
            f"missing={list(missing_keys)[:20]} extra={list(extra_keys)[:20]}"
        )
    if len(priced) != 2 * len(inp):
        raise RuntimeError(f"priced side-row count mismatch expected={2*len(inp)} actual={len(priced)}")

    odds_lookup: dict[tuple, tuple[float, float]] = {}
    for r in inp.itertuples(index=False):
        odds_lookup[getattr(r, "offer_key")] = (
            pd.to_numeric(pd.Series([getattr(r, "over_odds", np.nan)]), errors="coerce").iloc[0],
            pd.to_numeric(pd.Series([getattr(r, "under_odds", np.nan)]), errors="coerce").iloc[0],
        )
    for r in priced.itertuples(index=False):
        key = getattr(r, "offer_key")
        expected = odds_lookup[key][0 if str(r.side) == "OVER" else 1]
        actual = pd.to_numeric(pd.Series([getattr(r, "vegas_odds", np.nan)]), errors="coerce").iloc[0]
        market = str(getattr(r, "source_market"))
        if market == ANYTIME and str(r.side) == "UNDER" and pd.isna(expected) and pd.isna(actual):
            continue
        if pd.isna(expected) != pd.isna(actual) or (pd.notna(expected) and abs(float(expected) - float(actual)) > 1e-9):
            raise RuntimeError(
                f"sportsbook odds changed between adapter and pricing key={key} side={r.side} "
                f"expected={expected} actual={actual}"
            )

    for col in ("model_proj", "fair_prob"):
        vals = pd.to_numeric(priced[col], errors="coerce")
        if vals.isna().any() or not np.isfinite(vals).all():
            raise RuntimeError(f"priced output contains missing/non-finite {col}")
    fair = pd.to_numeric(priced["fair_prob"], errors="coerce")
    if not fair.between(0.0, 1.0, inclusive="both").all():
        raise RuntimeError("priced fair probability outside [0,1]")

    consumption: dict[str, dict[str, int]] = {}
    for market, part in priced.groupby("source_market"):
        consumption[str(market)] = {
            c: int(pd.to_numeric(part[c], errors="coerce").fillna(0).sum())
            for c in ("ml_applied", "state_applied", "bayes_applied", "rules_applied")
        }
    for market in sorted(NUMERIC_MARKETS & set(consumption)):
        for component in ("ml_applied", "state_applied", "bayes_applied", "rules_applied"):
            if consumption[market][component] <= 0:
                raise RuntimeError(f"{component} consumed zero rows for priced market={market}")

    pass_rows = priced.loc[priced["source_market"].astype(str).eq("player_pass_yds")].copy()
    if not pass_rows.empty:
        if "qb_synthesis_applied" not in pass_rows.columns or not pd.to_numeric(pass_rows["qb_synthesis_applied"], errors="coerce").eq(1).all():
            raise RuntimeError("not every pass-yards side row used promoted QB synthesis")
        qproj = pd.to_numeric(pass_rows["qb_synthesis_proj"], errors="coerce")
        final = pd.to_numeric(pass_rows["model_proj"], errors="coerce")
        if qproj.isna().any() or not np.allclose(qproj, final, rtol=0, atol=1e-8):
            raise RuntimeError("final pass-yards projection differs from promoted QB synthesis")

    # RB P3 is position-specific, not a generic rush-yards model. Require every
    # current RB/FB rush prop to use P3 and every QB/WR/TE rush prop not to use it.
    priced = _attach_pricing_positions(priced)
    rush_rows = priced.loc[priced["source_market"].astype(str).eq("player_rush_yds")].copy()
    rb_rush = rush_rows.loc[rush_rows["position_family_audit"].isin({"RB", "FB"})].copy()
    non_rb_rush = rush_rows.loc[~rush_rows["position_family_audit"].isin({"RB", "FB"})].copy()
    if not rb_rush.empty:
        applied = pd.to_numeric(rb_rush.get("rb_synthesis_applied", 0), errors="coerce").fillna(0)
        if not applied.eq(1).all():
            sample = rb_rush.loc[~applied.eq(1), ["player", "team", "position_family_audit"]].drop_duplicates().head(20).to_dict("records")
            raise RuntimeError(f"eligible RB/FB rush-yards rows did not all use promoted RB synthesis: {sample}")
        if not rb_rush["rb_synthesis_version"].astype(str).eq("RB_P3_SYNTHESIS_V1").all():
            raise RuntimeError("eligible RB/FB rush-yards pricing did not use RB_P3_SYNTHESIS_V1 everywhere")
        if not rb_rush["rb_synthesis_route"].astype(str).eq("WEEK1_STACK_OVERRIDE").all():
            raise RuntimeError("eligible RB/FB rush-yards pricing used a non-Week1 RB route")
        rb_proj = pd.to_numeric(rb_rush["rb_synthesis_proj"], errors="coerce")
        final = pd.to_numeric(rb_rush["model_proj"], errors="coerce")
        if rb_proj.isna().any() or not np.allclose(rb_proj, final, rtol=0, atol=1e-8):
            raise RuntimeError("final RB/FB rush-yards projection differs from promoted RB synthesis")
    if not non_rb_rush.empty:
        applied = pd.to_numeric(non_rb_rush.get("rb_synthesis_applied", 0), errors="coerce").fillna(0)
        if not applied.eq(0).all():
            sample = non_rb_rush.loc[~applied.eq(0), ["player", "team", "position_family_audit"]].drop_duplicates().head(20).to_dict("records")
            raise RuntimeError(f"non-RB rushing rows incorrectly consumed RB P3: {sample}")

    live = _json(DATA / "live_odds_status.json")
    quarantine = pd.read_csv(DATA / "live_odds_placeholder_rows.csv", low_memory=False)
    if len(quarantine) != int(live.get("production_quarantined_rows", -1)):
        raise RuntimeError("post-pricing quarantine evidence no longer matches live odds status")

    quality = _json(DATA / "full_slate_data_quality_audit.json")
    blockers = int(quality.get("certification_blockers", 0))
    quality_disposition = str(quality.get("disposition", "unknown"))

    # Runtime proof for the declared direct-WR/CB limitation. When the data
    # quality layer says the feature is unavailable/gated off, the model context
    # must contain zero eligible direct matchup flags; otherwise certification is fatal.
    coverage_components = {
        str(c.get("component")): str(c.get("status"))
        for c in quality.get("components", []) if isinstance(c, dict)
    }
    coverage_status = coverage_components.get("coverage_v2", "")
    direct_context_rows = -1
    if coverage_status == "DIRECT_MATCHUP_UNAVAILABLE_GATED_OFF":
        bridge = _read(MODEL_CONTEXT)
        direct_flags = pd.to_numeric(bridge.get("matchup_available", 0), errors="coerce").fillna(0)
        direct_context_rows = int(direct_flags.eq(1).sum())
        if direct_context_rows != 0 or not direct_flags.eq(0).all():
            raise RuntimeError(
                f"direct WR/CB feature declared unavailable but model context has eligible rows={direct_context_rows}"
            )

    rows.extend([
        {"check": "pricing_offer_reconciliation", "status": "PASS", "detail": f"book_line_rows={len(inp)} priced_side_rows={len(priced)}"},
        {"check": "sportsbook_value_preservation", "status": "PASS", "detail": "book/line/side odds exact"},
        {"check": "model_component_consumption", "status": "PASS", "detail": json.dumps(consumption, sort_keys=True)},
        {
            "check": "position_specific_synthesis_routing",
            "status": "PASS",
            "detail": f"qb_pass_side_rows={len(pass_rows)} rb_fb_rush_side_rows={len(rb_rush)} non_rb_rush_side_rows={len(non_rb_rush)}",
        },
        {
            "check": "direct_wr_cb_consumption_gate",
            "status": "PASS",
            "detail": f"coverage_status={coverage_status} direct_context_rows={direct_context_rows}",
        },
        {"check": "quarantine_preservation", "status": "PASS", "detail": f"rows={len(quarantine)}"},
        {"check": "data_quality_certification", "status": "PASS" if blockers == 0 else "BLOCKED", "detail": quality_disposition},
    ])
    pd.DataFrame(rows).to_csv(AUDIT_CSV, index=False)

    disposition = "PAID_FULL_SLATE_REPLAY_PRODUCTION_CERTIFIED" if blockers == 0 else "PAID_FULL_SLATE_REPLAY_EXECUTED_NOT_PRODUCTION_CERTIFIED"
    result = {
        "disposition": disposition,
        "source_run": 34152868136,
        "odds_api_refetched": False,
        "compact_rows": int(live.get("production_compact_rows", 0)),
        "quarantined_rows": int(len(quarantine)),
        "pricing_book_line_rows": int(len(inp)),
        "priced_side_rows": int(len(priced)),
        "priced_players": int(priced["player"].nunique()),
        "component_consumption_by_market": consumption,
        "qb_pass_side_rows": int(len(pass_rows)),
        "rb_fb_rush_side_rows": int(len(rb_rush)),
        "non_rb_rush_side_rows": int(len(non_rb_rush)),
        "direct_wr_cb_context_rows_consumed": int(max(direct_context_rows, 0)),
        "data_quality_disposition": quality_disposition,
        "certification_blockers": blockers,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[full_slate_post_pricing] " + json.dumps(result, sort_keys=True))
    if blockers:
        raise SystemExit(
            f"Full Slate executed end-to-end but is NOT production-certified; semantic blockers={blockers}. "
            f"See {DATA / 'full_slate_data_quality_audit.csv'}"
        )
    return result


def main() -> int:
    audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
