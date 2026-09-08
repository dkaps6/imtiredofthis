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


def _key_tuples(df: pd.DataFrame, *, output: bool = False) -> pd.Series:
    market_col = "source_market" if output else "market"
    line_col = "vegas_line" if output else "line"
    need = ["event_id", "player", market_col, "book", line_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"offer reconciliation missing columns: {missing}")
    return pd.Series(
        list(
            zip(
                df["event_id"].astype(str),
                df["player"].astype(str),
                df[market_col].astype(str),
                df["book"].astype(str),
                _norm_line(df[line_col]),
            )
        ),
        index=df.index,
    )


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
    bad_sides = side_counts.loc[
        side_counts.map(lambda values: sorted(map(str, values)) != ["OVER", "UNDER"])
    ]
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
        raise RuntimeError(
            f"priced side-row count mismatch expected={2*len(inp)} actual={len(priced)}"
        )

    odds_lookup: dict[tuple, tuple[float, float]] = {}
    for r in inp.itertuples(index=False):
        key = getattr(r, "offer_key")
        odds_lookup[key] = (
            pd.to_numeric(pd.Series([getattr(r, "over_odds", np.nan)]), errors="coerce").iloc[0],
            pd.to_numeric(pd.Series([getattr(r, "under_odds", np.nan)]), errors="coerce").iloc[0],
        )
    for r in priced.itertuples(index=False):
        key = getattr(r, "offer_key")
        expected = odds_lookup[key][0 if str(r.side) == "OVER" else 1]
        actual = pd.to_numeric(
            pd.Series([getattr(r, "vegas_odds", np.nan)]), errors="coerce"
        ).iloc[0]
        market = str(getattr(r, "source_market"))
        if market == ANYTIME and str(r.side) == "UNDER" and pd.isna(expected) and pd.isna(actual):
            continue
        if pd.isna(expected) != pd.isna(actual) or (
            pd.notna(expected) and abs(float(expected) - float(actual)) > 1e-9
        ):
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

    pass_rows = priced.loc[
        priced["source_market"].astype(str).eq("player_pass_yds")
    ].copy()
    if not pass_rows.empty:
        if (
            "qb_synthesis_applied" not in pass_rows.columns
            or not pd.to_numeric(pass_rows["qb_synthesis_applied"], errors="coerce").eq(1).all()
        ):
            raise RuntimeError("not every pass-yards side row used promoted QB synthesis")
        qproj = pd.to_numeric(pass_rows["qb_synthesis_proj"], errors="coerce")
        final = pd.to_numeric(pass_rows["model_proj"], errors="coerce")
        if qproj.isna().any() or not np.allclose(qproj, final, rtol=0, atol=1e-8):
            raise RuntimeError("final pass-yards projection differs from promoted QB synthesis")

    rush_rows = priced.loc[
        priced["source_market"].astype(str).eq("player_rush_yds")
    ].copy()
    if not rush_rows.empty:
        if (
            "rb_synthesis_applied" not in rush_rows.columns
            or not pd.to_numeric(rush_rows["rb_synthesis_applied"], errors="coerce").eq(1).all()
        ):
            raise RuntimeError("not every rush-yards side row used promoted RB synthesis")
        if not rush_rows["rb_synthesis_version"].astype(str).eq("RB_P3_SYNTHESIS_V1").all():
            raise RuntimeError("rush-yards pricing did not use RB_P3_SYNTHESIS_V1 everywhere")
        if not rush_rows["rb_synthesis_route"].astype(str).eq("WEEK1_STACK_OVERRIDE").all():
            raise RuntimeError("rush-yards pricing used a non-Week1 RB route")

    live = _json(DATA / "live_odds_status.json")
    quarantine = pd.read_csv(DATA / "live_odds_placeholder_rows.csv", low_memory=False)
    if len(quarantine) != int(live.get("production_quarantined_rows", -1)):
        raise RuntimeError("post-pricing quarantine evidence no longer matches live odds status")

    quality = _json(DATA / "full_slate_data_quality_audit.json")
    blockers = int(quality.get("certification_blockers", 0))
    quality_disposition = str(quality.get("disposition", "unknown"))

    rows.extend(
        [
            {
                "check": "pricing_offer_reconciliation",
                "status": "PASS",
                "detail": f"book_line_rows={len(inp)} priced_side_rows={len(priced)}",
            },
            {
                "check": "sportsbook_value_preservation",
                "status": "PASS",
                "detail": "book/line/side odds exact",
            },
            {
                "check": "model_component_consumption",
                "status": "PASS",
                "detail": json.dumps(consumption, sort_keys=True),
            },
            {
                "check": "quarantine_preservation",
                "status": "PASS",
                "detail": f"rows={len(quarantine)}",
            },
            {
                "check": "data_quality_certification",
                "status": "PASS" if blockers == 0 else "BLOCKED",
                "detail": quality_disposition,
            },
        ]
    )
    pd.DataFrame(rows).to_csv(AUDIT_CSV, index=False)

    disposition = (
        "PAID_FULL_SLATE_REPLAY_PRODUCTION_CERTIFIED"
        if blockers == 0
        else "PAID_FULL_SLATE_REPLAY_EXECUTED_NOT_PRODUCTION_CERTIFIED"
    )
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
