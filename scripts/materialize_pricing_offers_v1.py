#!/usr/bin/env python3
"""Expand canonical compact live props into one real book/market/line pricing row.

The canonical compact layer keeps provider provenance and ``offers_json`` without
bookmaker cartesian expansion. Deterministic metrics/pricing, however, operate at
one bookmaker line per row. This adapter bridges those contracts without choosing
a consensus line, changing prices, or reintroducing duplicated offers.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
OUTPUTS = Path("outputs")
STATUS = DATA / "live_odds_status.json"
COMPACT = OUTPUTS / "props_raw_compact.csv"
LEGACY_COMPACT = OUTPUTS / "props_raw.csv"
PRICING = OUTPUTS / "props_pricing_offers.csv"
METRICS_INPUT = OUTPUTS / "props_raw.csv"
AUDIT = DATA / "live_pricing_offer_audit.json"

ANYTIME_MARKET = "player_anytime_td"
SUPPORTED_MARKETS = {
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_rush_reception_yds",
    ANYTIME_MARKET,
}


def _text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _finite_number(value, *, label: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise RuntimeError(f"pricing offer {label} is non-numeric: {value!r}") from exc
    if not np.isfinite(out):
        raise RuntimeError(f"pricing offer {label} is non-finite: {value!r}")
    return float(out)


def _read_compact() -> pd.DataFrame:
    path = COMPACT if COMPACT.exists() and COMPACT.stat().st_size > 0 else LEGACY_COMPACT
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError("canonical compact live props missing before pricing offer expansion")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise RuntimeError("canonical compact live props have zero rows before pricing offer expansion")
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"event_id", "market", "player", "team_abbr", "opponent_abbr", "offers_json"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"canonical compact live props missing pricing columns: {sorted(missing)}")
    for col in ("event_id", "market", "player", "team_abbr", "opponent_abbr", "offers_json"):
        if df[col].astype("string").fillna("").str.strip().eq("").any():
            raise RuntimeError(f"canonical compact live props contain blank {col}")
    unsupported = ~df["market"].astype(str).isin(SUPPORTED_MARKETS)
    if unsupported.any():
        raise RuntimeError(
            "canonical compact live props contain unsupported pricing markets: "
            f"{sorted(df.loc[unsupported, 'market'].astype(str).unique().tolist())}"
        )
    key = ["event_id", "player", "market", "team_abbr", "opponent_abbr"]
    dup = df.duplicated(key, keep=False)
    if dup.any():
        raise RuntimeError(
            "canonical compact live props are not unique at player/market grain: "
            f"{df.loc[dup, key].head(20).to_dict('records')}"
        )
    return df.reset_index(drop=True)


def _offer_json_key(offer: dict) -> str:
    normalized = {}
    for key, value in offer.items():
        if isinstance(value, float) and math.isnan(value):
            value = None
        normalized[str(key)] = value
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)


def materialize_pricing_offers() -> dict:
    compact = _read_compact()
    side_rows: list[dict] = []
    parsed_offer_entries = 0

    carry_candidates = [
        "event_id", "player", "canonical_player_name", "canonical_player_key",
        "player_canonical", "player_raw", "book_player_name", "team_abbr",
        "opponent_abbr", "market", "source", "fetched_at", "commence_time",
    ]

    for compact_idx, row in compact.iterrows():
        market = _text(row.get("market"))
        raw_json = _text(row.get("offers_json"))
        try:
            offers = json.loads(raw_json)
        except Exception as exc:
            raise RuntimeError(
                f"malformed offers_json compact_row={compact_idx} player={row.get('player')} market={market}: {exc}"
            ) from exc
        if not isinstance(offers, list) or not offers:
            raise RuntimeError(
                f"empty/non-list offers_json compact_row={compact_idx} player={row.get('player')} market={market}"
            )

        seen_exact: set[str] = set()
        seen_side_keys: dict[tuple, int] = {}
        before_for_row = len(offers)
        for offer_idx, offer in enumerate(offers):
            if not isinstance(offer, dict):
                raise RuntimeError(f"offers_json entry is not an object row={compact_idx} offer={offer_idx}")
            exact = _offer_json_key(offer)
            if exact in seen_exact:
                raise RuntimeError(
                    "duplicate offer survived live-odds hardening; "
                    f"compact_row={compact_idx} player={row.get('player')} market={market} offer={offer}"
                )
            seen_exact.add(exact)

            book = _text(offer.get("book"))
            book_title = _text(offer.get("book_title")) or book
            side = _text(offer.get("side")).upper().replace("YES", "OVER").replace("NO", "UNDER")
            if not book:
                raise RuntimeError(f"pricing offer has blank book row={compact_idx} player={row.get('player')}")
            if side not in {"OVER", "UNDER"}:
                raise RuntimeError(
                    f"pricing offer has unsupported side={side!r} row={compact_idx} player={row.get('player')}"
                )
            price = _finite_number(offer.get("price"), label="price")
            if price == 0:
                raise RuntimeError(f"pricing offer has invalid American odds 0 row={compact_idx}")

            source_line = offer.get("line")
            if market == ANYTIME_MARKET:
                if source_line is not None and not (isinstance(source_line, float) and math.isnan(source_line)):
                    parsed = _finite_number(source_line, label="line")
                    if abs(parsed - 0.5) > 1e-12:
                        raise RuntimeError(
                            f"anytime-TD source line must be null/0.5; got {parsed} player={row.get('player')}"
                        )
                line = 0.5
                source_line_value = (
                    np.nan
                    if source_line is None or (isinstance(source_line, float) and math.isnan(source_line))
                    else 0.5
                )
            else:
                line = _finite_number(source_line, label="line")
                source_line_value = line

            side_key = (book, float(line), side)
            if side_key in seen_side_keys:
                raise RuntimeError(
                    "multiple distinct prices exist for the same compact player/book/line/side; "
                    f"player={row.get('player')} market={market} key={side_key}"
                )
            seen_side_keys[side_key] = offer_idx

            rec = {c: row.get(c) for c in carry_candidates if c in compact.columns}
            rec.update(
                {
                    "compact_row_index": int(compact_idx),
                    "book": book,
                    "book_title": book_title,
                    "line": float(line),
                    "source_line": source_line_value,
                    "side": side,
                    "price_american": float(price),
                    "offer_source": "canonical_compact_offers_json",
                }
            )
            side_rows.append(rec)
            parsed_offer_entries += 1

        if len(seen_exact) != before_for_row:
            raise RuntimeError("internal offer reconciliation failure")

    sides = pd.DataFrame(side_rows)
    if sides.empty:
        raise RuntimeError("pricing offer expansion produced zero side rows")

    identity_cols = ["event_id", "player", "team_abbr", "opponent_abbr", "market", "book", "line"]
    for col in identity_cols:
        if col not in sides.columns:
            raise RuntimeError(f"expanded pricing side rows missing {col}")
    side_key_cols = [*identity_cols, "side"]
    dup = sides.duplicated(side_key_cols, keep=False)
    if dup.any():
        raise RuntimeError(
            "expanded pricing side rows are duplicate at offer grain: "
            f"{sides.loc[dup, side_key_cols].head(20).to_dict('records')}"
        )

    wide_rows: list[dict] = []
    for key, part in sides.groupby(identity_cols, dropna=False, sort=False):
        by_side = {str(r.side): r for r in part.itertuples(index=False)}
        market = str(part.iloc[0]["market"])
        if "OVER" not in by_side:
            raise RuntimeError(f"pricing book-line is missing OVER side: key={key}")
        if market != ANYTIME_MARKET and "UNDER" not in by_side:
            raise RuntimeError(f"two-sided pricing market is missing UNDER side: key={key}")
        if len(part) != len(by_side):
            raise RuntimeError(f"duplicate side rows within pricing book-line: key={key}")

        first = part.iloc[0].to_dict()
        out = {c: first.get(c) for c in carry_candidates if c in first}
        out.update(
            {
                "player": first.get("player"),
                "event_id": first.get("event_id"),
                "team_abbr": first.get("team_abbr"),
                "opponent_abbr": first.get("opponent_abbr"),
                "market": market,
                "book": first.get("book"),
                "book_title": first.get("book_title"),
                "line": float(first.get("line")),
                "source_line": first.get("source_line"),
                "over_odds": float(by_side["OVER"].price_american),
                "under_odds": float(by_side["UNDER"].price_american) if "UNDER" in by_side else np.nan,
                "compact_row_index": int(first.get("compact_row_index")),
                "offer_source": "canonical_compact_offers_json",
            }
        )
        wide_rows.append(out)

    pricing = pd.DataFrame(wide_rows)
    if pricing.empty:
        raise RuntimeError("pricing offer expansion produced zero book-line rows")
    if pricing.duplicated(identity_cols).any():
        raise RuntimeError("pricing offer output contains duplicate player/market/book/line rows")

    non_td = pricing["market"].astype(str).ne(ANYTIME_MARKET)
    for col in ("line", "over_odds", "under_odds"):
        vals = pd.to_numeric(pricing.loc[non_td, col], errors="coerce")
        if vals.isna().any() or not np.isfinite(vals).all():
            raise RuntimeError(f"two-sided pricing rows contain missing/non-finite {col}")
    td_over = pd.to_numeric(pricing.loc[~non_td, "over_odds"], errors="coerce")
    if td_over.isna().any() or not np.isfinite(td_over).all():
        raise RuntimeError("anytime-TD pricing rows contain missing/non-finite over odds")

    expected_side_rows = int(parsed_offer_entries)
    reconstructed_side_rows = int(
        pricing["over_odds"].notna().sum() + pricing["under_odds"].notna().sum()
    )
    if reconstructed_side_rows != expected_side_rows:
        raise RuntimeError(
            "pricing offer reconciliation failed; "
            f"parsed_side_entries={expected_side_rows} reconstructed={reconstructed_side_rows}"
        )

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    pricing.to_csv(PRICING, index=False)
    pricing.to_csv(METRICS_INPUT, index=False)

    market_counts = pricing["market"].astype(str).value_counts().sort_index().astype(int).to_dict()
    audit = {
        "disposition": "PRICING_OFFERS_MATERIALIZED_EXACTLY_ONCE",
        "compact_rows": int(len(compact)),
        "compact_unique_players": int(compact["player"].nunique()),
        "parsed_offer_side_entries": expected_side_rows,
        "pricing_book_line_rows": int(len(pricing)),
        "reconstructed_offer_side_entries": reconstructed_side_rows,
        "market_book_line_rows": market_counts,
        "duplicate_book_line_rows": 0,
        "consensus_line_created": False,
        "source": str(COMPACT if COMPACT.exists() else LEGACY_COMPACT),
        "pricing_artifact": str(PRICING),
        "metrics_input_artifact": str(METRICS_INPUT),
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if STATUS.exists() and STATUS.stat().st_size > 0:
        status = json.loads(STATUS.read_text(encoding="utf-8"))
        status["pricing_offer_disposition"] = audit["disposition"]
        status["pricing_offer_side_entries"] = expected_side_rows
        status["pricing_book_line_rows"] = int(len(pricing))
        status["pricing_market_book_line_rows"] = market_counts
        status["pricing_consensus_line_created"] = False
        STATUS.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[pricing_offer_adapter] " + json.dumps(audit, sort_keys=True))
    return audit


def main() -> int:
    materialize_pricing_offers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
