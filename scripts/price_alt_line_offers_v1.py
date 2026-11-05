#!/usr/bin/env python3
"""Price captured ALTERNATE-line offers against the empirical clearance table.

MARKET-ASSISTED, and deliberately so. This does NOT use the football model's
probability, and that is the whole point:

  pricing_v2._prob_over is 1 - Phi((line - mu)/sigma) -- a NORMAL approximation
  around a mean that is compressed in volume. Measured on 2024+2025, the
  resulting p_over is wrong by up to +0.448 (the model asserts a 3.9% chance of
  an over that happens 48.7% of the time on high-volume rush_rec_yards). A
  Normal fitted to a zero-inflated, right-skewed variable is worst exactly in
  the tails, which is where every alternate line lives. Pricing alt ladders off
  that distribution would be actively harmful.

So the probability comes from realized outcomes instead: the market-anchored,
volume-stratified clearance table (docs/research/PRISM_ALT_LINE_ANCHOR_RESULTS_V1.md),
which expresses a threshold as a fraction of the SPORTSBOOK LINE -- a near
unbiased median, realized over-rate 0.476-0.508 in every quartile -- and reads
clearance off two seasons of outcomes.

The sportsbook supplies the offered price. Edge = empirical clearance vs that
price. This script joins the two and ranks what is actually bettable.

It writes outputs/alt_line_edges.csv and changes no football projection.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data"
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
TABLE = DATA / "research" / "alt_line_clearance_v2.csv"
RECP = DATA / "research" / "alt_line_receptions_absolute_v2.csv"
BOUNDS = DATA / "research" / "alt_line_quartile_boundaries_v1.csv"

# Provider key -> canonical market. Both the alternate ladder keys and the
# main-line keys, because the main line is the ANCHOR the table is indexed on.
ALT_TO_CANON = {
    "player_pass_yds_alternate": "pass_yards",
    "player_rush_yds_alternate": "rush_yards",
    "player_reception_yds_alternate": "rec_yards",
    "player_receptions_alternate": "receptions",
    "player_rush_reception_yds_alternate": "rush_rec_yards",
}
MAIN_TO_CANON = {
    "player_pass_yds": "pass_yards",
    "player_rush_yds": "rush_yards",
    "player_reception_yds": "rec_yards",
    "player_receptions": "receptions",
    "player_rush_reception_yds": "rush_rec_yards",
}
# receptions is an INTEGER market, so it is priced from an ABSOLUTE-threshold
# table rather than a fraction of the line: 0.4x, 0.5x, 0.6x and 0.7x of a 2.5
# line all mean ">= 2". pass_yards is priced from a POOLED row (line_quartile
# -1) because stratification was measured not to help it.
INTEGER_MARKETS = {"receptions"}
POOLED_MARKETS = {"pass_yards"}


def _profit_per_dollar(american: float) -> float:
    """Profit on a winning $1 stake."""
    if not np.isfinite(american) or american == 0:
        return np.nan
    return american / 100.0 if american > 0 else 100.0 / abs(american)


def _ev(p: float, american: float) -> float:
    """EV per $1 staked. Negative means the price is worse than the odds of it."""
    prof = _profit_per_dollar(american)
    if not np.isfinite(p) or not np.isfinite(prof):
        return np.nan
    return p * prof - (1.0 - p)


def _quartile(line: float, q: tuple[float, float, float]) -> int:
    return int(np.digitize([line], list(q))[0])


def load_table():
    t = pd.read_csv(TABLE)
    r = pd.read_csv(RECP) if RECP.exists() else pd.DataFrame()
    b = pd.read_csv(BOUNDS)
    bounds = {str(x["market"]): (float(x["q25"]), float(x["q50"]), float(x["q75"]))
              for _, x in b.iterrows()}
    return t, r, bounds


def clearance(table: pd.DataFrame, market: str, qt: int, frac: float) -> tuple[float, float]:
    """Interpolated clearance and its drift-aware conservative bound.

    Returns (nan, nan) outside the table's supported fraction range rather than
    extrapolating: an alt ladder routinely quotes thresholds far beyond what two
    seasons of data can speak to, and a made-up number there is worse than none.
    """
    cell = table[(table["market"] == market) & (table["line_quartile"] == qt)]
    if cell.empty:
        return (np.nan, np.nan)
    cell = cell.sort_values("frac_of_LINE")
    xs = cell["frac_of_LINE"].to_numpy(float)
    if frac < xs.min() or frac > xs.max():
        return (np.nan, np.nan)
    return (float(np.interp(frac, xs, cell["cleared"].to_numpy(float))),
            float(np.interp(frac, xs, cell["cleared_conservative"].to_numpy(float))))


def clearance_integer(recp: pd.DataFrame, main_line: float, alt_line: float):
    """P(actual >= k) for an integer market, where k follows from the alt line.

    A book quoting "over 3.5 receptions" means ">= 4", so k = floor(T) + 1.
    """
    if recp.empty:
        return (np.nan, np.nan)
    k = int(np.floor(float(alt_line))) + 1
    cell = recp[(np.isclose(recp["main_line"], float(main_line))) &
                (recp["threshold_at_least"] == k)]
    if cell.empty:
        return (np.nan, np.nan)
    r = cell.iloc[0]
    return (float(r["cleared"]), float(r["cleared_conservative"]))


def _name_key(value) -> str:
    """Exact join key: lowercased alphanumerics of the whole name, suffix-stripped."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    raw = str(value).strip().lower().replace("\u2019", "'")
    if raw in {"", "nan", "none", "null", "<na>"}:
        return ""
    tokens = [t for t in (re.sub(r"[^a-z0-9]", "", tok) for tok in re.split(r"\s+", raw)) if t]
    while tokens and tokens[-1] in _SUFFIXES:
        tokens.pop()
    return "".join(tokens)


def _initial_last_key(value) -> str:
    """Fallback join key: first initial + surname.

    The alternate ladder and the main line are two separate OddsAPI calls and
    the books do not always spell a name the same way in both ("Matt Hibner"
    vs "Matthew Hibner"). Without this an alt ladder silently loses its
    anchor and every row on it comes back no_main_line_anchor.
    """
    if value is None:
        return ""
    raw = str(value).strip().lower().replace("\u2019", "'")
    tokens = [t for t in (re.sub(r"[^a-z0-9]", "", tok) for tok in re.split(r"\s+", raw)) if t]
    while tokens and tokens[-1] in _SUFFIXES:
        tokens.pop()
    if len(tokens) < 2:
        return ""
    return f"{tokens[0][0]}{tokens[-1]}"


def _main_player_series(main: pd.DataFrame) -> pd.Series:
    for col in ("book_player_name", "canonical_player_name", "player", "player_canonical", "player_raw"):
        if col in main.columns:
            vals = main[col].astype("string").fillna("").str.strip()
            if vals.ne("").any():
                return vals
    raise RuntimeError(
        f"main props artifact has no usable player-name column; columns={list(main.columns)}"
    )


def flatten_main(main: pd.DataFrame) -> pd.DataFrame:
    """Return main-line rows as (event_id, player, market, line), whichever schema.

    Production's outputs/props_raw.csv is the GROUPED artifact: one row per
    (event, market, player) with the per-book lines inside an offers_json
    blob, and NO 'line' column at all. Run 42094118645 bought the Week 9
    Thursday odds, captured 738 alternate offers, and then this script died on
    KeyError: 'line' because it had only ever been exercised against the older
    FLAT archive schema (data/market_track_record/raw_props/2026_wk08.csv),
    which does carry player/line/over_odds. Handle both, and say so loudly if
    neither is present, rather than crashing on a column lookup.
    """
    if main.empty:
        return pd.DataFrame(columns=["event_id", "player", "market", "line"])

    if "line" in main.columns:
        flat = main.copy()
        flat["player"] = _main_player_series(flat)
        flat["line"] = pd.to_numeric(flat["line"], errors="coerce")
        return flat[["event_id", "player", "market", "line"]]

    if "offers_json" not in main.columns:
        raise RuntimeError(
            "main props artifact carries neither a 'line' column nor 'offers_json'; "
            f"columns={list(main.columns)}"
        )

    players = _main_player_series(main)
    recs: list[dict] = []
    for (event_id, market, player, blob) in zip(
        main.get("event_id", pd.Series("", index=main.index)),
        main.get("market", pd.Series("", index=main.index)),
        players,
        main["offers_json"],
    ):
        if not isinstance(blob, str) or not blob.strip():
            continue
        try:
            offers = json.loads(blob)
        except (ValueError, TypeError):
            continue
        if isinstance(offers, dict):
            offers = offers.get("bookmakers") or []
        if not isinstance(offers, list):
            continue
        for off in offers:
            if not isinstance(off, dict):
                continue
            # OVER and UNDER carry the same posted line; keep OVER so a
            # one-sided book cannot skew the median.
            if str(off.get("side", "")).upper() not in {"OVER", ""}:
                continue
            line = pd.to_numeric(off.get("line"), errors="coerce")
            if not np.isfinite(line):
                continue
            recs.append({"event_id": event_id, "player": player,
                         "market": market, "line": float(line)})
    return pd.DataFrame.from_records(
        recs, columns=["event_id", "player", "market", "line"]
    )


def build(alt: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    table, recp, bounds = load_table()

    alt = alt.copy()
    alt["canon"] = alt["market"].map(ALT_TO_CANON)
    alt = alt[alt["canon"].notna()]
    alt["line"] = pd.to_numeric(alt["line"], errors="coerce")
    alt["price_american"] = pd.to_numeric(alt["price_american"], errors="coerce")
    alt = alt.dropna(subset=["line", "price_american"])
    alt["_k"] = alt["player"].map(_name_key)
    alt["_kf"] = alt["player"].map(_initial_last_key)

    main = flatten_main(main)
    main["canon"] = main["market"].map(MAIN_TO_CANON)
    main = main[main["canon"].notna()]
    main = main.dropna(subset=["line"])
    main["_k"] = main["player"].map(_name_key)
    main["_kf"] = main["player"].map(_initial_last_key)

    # One anchor per player-market: the median posted line across books.
    anchor = (main.groupby(["event_id", "_k", "canon"], as_index=False)["line"]
                  .median().rename(columns={"line": "main_line"}))
    m = alt.merge(anchor, on=["event_id", "_k", "canon"], how="left")

    # Spelling fallback, applied only where the exact key found no anchor and
    # the initial+surname form is unambiguous inside that event and market.
    fb = main.groupby(["event_id", "_kf", "canon"]).agg(
        main_line_fb=("line", "median"), _n=("_k", "nunique")
    ).reset_index()
    fb = fb.loc[fb["_n"].eq(1), ["event_id", "_kf", "canon", "main_line_fb"]]
    m = m.merge(fb, on=["event_id", "_kf", "canon"], how="left")
    m["main_line"] = m["main_line"].fillna(m["main_line_fb"])

    rows = []
    for _, r in m.iterrows():
        canon, ml = str(r["canon"]), r["main_line"]
        reason = ""
        frac = clr = lo = np.nan
        if not np.isfinite(ml) or ml <= 0:
            reason = "no_main_line_anchor"
        elif canon in INTEGER_MARKETS:
            frac = float(r["line"]) / float(ml)
            clr, lo = clearance_integer(recp, ml, float(r["line"]))
            if not np.isfinite(clr):
                reason = "integer_cell_outside_table_support"
        elif canon not in bounds and canon not in POOLED_MARKETS:
            reason = "no_quartile_boundaries"
        else:
            frac = float(r["line"]) / float(ml)
            qt = -1 if canon in POOLED_MARKETS else _quartile(float(ml), bounds[canon])
            clr, lo = clearance(table, canon, qt, frac)
            if not np.isfinite(clr):
                reason = "fraction_outside_table_support"
        side = str(r.get("side", "")).upper()
        # The table gives P(actual > threshold). An UNDER wins on the complement.
        #
        # cleared_conservative is a ONE-SIDED downward haircut on the clearance
        # rate: the cluster-bootstrap sampling margin plus a season-drift
        # allowance. For an OVER that is already the conservative direction.
        # For an UNDER it is not -- an UNDER is hurt by clearance being HIGHER
        # than measured, so taking 1 - cleared as "conservative" applies no
        # haircut whatsoever and makes every UNDER outrank every OVER on the
        # ev_conservative column that this board is sorted by. Mirror the same
        # haircut magnitude in the direction that actually hurts the UNDER.
        p = clr if side == "OVER" else (1.0 - clr if np.isfinite(clr) else np.nan)
        if side == "OVER":
            p_cons = lo
        elif np.isfinite(clr) and np.isfinite(lo):
            p_cons = float(np.clip(1.0 - clr - (clr - lo), 0.0, 1.0))
        else:
            p_cons = np.nan
        rows.append({
            "event_id": r.get("event_id"), "player": r.get("player"),
            "market": canon, "book": r.get("book"), "side": side,
            "alt_threshold": float(r["line"]), "main_line": ml,
            "frac_of_line": frac,
            "offered_american": float(r["price_american"]),
            "clearance": clr, "clearance_conservative": lo,
            "p_side": p, "p_side_conservative": p_cons,
            "ev_per_dollar": _ev(p, float(r["price_american"])),
            "ev_conservative": _ev(p_cons, float(r["price_american"])),
            "unpriceable_reason": reason,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["ev_conservative", "ev_per_dollar"], ascending=False,
                          na_position="last").reset_index(drop=True)
    return out


def main_cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alt", type=Path, default=OUTPUTS / "props_alt_raw.csv")
    ap.add_argument("--main", type=Path, default=OUTPUTS / "props_raw.csv")
    ap.add_argument("--out", type=Path, default=OUTPUTS / "alt_line_edges.csv")
    a = ap.parse_args()

    if not a.alt.exists() or not a.alt.stat().st_size:
        print(f"[alt_edges] no alternate-line capture at {a.alt}; nothing to price")
        return 0
    if not a.main.exists() or not a.main.stat().st_size:
        print(f"[alt_edges] FATAL: {a.main} missing; the main line is the anchor "
              f"the clearance table is indexed on", file=sys.stderr)
        return 1

    out = build(pd.read_csv(a.alt, low_memory=False),
                pd.read_csv(a.main, low_memory=False))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    if out.empty:
        print("[alt_edges] no rows")
        return 0
    priced = out[out["unpriceable_reason"].eq("")]
    pos = priced[priced["ev_conservative"] > 0]
    print(f"[alt_edges] wrote {a.out} rows={len(out)} priced={len(priced)} "
          f"positive_EV_conservative={len(pos)}")
    if not priced.empty:
        print(out["unpriceable_reason"].value_counts().to_string())
    for _, r in pos.head(15).iterrows():
        print(f"  {r['player'][:22]:22s} {r['market']:15s} {r['side']:5s} "
              f"{r['alt_threshold']:6.1f} ({r['frac_of_line']:.2f}x{r['main_line']:.1f}) "
              f"@ {int(r['offered_american']):+5d}  clears {r['clearance']*100:.1f}%  "
              f"EV {r['ev_per_dollar']:+.3f} / cons {r['ev_conservative']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
