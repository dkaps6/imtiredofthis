#!/usr/bin/env python3
"""Check a drafted ticket card against the operator's binding construction rules.

Source of the rules: WEEK8_USER_BINDING_TICKET_PAYOUT_AND_CONSTRUCTION_OVERRIDE
(with its 2026-11-13 SAFE-floor addendum), the lottery rejection record, and
MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1. This script decides nothing about
which bets are good. It only refuses a card that breaks a rule the operator has
already stated, so the rule does not depend on anyone remembering it.

Input CSV, one row per leg:
  ticket, tier, player, market, side, threshold, price_american, book,
  football_reason, state_label, price_basis
  tier        SAFE | MODERATE | MEDIUM | HIGH | LOTTERY | MOONSHOT
  price_basis quote (a real book parlay quote) | naive_product (legs multiplied)
  state_label HEALTHY_FULL | INJURY_LIMITED | PARTIAL_GAME | DNP_INACTIVE |
              RETURN_FROM_INJURY | UNRESOLVED_CONTEXT
Optional per-ticket column `status` = UNSOLVED for a LOTTERY that cannot be built.

Exit code 1 if any ERROR is raised. WARNINGs never fail the run but are printed.
"""
from __future__ import annotations
import argparse, sys
import pandas as pd

SAFE_FLOOR_AMERICAN = 150            # strictly over, per the 2026-11-13 addendum
LOTTERY_MIN_AMERICAN = 24900         # 250x gross on the stake (override + rejection record)
MAX_REGULAR_TICKETS_PER_PLAYER = 2   # five regular tickets; MOONSHOT is separate
REGULAR = {"SAFE", "MODERATE", "MEDIUM", "HIGH", "LOTTERY"}
BAD_STATES = {"DNP_INACTIVE", "RETURN_FROM_INJURY", "UNRESOLVED_CONTEXT", "PARTIAL_GAME"}
NEED = ["ticket", "tier", "player", "market", "side", "threshold", "price_american",
        "book", "football_reason", "state_label", "price_basis"]


def dec(a: float) -> float:
    a = float(a)
    return 1 + a / 100 if a > 0 else 1 + 100 / -a


def amer(d: float) -> int:
    return int(round((d - 1) * 100)) if d >= 2 else int(round(-100 / (d - 1)))


def validate(df: pd.DataFrame, safe_floor: float = SAFE_FLOOR_AMERICAN) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    E = lambda t, m: out.append(("ERROR", str(t), m))
    W = lambda t, m: out.append(("WARNING", str(t), m))
    miss = [c for c in NEED if c not in df.columns]
    if miss:
        return [("ERROR", "-", f"missing columns {miss}")]
    d = df.copy()
    d["tier"] = d.tier.astype(str).str.upper().str.strip()
    d["side"] = d.side.astype(str).str.upper().str.strip()
    d["market"] = d.market.astype(str).str.lower().str.strip()
    for t, g in d.groupby("ticket", sort=False):
        tier = g.tier.iloc[0]
        if g.tier.nunique() > 1:
            E(t, "ticket carries more than one tier"); continue
        books = set(g.book.astype(str).str.lower().str.strip())
        if len(books) != 1 or "" in books or "nan" in books:
            E(t, f"legs must all be priced at ONE book to be placeable; found {sorted(books)}")
        if g.player.duplicated().any():
            E(t, "same player twice on one ticket")
        prod = 1.0
        for p in g.price_american:
            prod *= dec(p)
        odds = amer(prod)
        basis = set(g.price_basis.astype(str).str.lower().str.strip())
        if basis != {"quote"}:
            W(t, f"combined {odds:+d} is {'/'.join(sorted(basis))}, not a verified book quote; "
                 "state it as an estimate, never as the price")
        if tier == "SAFE" and not odds > safe_floor:
            E(t, f"SAFE combined odds {odds:+d} is not over +{safe_floor:g}")
        unsolved = "status" in g.columns and g.status.astype(str).str.upper().eq("UNSOLVED").any()
        if tier == "LOTTERY" and not unsolved and odds < LOTTERY_MIN_AMERICAN:
            E(t, f"LOTTERY combined {odds:+d} is under +{LOTTERY_MIN_AMERICAN} (250x); "
                 "build it properly or mark it UNSOLVED, do not rebrand a smaller ticket")
        if (g.market == "receptions").all():
            E(t, "ticket is made only of reception lines")
        for r in g.itertuples():
            if not str(r.football_reason).strip() or str(r.football_reason).strip().lower() == "nan":
                E(t, f"{r.player} {r.market}: no football reason stated")
            if str(r.state_label).strip().upper() in BAD_STATES:
                W(t, f"{r.player}: state {r.state_label} cannot be affirmative evidence (anomaly gate)")
            if r.market in {"rush_yards", "rush_rec_yards"} and r.side == "UNDER":
                W(t, f"{r.player} {r.market} UNDER is in the known-systematic RB under family; "
                     "needs an explicit football exception")
    reg = d[d.tier.isin(REGULAR)]
    cnt = reg.groupby("player").ticket.nunique()
    for p, n in cnt[cnt > MAX_REGULAR_TICKETS_PER_PLAYER].items():
        E("-", f"{p} appears on {n} regular tickets (limit {MAX_REGULAR_TICKETS_PER_PLAYER})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("card")
    ap.add_argument("--safe-floor", type=float, default=SAFE_FLOOR_AMERICAN)
    a = ap.parse_args()
    res = validate(pd.read_csv(a.card), a.safe_floor)
    for sev, t, m in res:
        print(f"{sev:8s} [{t}] {m}")
    errs = sum(1 for s, *_ in res if s == "ERROR")
    print(f"\n{errs} error(s), {len(res) - errs} warning(s)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
