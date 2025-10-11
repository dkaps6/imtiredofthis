from __future__ import annotations
import argparse
from .engine import run_pipeline

def cli_main() -> int:
    ap = argparse.ArgumentParser(prog="engine", description="Run full slate pipeline")
    ap.add_argument("--season", required=True, help="Season year (e.g., 2025)")
    ap.add_argument("--date", default="", help="Slate date YYYY-MM-DD (optional)")
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--markets", default="", help="Comma-separated player markets; leave empty to probe")
    a = ap.parse_args()

    return run_pipeline(
        season=a.season,
        date=a.date,
        books=[b.strip() for b in a.books.split(",") if b.strip()],
        markets=[m.strip() for m in a.markets.split(",") if m.strip()] or None,
    )

if __name__ == "__main__":
    raise SystemExit(cli_main())
