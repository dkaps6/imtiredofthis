# engine/__main__.py
from .engine import run_pipeline
import argparse

def main():
    ap = argparse.ArgumentParser(prog="python -m engine", description="Run full slate pipeline")
    ap.add_argument("--season", required=True, help="Season year, e.g. 2025")
    ap.add_argument("--date", default="", help="Optional slate date YYYY-MM-DD")
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--markets", default="", help="Comma list of markets to request from Odds API")
    args = ap.parse_args()

    run_pipeline(
        season=args.season,
        date=args.date,
        books=[b.strip() for b in args.books.split(",") if b.strip()],
        markets=[m.strip() for m in args.markets.split(",") if m.strip()] or None,
    )

if __name__ == "__main__":
    main()
