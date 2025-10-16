# engine/__main__.py
from __future__ import annotations
import argparse
from .engine import run_pipeline

def cli_main():
    parser = argparse.ArgumentParser(prog="engine")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", type=str, default="")
    # Accept BOTH flags; use whichever the user supplied
    parser.add_argument("--books", type=str, default="")
    parser.add_argument("--bookmakers", type=str, default="")
    parser.add_argument("--markets", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Route to the function using the 'bookmakers' name
    use_books = args.bookmakers or args.books
    return run_pipeline(
        season=args.season,
        date=args.date,
        bookmakers=use_books,
        markets=args.markets,
        debug=args.debug,
    )

if __name__ == "__main__":
    raise SystemExit(cli_main())
