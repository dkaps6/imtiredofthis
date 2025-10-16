# engine/__main__.py
from __future__ import annotations
import argparse
from .engine import run_pipeline  # this must be the orchestrator that runs builders+pricing

def cli_main():
    p = argparse.ArgumentParser(prog="engine")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--date", type=str, default="")
    p.add_argument("--books", type=str, default="")
    p.add_argument("--bookmakers", type=str, default="")
    p.add_argument("--markets", type=str, default="")
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()

    books = a.bookmakers or a.books  # accept either flag
    return run_pipeline(
        season=a.season,
        date=a.date,
        bookmakers=books,
        markets=a.markets,
        debug=a.debug,
    )

if __name__ == "__main__":
    raise SystemExit(cli_main())
