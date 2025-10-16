# engine/engine.py
"""
Main orchestration pipeline for the Sports Betting Model.
Coordinates all data fetching, enrichment, pricing, and predictive modeling steps.

Version: 2025 NFL Season
"""

from __future__ import annotations
import os
import uuid
import time
import json
import shutil
import subprocess
import shlex
import sys
from datetime import datetime

# ------------------------------------------
# Shell helpers
# ------------------------------------------

def _run(cmd: str, check: bool = True):
    print(f"[ENGINE] ► {cmd}")
    res = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print(res.stderr.strip(), file=sys.stderr)
    if check and res.returncode != 0:
        raise SystemExit(res.returncode)

def _assert_nonempty_csv(path: str, name: str):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if os.path.getsize(path) <= 1:
            raise RuntimeError(f"{name} appears empty at {path}")
        import pandas as pd
        df = pd.read_csv(path, nrows=2)
        print(f"[ENGINE] • {name}: {len(df)} rows ({len(df.columns)} cols)")
    except Exception as e:
        if os.path.getsize(path) <= 1:
            raise RuntimeError(f"{name} appears empty ({path}); {e}")

# ------------------------------------------
# Engine core
# ------------------------------------------

def run_pipeline(season: int = 2025,
                 date: str = "",
                 books: str = "",           # accept --books
                 bookmakers: str = "",      # accept --bookmakers
                 markets: str = "",
                 debug: bool = False):

    runs_dir = os.path.join("outputs", f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}")
    os.makedirs(runs_dir, exist_ok=True)

    try:
        print("\n[ENGINE] 🧹 Preflight…")
        os.makedirs("data", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)

        # -------------------------------------------------
        # 1) Props & games (keep your existing fetchers)
        # -------------------------------------------------
        print("\n[ENGINE] 🎣 Fetching props & games…")
        _run(f"python scripts/fetch_props.py --season {season} --date {date} --books \"{books}\" --markets \"{markets}\"")
        _assert_nonempty_csv("outputs/props_raw.csv", "props_raw")

        _run(f"python scripts/fetch_games.py --season {season} --date {date}")
        _assert_nonempty_csv("outputs/odds_game.csv", "odds_game")

        # -------------------------------------------------
        # 2) TEAM metrics
        # -------------------------------------------------
        print("\n[ENGINE] 🧮 Building TEAM metrics…")
        _run(f"python scripts/make_team_form.py --season {season}")
        _assert_nonempty_csv("data/team_form.csv", "team_form")
        _run("python scripts/enrich_team_form.py || true")  # <-- added; non-fatal if absent
        _assert_nonempty_csv("data/team_form_weekly.csv", "team_form_weekly")

        # -------------------------------------------------
        # 3) PLAYER metrics
        # -------------------------------------------------
        print("\n[ENGINE] 🧮 Building PLAYER metrics…")
        _run(f"python scripts/make_player_form.py --season {season}")
        _run("python scripts/enrich_player_form.py || true")
        _assert_nonempty_csv("data/player_form.csv", "player_form")

        # -------------------------------------------------
        # 4) Assemble metrics_ready
        # -------------------------------------------------
        print("\n[ENGINE] 🔗 Joining metrics for pricing…")
        _run(f"python scripts/make_metrics.py --season {season}")
        _assert_nonempty_csv("data/metrics_ready.csv", "metrics_ready")

        # -------------------------------------------------
        # 5) Predictors / Pricing
        # -------------------------------------------------
        print("\n[ENGINE] 🤖 Running predictors & pricing…")
        _run(f"python scripts/models/run_predictors.py --season {season}")
        _run(f"python scripts/pricing.py --season {season}")

        # -------------------------------------------------
        # Artifacts
        # -------------------------------------------------
        print("\n[ENGINE] 📦 Exporting artifacts…")
        for f in ["outputs/props_raw.csv",
                  "outputs/odds_game.csv",
                  "data/team_form.csv",
                  "data/team_form_weekly.csv",
                  "data/player_form.csv",
                  "data/metrics_ready.csv",
                  "outputs/props_priced_clean.csv"]:
            if os.path.exists(f):
                shutil.copy2(f, os.path.join(runs_dir, os.path.basename(f)))

        meta = {
            "run_id": runs_dir.rsplit("_", 1)[-1],
            "season": season,
            "date": date,
            "bookmakers": bookmakers or books,
            "markets": markets,
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(os.path.join(runs_dir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

        print(f"[ENGINE] 🏁 Run completed → {runs_dir}")

    except Exception as e:
        print(f"[ENGINE] ⚠️ Export step failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--books", type=str, default="")
    parser.add_argument("--bookmakers", type=str, default="")
    parser.add_argument("--markets", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        season=args.season,
        date=args.date,
        books=args.books,
        bookmakers=args.bookmakers,
        markets=args.markets,
        debug=args.debug,
    )
