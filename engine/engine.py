# engine/engine.py
"""
Main orchestration pipeline for the Sports Betting Model.
Coordinates all data fetching, enrichment, pricing, and predictive modeling steps.

Version: 2025 NFL Season
Last updated: October 2025
"""

import os
import sys
import uuid
import time
import json
import shutil
import subprocess
from datetime import datetime

# ------------------------------------------
# Utilities
# ------------------------------------------

def _run(cmd: str, check: bool = True):
    """Run a subprocess command and stream output."""
    print(f"\n[ENGINE] ▶ Running: {cmd}")
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    dur = time.time() - start
    print(f"[ENGINE] ✅ Completed in {dur:.2f}s\n")
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result.returncode


def _safe_mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _write_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ------------------------------------------
# Engine core
# ------------------------------------------

def run_pipeline(season: int = 2025,
                 date: str = "",
                 bookmakers: str = "",
                 markets: str = "",
                 debug: bool = False):
    """
    Execute the full model pipeline:
      1) Fetch odds and external data
      2) Build metrics (team/player)
      3) Price props
      4) Run predictive models
      5) Export results
    """

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    runs_dir = os.path.join("runs", run_id)
    _safe_mkdir(runs_dir)
    print(f"[ENGINE] 🚀 Starting run {run_id} (season {season})")

    # -------------------------
    # STEP 1: Fetch data
    # -------------------------
    try:
        # Fetch live odds & lines
        _run("python scripts/fetch_props_oddsapi.py")

        # Optional: Depth charts / rosters (soft-fail)
        _run("python scripts/providers/espn_depth.py || true")
        _run("python scripts/providers/ourlads_depth.py || true")

        # Fetch injury reports (soft-fail)
        _run("python scripts/providers/injuries.py || true")

        # Optional: PFR / EPA enrichment sources
        _run("python scripts/providers/pfr_team.py || true")

    except Exception as e:
        print(f"[ENGINE] ⚠️ Data fetch step failed: {e}")

    # -------------------------
    # STEP 2: Build metrics
    # -------------------------
    try:
        print("\n[ENGINE] 🧮 Building TEAM metrics...")
        _run(f"python scripts/make_team_form.py --season {season}")
        _run("python scripts/enrich_team_form.py || true")

        print("\n[ENGINE] 🧮 Building PLAYER metrics...")
        _run(f"python scripts/make_player_form.py --season {season}")
        _run("python scripts/enrich_player_form.py || true")

        print("\n[ENGINE] 🔗 Joining metrics for pricing...")
        _run(f"python scripts/make_metrics.py --season {season}")

    except Exception as e:
        print(f"[ENGINE] ⚠️ Metrics build failed: {e}")

    # -------------------------
    # STEP 3: Price props
    # -------------------------
    try:
        _run(f"python scripts/pricing.py --season {season}")
    except Exception as e:
        print(f"[ENGINE] ⚠️ Pricing step failed: {e}")

    # -------------------------
    # STEP 4: Predictive models
    # -------------------------
    try:
        _run(f"python scripts/models/monte_carlo.py --season {season}")
        _run(f"python scripts/models/bayes_hier.py --season {season}")
        _run(f"python scripts/models/markov.py --season {season}")
        _run(f"python scripts/models/ml_ensemble.py --season {season}")
    except Exception as e:
        print(f"[ENGINE] ⚠️ Predictor step failed: {e}")

    # -------------------------
    # STEP 5: Export & archive
    # -------------------------
    try:
        # Copy key artifacts into this run folder
        for f in [
            "outputs/props_raw.csv",
            "outputs/props_priced_clean.csv",
            "data/team_form.csv",
            "data/player_form.csv",
            "data/metrics_ready.csv",
        ]:
            if os.path.exists(f):
                shutil.copy(f, os.path.join(runs_dir, os.path.basename(f)))

        meta = {
            "run_id": run_id,
            "season": season,
            "date": date,
            "bookmakers": bookmakers,
            "markets": markets,
            "timestamp": datetime.utcnow().isoformat(),
        }
        _write_json(os.path.join(runs_dir, "meta.json"), meta)

        print(f"[ENGINE] 🏁 Run completed successfully → {runs_dir}")

    except Exception as e:
        print(f"[ENGINE] ⚠️ Export step failed: {e}")

    # -------------------------
    # STEP 6: Debug snapshot
    # -------------------------
    if debug:
        for path in [
            "data/team_form.csv",
            "data/player_form.csv",
            "data/metrics_ready.csv",
            "outputs/props_priced_clean.csv",
        ]:
            if os.path.exists(path):
                import pandas as pd
                df = pd.read_csv(path)
                print(f"\n[DEBUG] {path}: {len(df)} rows, {len(df.columns)} cols")
                print(df.head(3).to_string(index=False))

    print("[ENGINE] ✅ All done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Sports Betting model pipeline.")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--bookmakers", type=str, default="")
    parser.add_argument("--markets", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        season=args.season,
        date=args.date,
        bookmakers=args.bookmakers,
        markets=args.markets,
        debug=args.debug,
    )
