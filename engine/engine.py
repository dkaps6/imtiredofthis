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
from datetime import datetime

# ------------------------------------------
# Utilities
# ------------------------------------------

def _run(cmd: str, check: bool = True):
    """Run a subprocess command and stream output."""
    print(f"\n[ENGINE] ▶ {cmd}")
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    dur = time.time() - start
    print(f"[ENGINE] ✅ Completed in {dur:.2f}s")
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result.returncode


def _safe_mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _write_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _assert_nonempty_csv(path: str, name: str):
    """Fail fast if a required artifact is missing or empty."""
    if not os.path.exists(path):
        raise RuntimeError(f"{name} not written: missing file → {path}")
    try:
        import pandas as pd  # local import to avoid hard dependency if not needed
        df = pd.read_csv(path)
        if df.empty:
            raise RuntimeError(f"{name} is empty; check build logs → {path}")
        print(f"[ENGINE] • {name}: {len(df)} rows ({len(df.columns)} cols)")
    except Exception as e:
        # If pandas isn't available here for some reason, at least check size > 0
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
    """
    Execute the full model pipeline:
      1) Fetch odds and external data
      2) Build metrics (team/player)
      3) Validate completeness (strict)
      4) Price props
      5) (Optional) Predictive models
      6) Export & snapshot
    """

    # Normalize books/bookmakers arg
    if not bookmakers and books:
        bookmakers = books

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    runs_dir = os.path.join("runs", run_id)
    _safe_mkdir(runs_dir)
    print(f"[ENGINE] 🚀 Run {run_id} | season={season} date='{date}' books='{bookmakers}' markets='{markets}'")

    # -------------------------
    # STEP 1: Fetch data (props + optional providers)
    # -------------------------
    try:
        # Sharp Football pull must succeed before we fetch props.
        _run(f"python scripts/providers/sharpfootball_pull.py --season {season}")
        _assert_nonempty_csv("data/sharp_team_form.csv", "sharp_team_form")

        # Your props fetcher is stable—leave it untouched.
        _run("python scripts/fetch_props_oddsapi.py")

        # Optional providers (soft-fail)
        _run("python scripts/providers/espn_depth.py || true")
        _run("python scripts/providers/ourlads_depth.py || true")
        _run("python scripts/providers/injuries.py || true")
        _run("python scripts/providers/pfr_team.py || true")

    except Exception as e:
        print(f"[ENGINE] ⚠️ Data fetch step failed: {e}")
        raise

    # -------------------------
    # STEP 2: Build metrics (REQUIRED)
    # -------------------------
    try:
        print("\n[ENGINE] 🧮 Building TEAM metrics…")
        _run(f"python scripts/make_team_form.py --season {season}")
        _assert_nonempty_csv("data/team_form.csv", "team_form")

        print("\n[ENGINE] 🧮 Building PLAYER metrics…")
        _run(f"python scripts/make_player_form.py --season {season}")
        _run("python scripts/enrich_player_form.py || true")
        _assert_nonempty_csv("data/player_form.csv", "player_form")

        print("\n[ENGINE] 🔗 Joining metrics for pricing…")
        _run(f"python scripts/make_metrics.py --season {season}")
        _assert_nonempty_csv("data/metrics_ready.csv", "metrics_ready")

    except Exception as e:
        print(f"[ENGINE] ❌ Metrics build failed: {e}")
        raise  # hard fail; pricing without metrics is pointless

    # -------------------------
    # STEP 3: Validate completeness (STRICT)
    # -------------------------
    try:
        strict_flag = os.getenv("STRICT_VALIDATION", "1")
        if strict_flag == "1":
            print("\n[ENGINE] ✅ Validating 2025 completeness (strict)…")
            _run("python scripts/validate_metrics.py")
        else:
            print("\n[ENGINE] ⚠️ STRICT_VALIDATION=0 → skipping strict validator.")
    except Exception as e:
        print(f"[ENGINE] ❌ Validation failed: {e}")
        raise

    # -------------------------
    # STEP 4: Price props (REQUIRED)
    # -------------------------
    try:
        # pricing.py expects props path (not --season). Do NOT change your fetcher.
        _run("python scripts/pricing.py --props outputs/props_raw.csv")
        _assert_nonempty_csv("outputs/props_priced_clean.csv", "props_priced_clean")

    except Exception as e:
        print(f"[ENGINE] ❌ Pricing step failed: {e}")
        raise

    # -------------------------
    # STEP 5: Predictive models (optional; keep soft-fail while wiring)
    # -------------------------
    try:
        # Run model modules to avoid relative-import errors.
        _run(f"python -m scripts.models.monte_carlo --season {season} || true")
        _run(f"python -m scripts.models.bayes_hier --season {season} || true")
        _run(f"python -m scripts.models.markov --season {season} || true")
        _run(f"python -m scripts.models.ml_ensemble --season {season} || true")
    except Exception as e:
        print(f"[ENGINE] ⚠️ Predictor step failed (soft): {e}")

    # -------------------------
    # STEP 6: Export & archive
    # -------------------------
    try:
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

        print(f"[ENGINE] 🏁 Run completed → {runs_dir}")

    except Exception as e:
        print(f"[ENGINE] ⚠️ Export step failed: {e}")

    # -------------------------
    # STEP 7: Debug snapshot (optional)
    # -------------------------
    if debug:
        for path in [
            "data/team_form.csv",
            "data/player_form.csv",
            "data/metrics_ready.csv",
            "outputs/props_priced_clean.csv",
        ]:
            if os.path.exists(path):
                try:
                    import pandas as pd
                    df = pd.read_csv(path)
                    print(f"\n[DEBUG] {path}: {len(df)} rows, {len(df.columns)} cols")
                    print(df.head(5).to_string(index=False))
                except Exception:
                    print(f"[DEBUG] {path}: exists (size {os.path.getsize(path)} bytes)")

    print("[ENGINE] ✅ All done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="engine", description="Run the Sports Betting model pipeline.")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--date", type=str, default="")
    # accept both flags to match your workflow & package __main__
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
