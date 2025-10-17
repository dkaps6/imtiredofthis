# engine/engine.py
"""
Main orchestration pipeline for the Sports Betting Model.
Coordinates all data fetching, enrichment, pricing, and predictive modeling steps.

Version: 2025 NFL Season
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


# ------------------------------------------
# Utilities
# ------------------------------------------

def _run(cmd: str, check: bool = True) -> int:
    """Run a subprocess command and stream output with basic telemetry."""
    print(f"\n[ENGINE] ▶ {cmd}")
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    dur = time.time() - start
    status = "✅" if result.returncode == 0 else "❌"
    print(f"[ENGINE] {status} Completed in {dur:.2f}s (rc={result.returncode})")
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result.returncode


def _run_optional(cmd: str) -> int:
    """Run a helper command but do not raise if it exits non-zero."""
    try:
        return _run(cmd, check=False)
    except Exception as exc:  # pragma: no cover - extremely defensive
        print(f"[ENGINE] ⚠️ optional step wrapper failed unexpectedly: {exc}")
        return 1


def _safe_mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _write_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _csv_info(path: str, sample_cols: tuple[str, ...] = ()) -> Dict[str, Any]:
    """Return row/column counts plus a handful of sample column values."""
    info: Dict[str, Any] = {
        "path": path,
        "rows": 0,
        "cols": 0,
    }
    if not os.path.exists(path):
        info["status"] = "missing"
        return info

    try:
        import pandas as pd  # local import keeps startup light

        df = pd.read_csv(path)
        info["rows"] = len(df)
        info["cols"] = len(df.columns)
        if df.empty:
            info["status"] = "empty"
        else:
            info["status"] = "ok"
            for col in sample_cols:
                if col in df.columns:
                    non_null = df[col].dropna()
                    if not non_null.empty:
                        val = non_null.iloc[0]
                        if hasattr(val, "item"):
                            try:
                                val = val.item()
                            except Exception:  # pragma: no cover - defensive
                                val = val
                        info[col] = val
    except Exception as exc:  # pragma: no cover - defensive
        info["status"] = "error"
        info["error"] = str(exc)
    return info


def _assert_nonempty_csv(path: str, name: str, sample_cols: tuple[str, ...] = ()) -> Dict[str, Any]:
    """Fail fast if a required artifact is missing or empty and return metadata."""
    info = _csv_info(path, sample_cols)
    if info.get("status") != "ok":
        raise RuntimeError(
            f"{name} invalid: status={info.get('status')} details={info.get('error', '')} → {path}"
        )
    print(
        f"[ENGINE] • {name}: {info['rows']} rows ({info['cols']} cols)"
        + (
            f" source_season={info.get('source_season')}" if "source_season" in info else ""
        )
    )
    return info


def _append_run_summary(summary: Dict[str, Any]) -> None:
    """Persist a machine-readable summary for GitHub Actions or local review."""
    logs_dir = Path("logs")
    daily_dir = logs_dir / "daily"
    logs_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)

    summary_path = logs_dir / "actions_summary.log"
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, sort_keys=True) + "\n")

    run_path = daily_dir / f"run_{summary['run_id']}.json"
    with run_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


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
      3) Price props
      4) (Optional) Predictive models
      5) Export & snapshot
    """

    # Normalize books/bookmakers arg
    if not bookmakers and books:
        bookmakers = books

    start_wall = datetime.utcnow()
    start_ts = time.time()
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    runs_dir = os.path.join("runs", run_id)
    _safe_mkdir(runs_dir)
    print(f"[ENGINE] 🚀 Run {run_id} | season={season} date='{date}' books='{bookmakers}' markets='{markets}'")

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "season": season,
        "date": date,
        "bookmakers": bookmakers,
        "markets": markets,
        "started_at": start_wall.isoformat() + "Z",
        "steps": {},
        "status": "ok",
    }

    overall_error: Exception | None = None


    allow_fallback = (
        os.environ.get("ALLOW_NFL_FALLBACK", "").strip().lower() in {"1", "true", "yes"}
    )

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "season": season,
        "date": date,
        "bookmakers": bookmakers,
        "markets": markets,
        "started_at": start_wall.isoformat() + "Z",
        "steps": {},
        "status": "ok",
        "allow_fallback": allow_fallback,
    }

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "season": season,
        "date": date,
        "bookmakers": bookmakers,
        "markets": markets,
        "started_at": start_wall.isoformat() + "Z",
        "steps": {},
        "status": "ok",
    }

    overall_error: Exception | None = None

    try:
        # -------------------------
        # STEP 1: Fetch data (props + optional providers)
        # -------------------------
        try:
            fetch_parts = ["python", "scripts/fetch_props_oddsapi.py"]
            if bookmakers:
                fetch_parts += ["--bookmakers", bookmakers, "--books", bookmakers]
            if markets:
                fetch_parts += ["--markets", markets]
            fetch_cmd = " ".join(shlex.quote(p) for p in fetch_parts)
            _run(fetch_cmd)

            depth_cmds = [
                "python scripts/providers/espn_depth.py",
                "python scripts/providers/ourlads_depth.py",
            ]
            for cmd in depth_cmds:
                _run_optional(cmd)

            provider_scripts = [
                "scripts/providers/espn_pull.py",
                "scripts/providers/msf_pull.py",
                "scripts/providers/gsis_pull.py",
                "scripts/providers/apisports_pull.py",
                "scripts/providers/pfr_pull.py",
                "scripts/providers/injuries.py",
            ]
            for script in provider_scripts:
                env_cmd = f"SEASON={season} python {script}"
                _run_optional(env_cmd)
            _run("python scripts/providers/espn_depth.py || true")
            _run("python scripts/providers/ourlads_depth.py || true")
            _run("python scripts/providers/injuries.py || true")
            _run("python scripts/providers/pfr_team.py || true")

            props_info = _csv_info("outputs/props_raw.csv")
            games_info = _csv_info("outputs/game_lines.csv")
            summary["steps"]["fetch"] = {
                "status": "ok",
                "props_raw": {k: props_info.get(k) for k in ("rows", "cols", "status")},
                "game_lines": {k: games_info.get(k) for k in ("rows", "cols", "status")},
            }

        except Exception as e:
            print(f"[ENGINE] ⚠️ Data fetch step failed: {e}")
            summary["steps"]["fetch"] = {
                "status": "error",
                "error": str(e),
            }
            if summary["status"] == "ok":
                summary["status"] = "warning"

        # -------------------------
        # STEP 2: Build metrics (REQUIRED)
        # -------------------------
        try:
            print("\n[ENGINE] 🧮 Building TEAM metrics…")
            team_cmd = f"python scripts/make_team_form.py --season {season}"
            if allow_fallback:
                team_cmd += " --allow-fallback"
            _run(team_cmd)
            team_info = _assert_nonempty_csv(
                "data/team_form.csv",
                "team_form",
                sample_cols=("source_season",),
            )
            summary["steps"]["team_form"] = {k: team_info.get(k) for k in ("rows", "cols", "source_season")}
            summary["steps"]["team_form"]["status"] = "ok"

            _run("python scripts/enrich_team_form.py || true")  # ensures enrich step runs after team_form

            print("\n[ENGINE] 🧮 Building PLAYER metrics…")
            player_cmd = f"python scripts/make_player_form.py --season {season}"
            _run(player_cmd)
            if allow_fallback:
                player_cmd += " --allow-fallback"
            _run(player_cmd)
            _run(f"python scripts/make_player_form.py --season {season}")
            _run("python scripts/enrich_player_form.py || true")
            player_info = _assert_nonempty_csv(
                "data/player_form.csv",
                "player_form",
                sample_cols=("source_season",),
            )
            summary["steps"]["player_form"] = {k: player_info.get(k) for k in ("rows", "cols", "source_season")}
            summary["steps"]["player_form"]["status"] = "ok"

            print("\n[ENGINE] 🔗 Joining metrics for pricing…")
            _run(f"python scripts/make_metrics.py --season {season}")
            metrics_info = _assert_nonempty_csv("data/metrics_ready.csv", "metrics_ready")
            summary["steps"]["metrics_ready"] = {k: metrics_info.get(k) for k in ("rows", "cols")}
            summary["steps"]["metrics_ready"]["status"] = "ok"

        except Exception as e:
            print(f"[ENGINE] ❌ Metrics build failed: {e}")
            summary["status"] = "failed"
            summary["steps"].setdefault("metrics", {})
            summary["steps"]["metrics"]["status"] = "error"
            summary["steps"]["metrics"]["error"] = str(e)
            summary.setdefault("error", str(e))
            overall_error = e
            raise  # hard fail; pricing without metrics is pointless

        # -------------------------
        # STEP 3: Price props (REQUIRED)
        # -------------------------
        try:
            _run("python scripts/pricing.py --props outputs/props_raw.csv")
            priced_info = _assert_nonempty_csv("outputs/props_priced_clean.csv", "props_priced_clean")
            summary["steps"]["pricing"] = {k: priced_info.get(k) for k in ("rows", "cols")}
            summary["steps"]["pricing"]["status"] = "ok"

        except Exception as e:
            print(f"[ENGINE] ❌ Pricing step failed: {e}")
            summary["status"] = "failed"
            summary["steps"]["pricing"] = {"status": "error", "error": str(e)}
            summary.setdefault("error", str(e))
            overall_error = e
            raise

        # -------------------------
        # STEP 4: Predictive models (optional; keep soft-fail while wiring)
        # -------------------------
        try:
            _run(f"python -m scripts.models.monte_carlo --season {season} || true")
            _run(f"python -m scripts.models.bayes_hier --season {season} || true")
            _run(f"python -m scripts.models.markov --season {season} || true")
            _run(f"python -m scripts.models.ml_ensemble --season {season} || true")
            summary["steps"]["predictors"] = {"status": "ok"}
        except Exception as e:
            print(f"[ENGINE] ⚠️ Predictor step failed (soft): {e}")
            summary["steps"]["predictors"] = {"status": "warning", "error": str(e)}
            if summary["status"] == "ok":
                summary["status"] = "warning"

        # -------------------------
        # STEP 5: Export & archive
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
            summary["steps"]["export"] = {"status": "ok", "runs_dir": runs_dir}

        except Exception as e:
            print(f"[ENGINE] ⚠️ Export step failed: {e}")
            summary["steps"]["export"] = {"status": "warning", "error": str(e)}
            if summary["status"] == "ok":
                summary["status"] = "warning"

        # -------------------------
        # STEP 6: Debug snapshot (optional)
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

    except Exception as exc:
        overall_error = exc
        summary.setdefault("error", str(exc))
        raise

    finally:
        summary["finished_at"] = datetime.utcnow().isoformat() + "Z"
        summary["duration_s"] = round(time.time() - start_ts, 2)
        if overall_error is not None and summary.get("status") != "failed":
            summary["status"] = "failed"
        _append_run_summary(summary)
        if overall_error is None:
            print("[ENGINE] ✅ All done!")
        else:
            print("[ENGINE] ❌ Run aborted.")


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
