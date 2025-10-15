# engine/engine.py
from __future__ import annotations

import os, sys, shlex, subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd

# NEW: snapshot helpers
import shutil, time, traceback  # <— NEW

# --- config wiring (optional; safe fallback if missing) ---
try:
    from scripts.config import (
        FILES, DIR, ensure_dirs,
        books_from_env, markets_from_env, ODDS
    )
except Exception:
    FILES = {
        "props_raw": "outputs/props_raw.csv",
        "odds_game": "outputs/odds_game.csv",
    }
    def ensure_dirs():  # no-op if config not present
        for d in ("data", "outputs", "outputs/metrics", "logs", "outputs/_tmp_props"):
            Path(d).mkdir(parents=True, exist_ok=True)
    def books_from_env(): return ["draftkings","fanduel","betmgm","caesars"]
    def markets_from_env(): return []
    ODDS = {"region": "us"}

# Providers (soft fallback if module not present)
try:
    from engine.adapters.providers import (
        run_nflverse, run_espn, run_nflgsis, run_msf, run_apisports
    )
except Exception:
    def _stub(name):
        def _run(season: int, date: str | None):
            return {"ok": False, "source": name, "notes": [f"{name} adapter missing"]}
        return _run
    run_nflverse   = _stub("nflverse")
    run_espn       = _stub("ESPN")
    run_nflgsis    = _stub("NFLGSIS")
    run_msf        = _stub("MySportsFeeds")
    run_apisports  = _stub("API-Sports")

# NEW: per-run snapshot directory & helpers
RUN_ID = os.getenv("RUN_ID", datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
SNAP_DIR = Path("runs") / RUN_ID
SNAP_DIR.mkdir(parents=True, exist_ok=True)

def _safe_exists(p: Path) -> bool:
    try:
        return p.exists() and (p.stat().st_size > 0)
    except Exception:
        return False

def snapshot(paths, label=None):
    """
    Copy files that exist into runs/<RUN_ID>/<label>_<epoch>/
    so you can inspect outputs even if later steps fail.
    """
    stamp = f"{int(time.time())}"
    tag = f"{label}_" if label else ""
    outdir = SNAP_DIR / f"{tag}{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    wrote = []
    for s in (paths or []):
        p = Path(s)
        if _safe_exists(p):
            shutil.copy2(p, outdir / p.name)
            wrote.append(p.name)
    (outdir / "_snapshot.txt").write_text(
        f"Snapshot {label or ''} at {datetime.utcnow().isoformat()}Z\n"
        f"Files: {', '.join(wrote) if wrote else '(none)'}\n",
        encoding="utf-8"
    )
    print(f"[engine] snapshot → {outdir} ({len(wrote)} files)")

def finalize_run():
    """Zip runs/<RUN_ID> so CI can upload artifacts even on failure."""
    try:
        _dump_diag()  # keep your existing heads/sizes
        bundle = SNAP_DIR.with_suffix(".zip")
        if bundle.exists():
            bundle.unlink()
        shutil.make_archive(str(SNAP_DIR), "zip", root_dir=SNAP_DIR)
        print(f"[engine] packaged snapshots → {bundle}")
    except Exception:
        traceback.print_exc()

def _run(cmd: str, label: str | None = None, snap_after: list[str] | None = None):
    """Run a shell step; snapshot selected files; raise on failure (no sys.exit)."""
    print(f"[engine] ▶ {cmd}", flush=True)
    rc = subprocess.call(shlex.split(cmd))
    # snapshot even if the step fails (best-effort)
    if snap_after:
        snapshot(snap_after, label or "step")
    if rc != 0:
        print(f"[engine] ✖ step failed (exit {rc})", flush=True)
        # raise instead of sys.exit, so finalize_run() still packages artifacts
        raise RuntimeError(f"step failed: {label or cmd} (exit {rc})")
    return rc

def _size(p: str) -> str:
    fp = Path(p)
    return f"{fp.stat().st_size}B" if fp.exists() else "MISSING"

def _dump_diag():
    Path("logs").mkdir(exist_ok=True)
    targets = [
        "data/team_form.csv","data/player_form.csv","data/metrics_ready.csv",
        "outputs/props_raw.csv","outputs/odds_game.csv",
        "outputs/master_model_predictions.csv","outputs/sgp_candidates.csv",
    ]
    with open("logs/run_diagnostics.txt","w") as f:
        for t in targets:
            f.write(f"{t}: {_size(t)}\n")
    for t in targets:
        try:
            df = pd.read_csv(t)
            df.head(20).to_csv(f"logs/head__{Path(t).name}", index=False)
        except Exception:
            pass
    print("[engine] wrote logs/run_diagnostics.txt")

def _provider_chain(season: int, date: str | None):
    print("[engine] Provider order: nflverse → ESPN → NFLGSIS → MySportsFeeds → API-Sports")
    for runner in (run_nflverse, run_espn, run_nflgsis, run_msf, run_apisports):
        try:
            res = runner(season, date)
        except Exception as e:
            res = {"ok": False, "source": getattr(runner, "__name__", "unknown"), "notes": [str(e)]}
        print(f"[engine] provider={res.get('source')} ok={res.get('ok')} notes={'; '.join(res.get('notes', []))}")
        if res.get("ok"):
            os.environ["PROVIDER_USED"] = res.get("source", "")
            return
    print("[engine] ⚠ no external provider succeeded; will rely on builders")

def run_pipeline(season: str, date: str, books: list[str] | None, markets: list[str] | None) -> int:
    # --- setup dirs & keys status ---
    ensure_dirs()
    for d in ("data", "outputs", "outputs/metrics", "logs", "outputs/_tmp_props"):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(
        f"[engine] keys: ODDS_API_KEY={'set' if os.getenv('ODDS_API_KEY') else 'missing'} "
        f"ESPN_COOKIE={'set' if os.getenv('ESPN_COOKIE') else 'missing'}"
    )

    # env flag to enable strict checks that fail-fast on stub data
    STRICT = os.getenv("NFL_FORM_STRICT") == "1"  # NEW

    try:
        # 1) upstream providers (best-effort)
        try:
            _provider_chain(int(season), date or None)
        except Exception as e:
            print(f"[engine] provider chain error (non-fatal): {e}")

        # 2) builders — team & player form first
        _run(f"python scripts/make_team_form.py --season {season}",
             label="team_form", snap_after=["data/team_form.csv"])
        print(f"[engine]   data/team_form.csv → {_size('data/team_form.csv')}")
        if STRICT:
            try:
                tf = pd.read_csv("data/team_form.csv")
                if tf["team"].nunique() < 8:
                    snapshot(["data/team_form.csv"], "team_form_stub")
                    raise RuntimeError("team_form looks like a stub (too few teams). Check requirements install and nfl_data_py import.")
            except Exception as _e:
                raise

        _run(f"python scripts/make_player_form.py --season {season}",
             label="player_form", snap_after=["data/player_form.csv"])
        print(f"[engine]   data/player_form.csv → {_size('data/player_form.csv')}")
        if STRICT:
            try:
                pf = pd.read_csv("data/player_form.csv")
                if pf["team"].nunique() < 8 or len(pf) < 50:
                    snapshot(["data/player_form.csv"], "player_form_stub")
                    raise RuntimeError("player_form looks like a stub (too few teams/players). Check requirements install and nfl_data_py import.")
            except Exception as _e:
                raise

        # 3) odds props — v4 requires one market per request; also write odds_game.csv

        # Respect explicit empty list (no bookmaker filter). Only default if None.
        if books is None:
            b = "draftkings,fanduel,betmgm,caesars"
        else:
            b = ",".join(books)

        _default_markets = [
            "player_pass_yds",
            "player_reception_yds",    # receiving yards (canonical key)
            "player_receiving_yards",  # alias (safety)
            "player_rush_yds",
            "player_receptions",
            "player_rush_reception_yds",
            "player_rush_and_receive_yards",
            "player_anytime_td",
        ]
        markets_to_pull = [m.strip() for m in (markets or _default_markets) if m.strip()]
        markets_to_pull = list(dict.fromkeys(markets_to_pull))

        # ensure game lines exist once
        game_cmd = "python scripts/fetch_props_oddsapi.py "
        if b:
            game_cmd += f"--bookmakers {b} "
        game_cmd += (
            f"--markets h2h,spreads,totals "
            f"--date {date or ''} "
            "--out outputs/_tmp_props/_game.csv --out_game outputs/odds_game.csv"
        )
        _run(game_cmd, label="odds_game", snap_after=["outputs/odds_game.csv", "outputs/_tmp_props/_game.csv"])

        # fetch ALL player markets in one call → writes outputs/props_raw.csv (+ props_raw_wide.csv)
        all_mk = ",".join(markets_to_pull)
        props_cmd = "python scripts/fetch_props_oddsapi.py "
        if b:
            props_cmd += f"--bookmakers {b} "
        props_cmd += (
            f"--markets {all_mk} "
            f"--date {date or ''} "
            "--out outputs/props_raw.csv --out_game outputs/odds_game.csv"
        )
        _run(props_cmd, label="props_raw", snap_after=["outputs/props_raw.csv"])

        # 3.5) build metrics_ready (features for pricing)
        _run("python scripts/make_metrics.py",
             label="metrics", snap_after=["data/metrics_ready.csv", "outputs/metrics/metrics_ready.csv"])
        print(
            f"[engine]   data/metrics_ready.csv → "
            f"{os.path.getsize('data/metrics_ready.csv') if Path('data/metrics_ready.csv').exists() else 'MISSING'}"
        )

        # 4) pricing + predictors
        if not os.path.exists("outputs/props_raw.csv") or os.stat("outputs/props_raw.csv").st_size == 0:
            snapshot(["outputs/props_raw.csv"], "props_check")
            raise RuntimeError("[engine] outputs/props_raw.csv is missing or empty – skipping pricing")

        _run("python scripts/pricing.py --props outputs/props_raw.csv",
             label="pricing", snap_after=["outputs/props_priced_clean.csv"])

        _run(f"python -m scripts.models.run_predictors --season {season}",
             label="predictors", snap_after=[
                 "outputs/master_model_predictions.csv",
                 "outputs/sgp_candidates.csv",
                 "logs/master_model_predictions.csv",
                 "logs/summary.json"
             ])
        print(f"[engine]   outputs/master_model_predictions.csv → {_size('outputs/master_model_predictions.csv')}")

        # 5) export Excel + diagnostics
        _run("python scripts/export_excel.py",
             label="export", snap_after=["outputs/model_report.xlsx"])
        _dump_diag()
        snapshot([
            "logs/run_diagnostics.txt",
            "outputs/master_model_predictions.csv",
            "outputs/sgp_candidates.csv",
            "data/metrics_ready.csv",
            "outputs/props_raw.csv",
            "outputs/odds_game.csv"
        ], "final")

        rid = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        print(f"[engine] ✅ complete (run_id={rid})")
        return 0

    except Exception as e:
        print(f"[engine] ERROR: {e}", file=sys.stderr)
        return 1
    finally:
        finalize_run()

def cli_main() -> int:
    """CLI entrypoint used by `python -m engine`."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True)
    ap.add_argument("--date", default="")
    # Accept both names; map to same var
    ap.add_argument("--books", "--bookmakers", dest="books",
                    default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--markets", default="")
    args = ap.parse_args()

    return run_pipeline(
        season=args.season,
        date=args.date,
        books=[b.strip() for b in args.books.split(",") if b.strip()],  # [] if user passes --bookmakers=""
        markets=[m.strip() for m in args.markets.split(",") if m.strip()] or None,
    )

if __name__ == "__main__":
    sys.exit(cli_main())
