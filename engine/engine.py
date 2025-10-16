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
    # Fallbacks if scripts.config is missing in some workflows
    def ensure_dirs():
        for d in ("data", "outputs", "outputs/metrics", "logs", "outputs/_tmp_props"):
            Path(d).mkdir(parents=True, exist_ok=True)
    def books_from_env(): return ["draftkings","fanduel","betmgm","caesars"]
    def markets_from_env(): return []
    ODDS = {"region": "us"}

# Additional provider pulls (depth charts & PFR enrich)
try:
    from scripts.providers.pfr_pull import main as _pfr_pull_main
except Exception:
    _pfr_pull_main = None
try:
    from scripts.providers.espn_depth import main as _espn_depth_main
except Exception:
    _espn_depth_main = None
try:
    from scripts.providers.ourlads_depth import main as _ourlads_depth_main
except Exception:
    _ourlads_depth_main = None

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
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False

def _size(p: str) -> str:
    try:
        n = Path(p).stat().st_size
        if n < 1024: return f"{n} B"
        if n < 1024**2: return f"{n/1024:.1f} KB"
        if n < 1024**3: return f"{n/1024**2:.1f} MB"
        return f"{n/1024**3:.1f} GB"
    except Exception:
        return "n/a"

def finalize_run():
    try:
        snapshot([
            "outputs/props_raw.csv", "outputs/props_priced_clean.csv",
            "outputs/odds_game.csv", "data/player_form.csv", "data/team_form.csv",
            "logs/run_diagnostics.txt"
        ], "end")
    except Exception:
        pass

def _run(cmd: str, *, label: str, snap_after: list[str] | None = None) -> int:
    """Run a shell command; snapshot selected paths if provided."""
    print(f"[engine] ▶ {cmd}")
    rc = subprocess.call(shlex.split(cmd))
    if rc != 0:
        snapshot(snap_after or [], f"{label}_failed")
        raise RuntimeError(f"step '{label}' failed rc={rc}")
    if snap_after:
        snapshot(snap_after, label)
    return rc

def snapshot(paths: list[str] | None, label: str = ""):
    """
    Copy specified files into runs/<RUN_ID>/<timestamp>_<label>/ for artifact upload.
    Only copies files that exist and are non-empty.
    Also writes a small _snapshot.txt with the file list + timestamp,
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
    print(f"[engine] snap[{label}]: {', '.join(wrote) if wrote else '(none)'}")

def _dump_diag():
    try:
        (Path("logs") / "run_diagnostics.txt").write_text(
            f"props_raw.csv: {_size('outputs/props_raw.csv')}\n"
            f"props_priced_clean.csv: {_size('outputs/props_priced_clean.csv')}\n"
            f"player_form.csv: {_size('data/player_form.csv')}\n"
            f"team_form.csv: {_size('data/team_form.csv')}\n",
            encoding="utf-8",
        )
        for t in ("outputs/props_raw.csv", "outputs/props_priced_clean.csv"):
            try:
                df = pd.read_csv(t)
                df.head(20).to_csv(f"logs/head__{Path(t).name}", index=False)
            except Exception:
                pass
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

    STRICT = False  # keep permissive; we snapshot and proceed

    try:
        # Pre-materialize enrichers that are independent of the odds/providers
        try:
            if _pfr_pull_main:
                _pfr_pull_main(int(season))
        except Exception as e:
            print(f"[engine] PFR enrich failed (non-fatal): {type(e).__name__}: {e}")
        try:
            if _espn_depth_main:
                _espn_depth_main()
        except Exception as e:
            print(f"[engine] ESPN depth chart fetch failed (non-fatal): {type(e).__name__}: {e}")
        try:
            if _ourlads_depth_main:
                _ourlads_depth_main()
        except Exception as e:
            print(f"[engine] OurLads depth chart fetch failed (non-fatal): {type(e).__name__}: {e}")

        _provider_chain(int(season), date or None)

        # 1) team form
        _run(f"python scripts/make_team_form.py --season {season}",
             label="team_form", snap_after=["data/team_form.csv"])
        print(f"[engine]   data/team_form.csv → {_size('data/team_form.csv')}")
        if STRICT:
            try:
                tf = pd.read_csv("data/team_form.csv")
                if tf["team"].nunique() < 28:
                    raise RuntimeError("too few teams; team form likely empty")
            except Exception:
                raise

        # 2) player form
        _run(f"python scripts/make_player_form.py --season {season}",
             label="player_form", snap_after=["data/player_form.csv"])
        print(f"[engine]   data/player_form.csv → {_size('data/player_form.csv')}")

        # --- NEW: non-invasive enrichers (post-build; no changes to your builders) ---
        try:
            _run("python scripts/enrich_team_form.py",
                 label="enrich_team", snap_after=["data/team_form.csv"])
            _run("python scripts/enrich_player_form.py",
                 label="enrich_player", snap_after=["data/player_form.csv"])
        except Exception as e:
            print(f"[engine] enrichers skipped: {type(e).__name__}: {e}")

        # 3) merge metrics  (prefer your make_metrics.py; fall back to *_ready if present)
        metrics_script = None
        if Path("scripts/make_metrics.py").exists():
            metrics_script = "scripts/make_metrics.py"
        elif Path("scripts/make_metrics_ready.py").exists():
            metrics_script = "scripts/make_metrics_ready.py"

        if metrics_script:
            _run(f"python {metrics_script}",
                 label="metrics_ready", snap_after=["data/metrics_ready.csv"])
        else:
            print("[engine] skip metrics: no make_metrics script found")

        # 4) props (oddsapi)
        # Check if caller wants an explicit bookmakers list.
        # Forward the exact user intent to the fetcher:
        # - books == []  → user passed --bookmakers "" → pass --bookmakers "" (no filter)
        # - books is None → no user flag → let fetcher use its own default
        # - books list (non-empty) → pass the same list
        if books is None:
            b = None  # do not include the flag; fetcher will use its default
        else:
            b = ",".join(books)  # may be "", which we will still forward

        _default_markets = [
            "player_pass_yds",
            "player_reception_yds",
            "player_rush_yds",
            "player_receptions",
            "player_rush_reception_yds",
            "player_anytime_td",
        ]
        markets_to_pull = markets or _default_markets
        all_mk = ",".join(markets_to_pull)
        props_cmd = "python scripts/fetch_props_oddsapi.py "
        if b is not None:
            props_cmd += f'--bookmakers "{b}" '
        props_cmd += (
            f"--markets {all_mk} "
            f"--date {date or ''} "
            "--out outputs/props_raw.csv --out_game outputs/odds_game.csv"
        )
        _run(props_cmd, label="props_raw", snap_after=["outputs/props_raw.csv"])
        # Copy props_raw_wide.csv to _tmp_props for downstream tools expecting that path
        try:
            pd.read_csv("outputs/props_raw.csv").to_csv("outputs/_tmp_props/props_raw.csv", index=False)
        except Exception:
            pass

        # 4.5) ensure props exist before pricing
        if not _safe_exists(Path("outputs/props_raw.csv")):
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
    ap.add_argument("--season", default=os.getenv("SEASON") or "2025")
    ap.add_argument("--date", default=os.getenv("SLATE_DATE") or "")
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
