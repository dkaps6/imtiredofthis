# engine/engine.py
from __future__ import annotations

import os, sys, shlex, subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd

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

def _run(cmd: str):
    print(f"[engine] ▶ {cmd}", flush=True)
    rc = subprocess.call(shlex.split(cmd))
    if rc != 0:
        print(f"[engine] ✖ step failed (exit {rc})", flush=True)
        sys.exit(rc)

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

    # 1) upstream providers (best-effort)
    try:
        _provider_chain(int(season), date or None)
    except Exception as e:
        print(f"[engine] provider chain error (non-fatal): {e}")

    # 2) builders — team & player form first
    _run(f"python scripts/make_team_form.py --season {season}")
    print(f"[engine]   data/team_form.csv → {_size('data/team_form.csv')}")
    _run(f"python scripts/make_player_form.py --season {season}")
    print(f"[engine]   data/player_form.csv → {_size('data/player_form.csv')}")

    # 3) odds props — v4 requires one market per request; also write odds_game.csv
    b = ",".join(books or ["draftkings","fanduel","betmgm","caesars"])

    _default_markets = [
        "player_pass_yds",
        "player_reception_yds",    # receiving yards (canonical key)
        "player_rush_yds",
        "player_receptions",
        "player_rush_reception_yds",
        "player_anytime_td",
    ]
    markets_to_pull = [m.strip() for m in (markets or _default_markets) if m.strip()]

    # ensure game lines exist once
    _run(
        "python scripts/fetch_props_oddsapi.py "
        f"--books {b} --markets h2h,spreads,totals "
        f"--date {date or ''} "
        "--out outputs/_tmp_props/_game.csv --out_game outputs/odds_game.csv"
    )

    # fetch ALL player markets in one call → writes outputs/props_raw.csv (+ props_raw_wide.csv)
    all_mk = ",".join(markets_to_pull)
    _run(
        "python scripts/fetch_props_oddsapi.py "
        f"--books {b} --markets {all_mk} "
        f"--date {date or ''} "
        "--out outputs/props_raw.csv --out_game outputs/odds_game.csv"
    )

    # 3.5) build metrics_ready (features for pricing)
    _run("python scripts/make_metrics.py")
    print(
        f"[engine]   data/metrics_ready.csv → "
        f"{os.path.getsize('data/metrics_ready.csv') if Path('data/metrics_ready.csv').exists() else 'MISSING'}"
    )

    # 4) pricing + predictors
    if not os.path.exists("outputs/props_raw.csv") or os.stat("outputs/props_raw.csv").st_size == 0:
        print("[engine] ERROR: outputs/props_raw.csv is missing or empty – skipping pricing")
        sys.exit(2)
    _run("python scripts/pricing.py --props outputs/props_raw.csv")
    _run(f"python -m scripts.models.run_predictors --season {season}")
    print(f"[engine]   outputs/master_model_predictions.csv → {_size('outputs/master_model_predictions.csv')}")

    # 5) export Excel + diagnostics
    _run("python scripts/export_excel.py")
    _dump_diag()

    rid = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    print(f"[engine] ✅ complete (run_id={rid})")
    return 0

def cli_main() -> int:
    """CLI entrypoint used by `python -m engine`."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True)
    ap.add_argument("--date", default="")
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--markets", default="")
    args = ap.parse_args()

    return run_pipeline(
        season=args.season,
        date=args.date,
        books=[b.strip() for b in args.books.split(",") if b.strip()],
        markets=[m.strip() for m in args.markets.split(",") if m.strip()] or None,
    )

if __name__ == "__main__":
    sys.exit(cli_main())
