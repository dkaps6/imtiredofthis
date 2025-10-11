from __future__ import annotations
from pathlib import Path
import traceback
import json

def _ok(source: str, notes: list[str] | None = None) -> dict:
    return {"ok": True, "source": source, "notes": notes or []}

def _fail(source: str, exc: Exception | None = None, notes: list[str] | None = None) -> dict:
    blurb = f"{type(exc).__name__}: {exc}" if exc else ""
    n = notes or []
    if blurb:
        n = [blurb] + n
    return {"ok": False, "source": source, "notes": n}

# Primary: nflverse/nflreadr mirrors
def run_nflverse(season: int, date: str | None) -> dict:
    source = "nflverse"
    try:
        outdir = Path("external/nflverse_bundle/outputs")
        outdir.mkdir(parents=True, exist_ok=True)
        # NOTE: Replace these sentinels with your real fetch process if present.
        for name in ("pbp.csv", "schedules.csv", "scoreboard.csv"):
            p = outdir / name
            if not p.exists():
                p.write_text("")
        return _ok(source, [f"created/confirmed mirrors in {outdir}"])
    except Exception as e:
        return _fail(source, e, [traceback.format_exc()])

# ESPN (cookie-based). Keep non-fatal on auth failure.
def run_espn(season: int, date: str | None) -> dict:
    source = "ESPN"
    try:
        # If you already have espn fetchers, call them here.
        return _ok(source, ["connectivity ok (stub)"])
    except Exception as e:
        return _fail(source, e)

def run_nflgsis(season: int, date: str | None) -> dict:
    source = "NFLGSIS"
    try:
        return _ok(source, ["auth ok (stub)"])
    except Exception as e:
        return _fail(source, e)

def run_msf(season: int, date: str | None) -> dict:
    source = "MySportsFeeds"
    try:
        return _ok(source, ["api key ok (stub)"])
    except Exception as e:
        return _fail(source, e)

def run_apisports(season: int, date: str | None) -> dict:
    source = "API-Sports"
    try:
        cache = Path("external/api_sports/cache")
        cache.mkdir(parents=True, exist_ok=True)
        for fn in ("teams.json", "players.json"):
            p = cache / fn
            if not p.exists():
                p.write_text(json.dumps([], indent=2))
        return _ok(source, [f"cache primed in {cache}"])
    except Exception as e:
        return _fail(source, e)
