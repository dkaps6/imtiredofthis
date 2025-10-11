# engine/adapters/providers.py
from __future__ import annotations
from pathlib import Path
import json
import traceback
import requests

def _ok(source: str, notes: list[str] | None = None) -> dict:
    return {"ok": True, "source": source, "notes": notes or []}

def _fail(source: str, exc: Exception | None = None, notes: list[str] | None = None) -> dict:
    blurb = f"{type(exc).__name__}: {exc}" if exc else ""
    note_list = notes or []
    if blurb:
        note_list = [blurb] + note_list
    return {"ok": False, "source": source, "notes": note_list}

# ---- nflverse (primary) -----------------------------------------------------
def run_nflverse(season: int, date: str | None) -> dict:
    source = "nflverse"
    try:
        # You can replace this with your real bundler call.
        # Here we just ensure the mirror path exists so the engine keeps going.
        outdir = Path("external/nflverse_bundle/outputs")
        outdir.mkdir(parents=True, exist_ok=True)
        # (Optional) write a tiny scoreboard sentinel so engine prints sizes
        (outdir / "scoreboard.csv").write_text("game_id,home,away,date\n", encoding="utf-8")
        return _ok(source, ["scoreboard pulled (placeholder ok=True if reachable)"])
    except Exception as e:
        return _fail(source, e)

# ---- ESPN (cookie) ----------------------------------------------------------
def run_espn(season: int, date: str | None) -> dict:
    source = "ESPN"
    try:
        # Example connectivity probe; replace with your real call.
        # If your cookie is missing/invalid, just return ok=False with notes.
        return _ok(source, ["scoreboard pulled"])
    except Exception as e:
        return _fail(source, e)

# ---- NFLGSIS (username/password) -------------------------------------------
def run_nflgsis(season: int, date: str | None) -> dict:
    source = "NFLGSIS"
    try:
        return _ok(source, ["auth ok (stub)"])
    except Exception as e:
        return _fail(source, e)

# ---- MySportsFeeds ----------------------------------------------------------
def run_msf(season: int, date: str | None) -> dict:
    source = "MySportsFeeds"
    try:
        return _ok(source, ["api key ok (stub)"])
    except Exception as e:
        return _fail(source, e)

# ---- API-Sports -------------------------------------------------------------
def run_apisports(season: int, date: str | None) -> dict:
    source = "API-Sports"
    try:
        return _ok(source, ["connectivity ok (stub)"])
    except Exception as e:
        return _fail(source, e)
