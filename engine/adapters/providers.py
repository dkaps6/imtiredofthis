from __future__ import annotations
from pathlib import Path
import traceback
import json
import os
import shlex
import subprocess
import pandas as pd
from typing import List, Dict, Any

# ---------------- helpers (added) ----------------

def _run(cmd: str) -> int:
    """Run a command without raising; return exit code."""
    try:
        return subprocess.call(shlex.split(cmd))
    except Exception:
        return 1

def _rows(p: str | Path) -> int:
    """Count rows if CSV exists; 0 otherwise (fast + safe)."""
    p = Path(p)
    if not p.exists() or p.stat().st_size < 8:
        return 0
    try:
        df = pd.read_csv(p)
        return len(df.index)
    except Exception:
        return 0

def _ok(source: str, notes: list[str] | None = None) -> dict:
    return {"ok": True, "source": source, "notes": notes or []}

def _fail(source: str, exc: Exception | None = None, notes: list[str] | None = None) -> dict:
    blurb = f"{type(exc).__name__}: {exc}" if exc else ""
    n = notes or []
    if blurb:
        n = [blurb] + n
    return {"ok": False, "source": source, "notes": n}

# ---------------- adapters ----------------

# Primary: nflverse/nflreadr mirrors (unchanged stub)
def run_nflverse(season: int, date: str | None) -> dict:
    source = "nflverse"
    try:
        outdir = Path("external/nflverse_bundle/outputs")
        outdir.mkdir(parents=True, exist_ok=True)
        # NOTE: Replace these sentinels with your real fetch if present.
        for name in ("pbp.csv", "schedules.csv", "scoreboard.csv"):
            p = outdir / name
            if not p.exists():
                p.write_text("")
        return _ok(source, [f"created/confirmed mirrors in {outdir}"])
    except Exception as e:
        return _fail(source, e, [traceback.format_exc()])

# ESPN (depth charts + optional normalized player sheet)
def run_espn(season: int, date: str | None) -> dict:
    source = "ESPN"
    notes: List[str] = []
    try:
        # 1) Depth chart (roles/positions)
        if Path("scripts/providers/espn_depth.py").exists():
            rc = _run("python scripts/providers/espn_depth.py")
            if rc != 0:
                notes.append(f"espn_depth rc={rc}")
        else:
            notes.append("scripts/providers/espn_depth.py missing")

        # 2) Optional normalizer to data/espn_player.csv (if you maintain it)
        if Path("scripts/providers/espn_pull.py").exists():
            rc = _run("python scripts/providers/espn_pull.py")
            if rc != 0:
                notes.append(f"espn_pull rc={rc}")
        else:
            notes.append("espn_pull.py missing (depth-only)")

        ok = (_rows("data/depth_chart_espn.csv") > 0) or (_rows("data/espn_player.csv") > 0)
        if not ok:
            notes.append("no rows in depth_chart_espn.csv or espn_player.csv")
        return _ok(source, notes) if ok else _fail(source, None, notes)
    except Exception as e:
        notes.append(traceback.format_exc())
        return _fail(source, e, notes)

# NFLGSIS via nfl_data_py → data/gsis_player.csv
def run_nflgsis(season: int, date: str | None) -> dict:
    source = "NFLGSIS"
    notes: List[str] = []
    try:
        if Path("scripts/providers/gsis_pull.py").exists():
            rc = _run("python scripts/providers/gsis_pull.py")
            if rc != 0:
                notes.append(f"gsis_pull rc={rc}")
        else:
            notes.append("scripts/providers/gsis_pull.py missing")

        ok = _rows("data/gsis_player.csv") > 0
        if not ok:
            notes.append("no rows in gsis_player.csv")
        return _ok(source, notes) if ok else _fail(source, None, notes)
    except Exception as e:
        notes.append(traceback.format_exc())
        return _fail(source, e, notes)

# MySportsFeeds → data/msf_player.csv (requires MSF_API_KEY)
def run_msf(season: int, date: str | None) -> dict:
    source = "MySportsFeeds"
    notes: List[str] = []
    try:
        if not os.getenv("MSF_API_KEY"):
            notes.append("MSF_API_KEY not set")
        if Path("scripts/providers/msf_pull.py").exists():
            rc = _run("python scripts/providers/msf_pull.py")
            if rc != 0:
                notes.append(f"msf_pull rc={rc}")
        else:
            notes.append("scripts/providers/msf_pull.py missing")

        ok = _rows("data/msf_player.csv") > 0
        if not ok:
            notes.append("no rows in msf_player.csv")
        return _ok(source, notes) if ok else _fail(source, None, notes)
    except Exception as e:
        notes.append(traceback.format_exc())
        return _fail(source, e, notes)

# API-Sports → data/apisports_player.csv (requires APISPORTS_KEY)
def run_apisports(season: int, date: str | None) -> dict:
    source = "API-Sports"
    notes: List[str] = []
    try:
        if not os.getenv("APISPORTS_KEY"):
            notes.append("APISPORTS_KEY not set")
        if Path("scripts/providers/apisports_pull.py").exists():
            rc = _run("python scripts/providers/apisports_pull.py")
            if rc != 0:
                notes.append(f"apisports_pull rc={rc}")
        else:
            notes.append("scripts/providers/apisports_pull.py missing")

        ok = _rows("data/apisports_player.csv") > 0
        if not ok:
            notes.append("no rows in apisports_player.csv")
        return _ok(source, notes) if ok else _fail(source, None, notes)
    except Exception as e:
        notes.append(traceback.format_exc())
        return _fail(source, e, notes)
