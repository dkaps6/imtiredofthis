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

def run_nflverse(season: int, date: str | None) -> dict:
    """
    Fetch core league data via nflreadpy (preferred) with nfl_data_py fallback.
    Writes:
      - data/pbp_<season>.csv
      - data/participation_<season>.csv   (if nflreadpy provides it)
      - data/schedules_<season>.csv
    Returns ok=True only if at least one file has rows.
    """
    source = "nflverse"
    notes: list[str] = []
    from pathlib import Path
    import pandas as pd

    def _safe_to_csv(df, path):
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(path, index=False)
                return True
        except Exception as e:
            notes.append(f"write {path}: {type(e).__name__}: {e}")
        return False

    wrote_any = False
    pbp_rows = 0
    sch_rows = 0
    part_rows = 0

    # --- Try nflreadpy first (preferred mirror) ---
    try:
        import nflreadpy as nfr  # modern python mirror of nflfastR
        # Play-by-play
        try:
            pf = nfr.load_pbp([season])        # returns a Polars DataFrame
            pbp_df = pf.to_pandas()
            if _safe_to_csv(pbp_df, f"data/pbp_{season}.csv"):
                pbp_rows = len(pbp_df)
                wrote_any = True
                notes.append(f"nflreadpy pbp rows={pbp_rows} (season {season})")
        except Exception as e:
            notes.append(f"nflreadpy pbp: {type(e).__name__}: {e}")

        # Participation (routes/snap participants) — not always available immediately
        try:
            part = nfr.load_participation([season])
            part_df = part.to_pandas()
            if _safe_to_csv(part_df, f"data/participation_{season}.csv"):
                part_rows = len(part_df)
                wrote_any = True
                notes.append(
                    f"nflreadpy participation rows={part_rows} (season {season})"
                )
        except Exception as e:
            notes.append(f"nflreadpy participation: {type(e).__name__}: {e}")

        # Schedules / games
        try:
            sch = nfr.load_schedules([season])
            sch_df = sch.to_pandas()
            if _safe_to_csv(sch_df, f"data/schedules_{season}.csv"):
                sch_rows = len(sch_df)
                wrote_any = True
                notes.append(f"nflreadpy schedules rows={sch_rows} (season {season})")
        except Exception as e:
            notes.append(f"nflreadpy schedules: {type(e).__name__}: {e}")

    except Exception as e:
        if "original_mlq" in str(e):
            notes.append(
                "nflreadpy not available: nfl_data_py missing `original_mlq` (upgrade to >=0.3.4)"
            )
        else:
            notes.append(f"nflreadpy not available: {type(e).__name__}: {e}")

    # --- Fallback to nfl_data_py if needed ---
    if pbp_rows == 0 or sch_rows == 0:
        try:
            import nfl_data_py as nfl
            # PBP
            if pbp_rows == 0:
                try:
                    pbp_df = nfl.import_pbp_data([season])
                    if _safe_to_csv(pbp_df, f"data/pbp_{season}.csv"):
                        pbp_rows = len(pbp_df)
                        wrote_any = True
                        notes.append(
                            f"nfl_data_py pbp rows={pbp_rows} (season {season})"
                        )
                except Exception as e:
                    notes.append(f"nfl_data_py pbp: {type(e).__name__}: {e}")

            # Schedules
            if sch_rows == 0:
                try:
                    sch_df = nfl.import_schedules([season])
                    if _safe_to_csv(sch_df, f"data/schedules_{season}.csv"):
                        sch_rows = len(sch_df)
                        wrote_any = True
                        notes.append(
                            f"nfl_data_py schedules rows={sch_rows} (season {season})"
                        )
                except Exception as e:
                    notes.append(f"nfl_data_py schedules: {type(e).__name__}: {e}")

        except Exception as e:
            notes.append(f"nfl_data_py not available: {type(e).__name__}: {e}")

    # Success if any core file has rows
    ok = wrote_any and (pbp_rows > 0 or sch_rows > 0 or part_rows > 0)
    if not ok:
        notes.append("no rows written by nflreadpy/nfl_data_py")
        return {"ok": False, "source": source, "notes": notes}

    return {"ok": True, "source": source, "notes": notes}

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
