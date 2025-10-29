# --- ADD: provider runners (non-destructive) ---

from pathlib import Path
import json, os, time, traceback
import pandas as pd

def _size(p): 
    try: return Path(p).stat().st_size
    except: return 0

def _exists_nonempty(p): 
    try: 
        pp = Path(p)
        return pp.exists() and pp.stat().st_size > 64  # >64 bytes as "non-empty"
    except:
        return False

def _skip_if_present(team_csv, player_csv):
    # if both already exist and look non-empty, we consider provider "ok"
    if _exists_nonempty(team_csv) or _exists_nonempty(player_csv):
        return True, [f"using existing: {team_csv if _exists_nonempty(team_csv) else '(none)'}",
                      f"using existing: {player_csv if _exists_nonempty(player_csv) else '(none)'}"]
    return False, []

def run_nflverse(season: int, date: str | None):
    # You already build from nflverse in make_team_form/make_player_form.
    # We just mark this stage as best-effort and non-fatal.
    return {"source":"nflverse","ok":False,"notes":["nflverse adapter missing"]}

def run_nflgsis(season: int, date: str | None):
    src = "NFLGSIS"
    team_csv   = "data/gsis_team.csv"
    player_csv = "data/gsis_player.csv"

    ok_existing, notes = _skip_if_present(team_csv, player_csv)
    if ok_existing:
        return {"source":src,"ok":True,"notes":notes}

    # Many setups use nfl_data_py / nflreadpy to materialize CSVs;
    # If you want an explicit GSIS fetcher, I can add scripts/providers/gsis_pull.py
    return {"source":src,"ok":False,"notes":["NFLGSIS adapter missing"]}

def run_msf(season: int, date: str | None):
    src = "MySportsFeeds"
    team_csv   = "data/msf_team.csv"
    player_csv = "data/msf_player.csv"

    ok_existing, notes = _skip_if_present(team_csv, player_csv)
    if ok_existing:
        return {"source":src,"ok":True,"notes":notes}

    key = os.getenv("MSF_API_KEY", "")
    if not key:
        return {"source":src,"ok":False,"notes":["MSF_API_KEY missing; skip"]}

    try:
        # I can wire scripts/providers/msf_pull.py to hit v2.1 endpoints and write those CSVs.
        return {"source":src,"ok":False,"notes":["MSF adapter present but fetcher not yet wired"]}

    except Exception as e:
        return {"source":src,"ok":False,"notes":[f"{type(e).__name__}: {e}"]}

def run_apisports(season: int, date: str | None):
    src = "API-Sports"
    team_csv   = "data/apisports_team.csv"
    player_csv = "data/apisports_player.csv"

    ok_existing, notes = _skip_if_present(team_csv, player_csv)
    if ok_existing:
        return {"source":src,"ok":True,"notes":notes}

    key = os.getenv("APISPORTS_KEY", "") or os.getenv("API_SPORTS_KEY", "")
    if not key:
        return {"source":src,"ok":False,"notes":["APISPORTS_KEY missing; skip"]}

    try:
        # I can wire scripts/providers/apisports_pull.py to hit /players & /teams endpoints.
        return {"source":src,"ok":False,"notes":["API-Sports adapter present but fetcher not yet wired"]}

    except Exception as e:
        return {"source":src,"ok":False,"notes":[f"{type(e).__name__}: {e}"]}
