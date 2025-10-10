from __future__ import annotations
import os, requests, pandas as pd
from pathlib import Path

OUT = Path("external"); OUT.mkdir(exist_ok=True)
NFLV_OUT = OUT/"nflverse_bundle"/"outputs"; NFLV_OUT.mkdir(parents=True, exist_ok=True)

def _ok(df: pd.DataFrame) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty

def _safe_write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
        return True
    except Exception as e:
        print(f"[provider] write failed for {path}: {e}")
        return False

def run_nflverse(season: int, date: str|None):
    try:
        import nfl_data_py as nfl
    except Exception as e:   # <- no bare 'Error'
        print(f"[provider:nflverse] import failed: {e}")
        _safe_write_csv(pd.DataFrame(), NFLV_OUT/"schedules.csv")
        _safe_write_csv(pd.DataFrame(), NFLV_OUT/"pbp.csv")
        return {"ok": False, "source":"nflverse", "notes":[f"nfl_data_py import error: {e}"]}
    try:
        print(f"[provider:nflverse] fetching PBP for {season} …")
        pbp = nfl.import_pbp_data([season])
        print(f"[provider:nflverse] fetching schedules for {season} …")
        sched = nfl.import_schedules([season])
    except Exception as e:   # <- no bare 'Error'
        print(f"[provider:nflverse] fetch error: {e}")
        _safe_write_csv(pd.DataFrame(), NFLV_OUT/"schedules.csv")
        _safe_write_csv(pd.DataFrame(), NFLV_OUT/"pbp.csv")
        return {"ok": False, "source":"nflverse", "notes":[f"fetch error: {e}"]}

    _safe_write_csv(pbp,   NFLV_OUT/"pbp.csv")
    _safe_write_csv(sched, NFLV_OUT/"schedules.csv")
    ok = _ok(pbp) and _ok(sched)
    return {"ok": ok, "source":"nflverse", "rows":{"pbp":len(pbp),"sched":len(sched)}, "notes":["pbp+sched via nfl_data_py"]}

def _espn_cookie():
    c = os.getenv("ESPN_COOKIE","")
    return c.strip()

def run_espn(season: int, date: str|None):
    try:
        headers = {"Cookie": _espn_cookie()} if _espn_cookie() else {}
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?year={season}"
        r = requests.get(url, headers=headers, timeout=20); r.raise_for_status()
        data = r.json()
        df = pd.json_normalize(data.get("events", []))
        _safe_write_csv(df, NFLV_OUT/"espn_scoreboard.csv")
        return {"ok": not df.empty, "source":"ESPN", "rows":{"events":len(df)}, "notes":["scoreboard pulled"]}
    except Exception as e:
        _safe_write_csv(pd.DataFrame(), NFLV_OUT/"espn_scoreboard.csv")
        return {"ok": False, "source":"ESPN", "notes":[str(e)]}

def run_nflgsis(season: int, date: str|None):
    if not (os.getenv("NFLGSIS_USERNAME") and os.getenv("NFLGSIS_PASSWORD")):
        return {"ok": False, "source":"NFLGSIS", "notes":["missing creds"]}
    return {"ok": False, "source":"NFLGSIS", "notes":["no public endpoint wired"]}

def run_msf(season: int, date: str|None):
    if not os.getenv("MSF_KEY"):
        return {"ok": False, "source":"MySportsFeeds", "notes":["missing key"]}
    return {"ok": False, "source":"MySportsFeeds", "notes":["stub (wire paid endpoint)"]}

def run_apisports(season: int, date: str|None):
    if not os.getenv("APISPORTS_KEY"):
        return {"ok": False, "source":"API-Sports", "notes":["missing key"]}
    return {"ok": False, "source":"API-Sports", "notes":["stub (last fallback)"]}
