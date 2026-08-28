#!/usr/bin/env python3
"""Migration 60B: fetch historical pregame QB passing-yard props.

The football projections must already exist. This script never exposes a
sportsbook line to the projection builders; it only retrieves historical lines
for post-hoc benchmarking.

Historical player props are a paid Odds API endpoint. Before any paid pull, the
script reads the zero-cost quota headers and estimates the worst-case credit
cost. A conservative quota guard stops the pull without failure if the request
would consume too much of the remaining allowance.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from scripts.utils.canonical_names import canonicalize_player_name_safe

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

TEAM_FULL = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def _to_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _canon_key(name) -> str:
    try:
        _, key = canonicalize_player_name_safe(name)
        if key:
            return str(key)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(name or "") if ch.isalnum())


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _kickoff_utc(gameday, gametime) -> datetime:
    raw = f"{str(gameday).strip()} {str(gametime).strip()}"
    dt = datetime.strptime(raw, "%Y-%m-%d %H:%M")
    return dt.replace(tzinfo=ET).astimezone(UTC)


def _get(url: str, params: dict, *, timeout: int = 30) -> requests.Response:
    last = None
    for delay in (0.0, 0.8, 1.5, 3.0):
        if delay:
            time.sleep(delay)
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code != 429:
                return r
            last = r
        except requests.RequestException as exc:
            last = exc
    if isinstance(last, requests.Response):
        return last
    raise RuntimeError(f"Odds API request failed: {last}")


def _write_preflight(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)
    print("=== M60 ODDS API PREFLIGHT ===")
    print(pd.DataFrame([row]).to_string(index=False))


def _parse_prop_response(payload: dict, *, game_id: str, snapshot: str, kickoff: str) -> list[dict]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return []
    rows: list[dict] = []
    for bm in data.get("bookmakers", []) or []:
        book = str(bm.get("key") or "").strip().lower()
        for market in bm.get("markets", []) or []:
            if market.get("key") != "player_pass_yds":
                continue
            grouped: dict[tuple[str, float], dict] = {}
            for out in market.get("outcomes", []) or []:
                side = str(out.get("name") or "").strip().upper()
                if side not in {"OVER", "UNDER"}:
                    continue
                player = out.get("description") or out.get("participant") or ""
                try:
                    line = float(out.get("point"))
                except Exception:
                    continue
                key = (_canon_key(player), line)
                rec = grouped.setdefault(key, {
                    "game_id": game_id,
                    "snapshot_utc": snapshot,
                    "kickoff_utc": kickoff,
                    "book": book,
                    "player": str(player),
                    "player_clean_key": key[0],
                    "line": line,
                    "over_odds": np.nan,
                    "under_odds": np.nan,
                    "book_last_update": market.get("last_update") or bm.get("last_update"),
                })
                price = pd.to_numeric(pd.Series([out.get("price")]), errors="coerce").iloc[0]
                rec["over_odds" if side == "OVER" else "under_odds"] = price
            rows.extend(grouped.values())
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--projection-file", action="append", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--snapshot-minutes", type=int, default=30)
    p.add_argument("--books", default="draftkings,fanduel")
    p.add_argument("--max-quota-fraction", type=float, default=0.50)
    p.add_argument("--quota-reserve", type=int, default=250)
    a = p.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = a.out_dir / "m60_odds_api_preflight.csv"
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        _write_preflight(preflight_path, {"status": "api_key_missing"})
        return 0

    frames = []
    for value in a.projection_file:
        path = Path(value)
        if not path.exists() or not path.stat().st_size:
            raise RuntimeError(f"missing projection file: {path}")
        frames.append(pd.read_csv(path))
    proj = pd.concat(frames, ignore_index=True)
    if proj.empty or "game_id" not in proj:
        raise RuntimeError("projection trace missing game_id")
    proj["season"] = pd.to_numeric(proj["season"], errors="coerce")
    wanted_ids = set(proj.game_id.dropna().astype(str))
    seasons = sorted(proj.season.dropna().astype(int).unique())

    import nflreadpy as nfl

    schedules = []
    for season in seasons:
        s = _to_pandas(nfl.load_schedules(int(season)))
        s.columns = [str(c).strip().lower() for c in s.columns]
        if "game_type" in s:
            s = s.loc[s.game_type.astype(str).str.upper().eq("REG")].copy()
        schedules.append(s)
    sched = pd.concat(schedules, ignore_index=True)
    sched = sched.loc[sched.game_id.astype(str).isin(wanted_ids)].copy()
    required = {"game_id", "season", "week", "gameday", "gametime", "home_team", "away_team"}
    missing = required - set(sched.columns)
    if missing:
        raise RuntimeError(f"nflverse schedule missing {sorted(missing)}")

    game_rows = []
    for _, r in sched.drop_duplicates("game_id").iterrows():
        try:
            kickoff = _kickoff_utc(r.gameday, r.gametime)
        except Exception as exc:
            print(f"[m60 odds] skip bad kickoff game={r.game_id}: {exc}")
            continue
        snap = kickoff - timedelta(minutes=int(a.snapshot_minutes))
        home = str(r.home_team).upper().strip()
        away = str(r.away_team).upper().strip()
        game_rows.append({
            "game_id": str(r.game_id), "season": int(r.season), "week": int(r.week),
            "home_team": home, "away_team": away,
            "home_full": TEAM_FULL.get(home, home), "away_full": TEAM_FULL.get(away, away),
            "kickoff_utc": _iso(kickoff), "snapshot_utc": _iso(snap),
        })
    games = pd.DataFrame(game_rows)
    if games.empty:
        raise RuntimeError("no stable-QB games resolved to nflverse kickoff times")

    # Zero-cost endpoint: use headers to learn remaining quota before paid history calls.
    quota_resp = _get(f"{BASE}/sports/", {"apiKey": api_key})
    remaining = pd.to_numeric(pd.Series([quota_resp.headers.get("x-requests-remaining")]), errors="coerce").iloc[0]
    unique_games = int(games.game_id.nunique())
    unique_snapshots = int(games.snapshot_utc.nunique())
    estimated_max_cost = int(unique_snapshots + 10 * unique_games)
    allowed_fraction = float(np.clip(a.max_quota_fraction, 0.0, 1.0))

    status = "ready"
    reason = ""
    if quota_resp.status_code != 200:
        status, reason = "quota_probe_failed", f"sports endpoint HTTP {quota_resp.status_code}"
    elif pd.isna(remaining):
        status, reason = "quota_unknown_blocked", "missing x-requests-remaining header"
    elif float(remaining) < estimated_max_cost + int(a.quota_reserve):
        status, reason = "quota_guard_blocked", "insufficient remaining credits plus reserve"
    elif estimated_max_cost > float(remaining) * allowed_fraction:
        status, reason = "quota_guard_blocked", f"estimated pull exceeds {allowed_fraction:.0%} of remaining quota"

    preflight = {
        "status": status,
        "reason": reason,
        "stable_qb_rows": int(len(proj)),
        "unique_games": unique_games,
        "unique_snapshot_times": unique_snapshots,
        "estimated_max_credits": estimated_max_cost,
        "quota_remaining_before": float(remaining) if pd.notna(remaining) else np.nan,
        "quota_reserve": int(a.quota_reserve),
        "max_quota_fraction": allowed_fraction,
        "snapshot_minutes_before_kickoff": int(a.snapshot_minutes),
        "books": a.books,
    }
    _write_preflight(preflight_path, preflight)
    games.to_csv(a.out_dir / "m60_target_games.csv", index=False)
    if status != "ready":
        return 0

    # Historical events cost 1 per non-empty request. Cache by exact snapshot time,
    # which lets the many Sunday games sharing a kickoff use one discovery call.
    event_cache: dict[str, list[dict]] = {}
    event_map_rows = []
    paid_access_denied = False
    for snap, gg in games.groupby("snapshot_utc"):
        kickoffs = pd.to_datetime(gg.kickoff_utc, utc=True)
        start = _iso((kickoffs.min() - pd.Timedelta(minutes=15)).to_pydatetime())
        end = _iso((kickoffs.max() + pd.Timedelta(minutes=15)).to_pydatetime())
        resp = _get(
            f"{BASE}/historical/sports/{SPORT}/events",
            {"apiKey": api_key, "date": snap, "commenceTimeFrom": start, "commenceTimeTo": end},
        )
        if resp.status_code in {401, 402, 403}:
            paid_access_denied = True
            print(f"[m60 odds] historical access denied HTTP {resp.status_code}")
            break
        if resp.status_code != 200:
            print(f"[m60 odds] historical events HTTP {resp.status_code} snapshot={snap}")
            event_cache[snap] = []
            continue
        payload = resp.json()
        events = payload.get("data", []) if isinstance(payload, dict) else []
        event_cache[snap] = events if isinstance(events, list) else []
        time.sleep(0.12)

    if paid_access_denied:
        preflight["status"] = "historical_plan_access_denied"
        preflight["reason"] = "Odds API key does not have historical endpoint access"
        _write_preflight(preflight_path, preflight)
        return 0

    event_ids: dict[str, str] = {}
    for _, g in games.iterrows():
        events = event_cache.get(g.snapshot_utc, [])
        match = next(
            (
                e for e in events
                if str(e.get("home_team") or "") == g.home_full
                and str(e.get("away_team") or "") == g.away_full
            ),
            None,
        )
        event_id = str(match.get("id") or "") if match else ""
        event_ids[g.game_id] = event_id
        event_map_rows.append({**g.to_dict(), "odds_event_id": event_id, "event_found": int(bool(event_id))})
    event_map = pd.DataFrame(event_map_rows)
    event_map.to_csv(a.out_dir / "m60_event_map.csv", index=False)

    prop_rows: list[dict] = []
    books = ",".join(x.strip().lower() for x in a.books.split(",") if x.strip())
    for i, g in games.iterrows():
        eid = event_ids.get(g.game_id, "")
        if not eid:
            continue
        resp = _get(
            f"{BASE}/historical/sports/{SPORT}/events/{eid}/odds",
            {
                "apiKey": api_key,
                "date": g.snapshot_utc,
                "regions": "us",
                "bookmakers": books,
                "markets": "player_pass_yds",
                "oddsFormat": "american",
            },
        )
        if resp.status_code != 200:
            print(f"[m60 odds] event odds HTTP {resp.status_code} game={g.game_id}")
            continue
        payload = resp.json()
        prop_rows.extend(
            _parse_prop_response(
                payload, game_id=g.game_id, snapshot=g.snapshot_utc, kickoff=g.kickoff_utc
            )
        )
        if (i + 1) % 25 == 0:
            print(f"[m60 odds] fetched {i+1}/{len(games)} target games; prop_rows={len(prop_rows)}")
        time.sleep(0.12)

    props = pd.DataFrame(prop_rows)
    if props.empty:
        props = pd.DataFrame(columns=[
            "game_id", "snapshot_utc", "kickoff_utc", "book", "player", "player_clean_key",
            "line", "over_odds", "under_odds", "book_last_update",
        ])
    props.to_csv(a.out_dir / "m60_historical_qb_pass_props.csv", index=False)

    # Capture ending quota without consuming credits.
    q2 = _get(f"{BASE}/sports/", {"apiKey": api_key})
    remaining_after = pd.to_numeric(pd.Series([q2.headers.get("x-requests-remaining")]), errors="coerce").iloc[0]
    preflight["status"] = "complete"
    preflight["quota_remaining_after"] = float(remaining_after) if pd.notna(remaining_after) else np.nan
    if pd.notna(remaining) and pd.notna(remaining_after):
        preflight["credits_used_observed"] = float(remaining - remaining_after)
    preflight["event_id_coverage"] = float(event_map.event_found.mean()) if len(event_map) else 0.0
    preflight["raw_prop_rows"] = int(len(props))
    _write_preflight(preflight_path, preflight)
    print(f"[m60 odds] wrote {len(props)} historical prop rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
