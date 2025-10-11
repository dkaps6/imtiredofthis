#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys, time, datetime as dt
from pathlib import Path
from typing import List, Dict, Any, Tuple
import requests
import pandas as pd

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

# ---------- helpers ----------
def _iso_bounds_for_date(d: str) -> Dict[str, str]:
    day = dt.date.fromisoformat(d)
    start = dt.datetime.combine(day, dt.time(0, 0))
    end = start + dt.timedelta(days=1)
    return {
        "commenceTimeFrom": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dateFormat": "iso",
    }

def american_to_prob(american: float | int | None) -> float:
    """Convert American odds to implied probability (vigged)."""
    if american is None:
        return 0.0
    try:
        a = float(american)
    except Exception:
        return 0.0
    if a > 0:
        return 100.0 / (a + 100.0)
    else:
        return (-a) / ((-a) + 100.0)

def _normalize_event_rows(payload: List[Dict[str, Any]], market_key: str, allowed_books: List[str]) -> List[Dict[str, Any]]:
    """
    Flatten Odds API JSON into rows:
    event_id, commence_time, home_team, away_team, book, market, player, line, side, odds
    For non-player markets (h2h/spreads/totals) we still emit rows (player blank).
    """
    rows = []
    for ev in payload or []:
        event_id = ev.get("id")
        commence_time = ev.get("commence_time")
        home_team = ev.get("home_team") or ""
        away_team = ev.get("away_team") or ""

        for bk in (ev.get("bookmakers") or []):
            book_key = bk.get("key")
            if allowed_books and book_key not in allowed_books:
                continue
            for mk in (bk.get("markets") or []):
                if mk.get("key") != market_key:
                    continue
                for oc in (mk.get("outcomes") or []):
                    rows.append({
                        "event_id": event_id,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "book": book_key,
                        "market": market_key,
                        "player": oc.get("name") or "",
                        "line": oc.get("point"),
                        "side": oc.get("description") or oc.get("name") or "",
                        "odds": oc.get("price"),
                    })
    return rows

def fetch_market(
    api_key: str,
    market: str,
    books: List[str],
    date_str: str = "",
    region: str = "us",
    odds_format: str = "american",
    pause_sec: float = 0.30,
) -> Tuple[pd.DataFrame, int]:
    params = {
        "apiKey": api_key,
        "regions": region,
        "oddsFormat": odds_format,
        "bookmakers": ",".join(books) if books else "",
        "markets": market,
    }
    if date_str:
        params.update(_iso_bounds_for_date(date_str))

    url = BASE
    print(f"[oddsapi] request market={market} -> {url}")
    r = requests.get(url, params=params, timeout=20)

    # Rate-limit headers
    xr = r.headers.get("x-requests-remaining")
    xu = r.headers.get("x-requests-used")
    if xr is not None and xu is not None:
        print(f"[oddsapi] x-requests-remaining: {xr}")
        print(f"[oddsapi] x-requests-used: {xu}")

    status = r.status_code
    if status == 200:
        data = r.json() if r.text else []
        rows = _normalize_event_rows(data, market, books)
        df = pd.DataFrame(rows)
        print(f"[oddsapi] market={market} rows={len(df)}")
        time.sleep(pause_sec)
        return df, status

    print(f"[oddsapi] error market={market}: {status} {r.text[:220]}")
    time.sleep(pause_sec)
    # empty with correct schema
    cols = ["event_id","commence_time","home_team","away_team","book","market","player","line","side","odds"]
    return pd.DataFrame(columns=cols), status

# Optional: FanDuel CSV fallback hook stays intact (no change needed here)
def fallback_fanduel_csv(market: str, date_str: str, fd_dir: str) -> pd.DataFrame:
    if not fd_dir:
        return pd.DataFrame()
    Path(fd_dir).mkdir(parents=True, exist_ok=True)
    fname = f"props_{market}_{date_str}.csv" if date_str else f"props_{market}.csv"
    path = Path(fd_dir) / fname
    if not path.exists():
        print(f"[fd-fallback] no file at {path}")
        return pd.DataFrame()
    try:
        raw = pd.read_csv(path)
    except Exception as e:
        print(f"[fd-fallback] failed reading {path}: {e}")
        return pd.DataFrame()
    out = pd.DataFrame({
        "event_id": raw.get("event_id"),
        "commence_time": raw.get("commence_time"),
        "home_team": raw.get("home_team",""),
        "away_team": raw.get("away_team",""),
        "book": "fanduel",
        "market": market,
        "player": raw.get("player",""),
        "line": raw.get("line"),
        "side": raw.get("side",""),
        "odds": raw.get("odds"),
    })
    out = out.dropna(subset=["odds"])
    print(f"[fd-fallback] merged {len(out)} rows from {path.name}")
    return out

def build_game_lines(all_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-event game lines with:
      event_id, home_team, away_team, home_wp, away_wp
    Using h2h odds only:
      - compute implied prob per book per team from American odds
      - devig per book by normalizing the pair to 1
      - average across books
    """
    df = all_rows
    if df.empty:
        return pd.DataFrame(columns=["event_id","home_team","away_team","home_wp","away_wp"])

    h2h = df[df["market"] == "h2h"].copy()
    if h2h.empty:
        # no h2h; return structure so pricing won’t explode
        return pd.DataFrame(columns=["event_id","home_team","away_team","home_wp","away_wp"])

    # Tag outcome as home or away by matching side text against home/away team names
    h2h["is_home_side"] = (h2h["side"].fillna("").str.strip().str.lower()
                           == h2h["home_team"].fillna("").str.strip().str.lower())
    h2h["is_away_side"] = (h2h["side"].fillna("").str.strip().str.lower()
                           == h2h["away_team"].fillna("").str.strip().str.lower())

    # Keep only rows that matched one of the teams
    h2h = h2h[(h2h["is_home_side"]) | (h2h["is_away_side"])].copy()
    if h2h.empty:
        return pd.DataFrame(columns=["event_id","home_team","away_team","home_wp","away_wp"])

    # implied prob per outcome
    h2h["p_implied"] = h2h["odds"].apply(american_to_prob)

    # devig per (event_id, book): normalize home+away to 1
    def _devig(group: pd.DataFrame) -> pd.DataFrame:
        # compute sums
        ph = group.loc[group["is_home_side"], "p_implied"].sum()
        pa = group.loc[group["is_away_side"], "p_implied"].sum()
        total = ph + pa
        if total > 0:
            group.loc[group["is_home_side"], "p_fair"] = group.loc[group["is_home_side"], "p_implied"] / total
            group.loc[group["is_away_side"], "p_fair"] = group.loc[group["is_away_side"], "p_implied"] / total
        else:
            group["p_fair"] = 0.0
        return group

    h2h = h2h.groupby(["event_id","book"], group_keys=False).apply(_devig)

    # aggregate per event by averaging fair probs across books
    agg = h2h.groupby(["event_id","home_team","away_team"]).apply(
        lambda g: pd.Series({
            "home_wp": g.loc[g["is_home_side"], "p_fair"].mean() if (g["is_home_side"]).any() else 0.0,
            "away_wp": g.loc[g["is_away_side"], "p_fair"].mean() if (g["is_away_side"]).any() else 0.0,
        })
    ).reset_index()

    # final safety: renormalize small drift so home_wp + away_wp = 1
    s = agg["home_wp"] + agg["away_wp"]
    s = s.where(s > 0, 1.0)
    agg["home_wp"] = agg["home_wp"] / s
    agg["away_wp"] = agg["away_wp"] / s

    return agg[["event_id","home_team","away_team","home_wp","away_wp"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--markets", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--out", default="outputs/props_raw.csv")
    ap.add_argument("--fd-fallback-dir", default="external/fanduel")
    a = ap.parse_args()

    api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key:
        print("[oddsapi] ERROR: ODDS_API_KEY is not set"); sys.exit(2)

    books = [b.strip() for b in a.books.split(",") if b.strip()]
    if a.markets.strip():
        markets = [m.strip() for m in a.markets.split(",") if m.strip()]
    else:
        markets = [
            "h2h","spreads","totals",
            "player_pass_yds","player_rush_yds","player_rec_yds",
            "player_receptions","player_passing_tds","player_anytime_td",
        ]

    frames = []
    for m in markets:
        df, status = fetch_market(api_key, m, books=books, date_str=a.date)
        if df.empty:
            # try FanDuel CSV fallback for props only
            if m.startswith("player_"):
                fd_df = fallback_fanduel_csv(m, a.date, a.fd_fallback_dir)
                if not fd_df.empty:
                    df = fd_df
        if not df.empty:
            frames.append(df)

    # props_raw.csv (merged)
    props = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["event_id","commence_time","home_team","away_team","book","market","player","line","side","odds"]
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    props.to_csv(a.out, index=False)
    print(f"[oddsapi] wrote props={len(props)} -> {a.out}")

    # odds_game.csv (proper game-lines for pricing.py)
    odds_game = build_game_lines(props)
    game_out = "outputs/odds_game.csv"
    odds_game.to_csv(game_out, index=False)
    print(f"[oddsapi] wrote game lines={len(odds_game)} -> {game_out}")

if __name__ == "__main__":
    main()
