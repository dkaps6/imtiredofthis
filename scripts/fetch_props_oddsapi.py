#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, os, sys, time
from pathlib import Path
import requests

SPORT = "americanfootball_nfl"  # TheOddsAPI sport key

def _q(url, params):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    ap = argparse.ArgumentParser(description="Fetch NFL odds/props from TheOddsAPI with graceful fallbacks")
    ap.add_argument("--books", required=True, help="Comma list of bookmakers")
    ap.add_argument("--markets", default="", help="Comma list of player markets to try")
    ap.add_argument("--date", default="", help="YYYY-MM-DD (optional)")
    ap.add_argument("--out", default="outputs/props_raw.csv")
    args = ap.parse_args()

    api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key:
        print("[oddsapi] ERROR: ODDS_API_KEY not set", file=sys.stderr)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("")
        Path("outputs/odds_game.csv").write_text("")
        return 0

    Path("outputs").mkdir(exist_ok=True)
    books = [b.strip() for b in args.books.split(",") if b.strip()]
    player_markets = [m.strip() for m in args.markets.split(",") if m.strip()]

    base = "https://api.the-odds-api.com/v4"
    common = {
        "regions": "us",
        "oddsFormat": "american",
        "bookmakers": ",".join(books),
        "apiKey": api_key,
    }
    if args.date:
        common["dateFormat"] = "iso"
        common["commenceTimeFrom"] = args.date

    # 1) Game markets (h2h, spreads, totals)
    try:
        print("[oddsapi] game markets=h2h,spreads,totals")
        game = _q(f"{base}/sports/{SPORT}/odds", {**common, "markets": "h2h,spreads,totals"})
        with open("outputs/odds_game.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id","commence_time","home","away","bookmaker","market","price"])
            for g in game:
                gid = g.get("id")
                ct  = g.get("commence_time","")
                home = g.get("home_team",""); away = g.get("away_team","")
                for bk in g.get("bookmakers", []):
                    bname = bk.get("key")
                    for mkt in bk.get("markets", []):
                        mname = mkt.get("key")
                        for o in mkt.get("outcomes", []):
                            w.writerow([gid, ct, home, away, bname, mname, o.get("price")])
    except requests.HTTPError as e:
        print(f"[oddsapi] WARNING game markets failed: {e}", file=sys.stderr)
        Path("outputs/odds_game.csv").write_text("")

    # 2) Player markets (probe individually; handle 401 cleanly)
    wrote_any = False
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "game_id","commence_time","player","team","position",
            "market","outcome","line","price","bookmaker"
        ])
        targets = player_markets or [
            "player_receiving_yards","player_receptions","player_rushing_yards",
            "player_rush_and_receive_yards","player_passing_yards","player_passing_tds",
            "player_anytime_td"
        ]
        for m in targets:
            try:
                print(f"[oddsapi] probe market={m}")
                js = _q(f"{base}/sports/{SPORT}/odds", {**common, "markets": m})
                for g in js:
                    gid = g.get("id"); ct = g.get("commence_time","")
                    for bk in g.get("bookmakers", []):
                        bname = bk.get("key")
                        for mkt in bk.get("markets", []):
                            for o in mkt.get("outcomes", []):
                                w.writerow([gid, ct, o.get("participant","") or o.get("description",""),
                                            o.get("team",""), o.get("position",""),
                                            m, o.get("name",""), o.get("point",""), o.get("price",""), bname])
                                wrote_any = True
                time.sleep(0.35)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 401:
                    print(f"[oddsapi] 401 Unauthorized on {m} — continuing with others.", file=sys.stderr)
                    continue
                print(f"[oddsapi] error market={m}: {e}", file=sys.stderr)
                continue

    if not wrote_any:
        print("[oddsapi] no supported player markets detected (props=0)")
    else:
        print("[oddsapi] props written")

    return 0

if __name__ == "__main__":
    sys.exit(main())
