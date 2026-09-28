"""Emit every player stat line for one NFL week from the public ESPN box scores.

This exists because the research container has no general outbound egress -- every
stats host is refused by the network proxy -- while a CI runner does. Run it in
Actions and read the CSV back out of the job log.

The output is one row per player per game with the four quantities the betting
board prices: pass_yards, rush_yards, rec_yards and receptions. Players who did
not record a stat line are absent, which is itself signal: a priced player with
no row was inactive or did not play.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from collections import defaultdict

SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    "?dates={season}&seasontype=2&week={week}"
)
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event}"

# ESPN reports each category's stats as a positional list described by its own
# "labels" array. Read by label rather than by index -- the order is not stable
# across categories and silently shifts meaning if you hard-code positions.
WANTED = {
    "passing": {"YDS": "pass_yards"},
    "rushing": {"YDS": "rush_yards"},
    "receiving": {"YDS": "rec_yards", "REC": "receptions"},
}


def get(url: str, attempts: int = 4) -> dict:
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - surface the final failure only
            last = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    args = ap.parse_args()

    board = get(SCOREBOARD.format(season=args.season, week=args.week))
    events = board.get("events", [])
    print(f"week {args.week}: {len(events)} games", file=sys.stderr)

    rows: dict[tuple, dict] = {}
    meta: list[str] = []

    for ev in events:
        eid = ev["id"]
        comp = ev["competitions"][0]
        status = comp["status"]["type"]["name"]
        teams = {t["homeAway"]: t["team"]["abbreviation"] for t in comp["competitors"]}
        label = f"{teams.get('away')}@{teams.get('home')}"
        meta.append(f"{label}\t{status}\t{ev.get('date', '')}")
        if status != "STATUS_FINAL":
            print(f"  skip {label}: {status}", file=sys.stderr)
            continue

        summary = get(SUMMARY.format(event=eid))
        players = (summary.get("boxscore") or {}).get("players") or []
        for team_block in players:
            abbr = team_block["team"]["abbreviation"]
            for cat in team_block.get("statistics", []):
                name = cat.get("name")
                if name not in WANTED:
                    continue
                labels = cat.get("labels", [])
                for ath in cat.get("athletes", []):
                    who = ath["athlete"]["displayName"]
                    stats = ath.get("stats", [])
                    key = (label, abbr, who)
                    row = rows.setdefault(
                        key,
                        {"game": label, "team": abbr, "player": who,
                         "pass_yards": "", "rush_yards": "", "rec_yards": "", "receptions": ""},
                    )
                    for lab, col in WANTED[name].items():
                        if lab in labels:
                            v = stats[labels.index(lab)]
                            row[col] = v
        print(f"  {label}: ok", file=sys.stderr)

    print("<<<GAMESTATUS")
    for m in meta:
        print(m)
    print("GAMESTATUS>>>")

    print("<<<BOXCSV")
    w = csv.DictWriter(
        sys.stdout,
        fieldnames=["game", "team", "player", "pass_yards", "rush_yards", "rec_yards", "receptions"],
        lineterminator="\n",
    )
    w.writeheader()
    for key in sorted(rows):
        w.writerow(rows[key])
    print("BOXCSV>>>")
    print(f"total player rows: {len(rows)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
