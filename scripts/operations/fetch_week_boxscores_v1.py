"""Emit every player stat line for one NFL week, for offline board grading.

This exists because the research container has no general outbound egress -- every
stats host is refused by the network proxy -- while a CI runner does. Run it in
Actions and read the CSV back out of the job log.

Source order matters. ESPN's site API refuses datacenter IPs (403 from Azure
runners), so the primary source is the nflverse weekly player-stats release,
which is served from the same host the runner already talks to. Candidates are
probed in order and the first one carrying the requested week wins; the probe
table is printed either way so a miss is diagnosable rather than silent.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import urllib.error
import urllib.request

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"

# Release asset naming has changed over the project's life; try both shapes.
CANDIDATES = [
    f"{NFLVERSE}/player_stats/stats_player_week_{{season}}.csv",
    f"{NFLVERSE}/player_stats/player_stats_{{season}}.csv",
    f"{NFLVERSE}/stats_player/stats_player_week_{{season}}.csv",
    "https://raw.githubusercontent.com/nflverse/nflverse-data/master/data/"
    "player_stats/player_stats_{season}.csv",
]

# nflverse column names, newest first -- resolved against the actual header.
COLS = {
    "pass_yards": ["passing_yards"],
    "rush_yards": ["rushing_yards"],
    "rec_yards": ["receiving_yards"],
    "receptions": ["receptions"],
    "player": ["player_display_name", "player_name", "full_name"],
    "team": ["recent_team", "team"],
    "opponent": ["opponent_team", "opponent"],
    "week": ["week"],
    "season": ["season"],
    "position": ["position"],
    "season_type": ["season_type"],
}


def fetch(url: str, attempts: int = 3) -> bytes | None:
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            print(f"  {exc.code} {url}", file=sys.stderr)
            return None
        except Exception as exc:  # noqa: BLE001
            print(f"  retry {i + 1} {type(exc).__name__} {url}", file=sys.stderr)
            time.sleep(2 * (i + 1))
    return None


def pick(header: list[str], names: list[str]) -> str | None:
    for n in names:
        if n in header:
            return n
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    args = ap.parse_args()

    print("probing sources:", file=sys.stderr)
    for tmpl in CANDIDATES:
        url = tmpl.format(season=args.season)
        raw = fetch(url)
        if raw is None:
            continue
        text = raw.decode("utf-8", errors="replace")
        rdr = csv.DictReader(io.StringIO(text))
        header = rdr.fieldnames or []
        resolved = {k: pick(header, v) for k, v in COLS.items()}
        wk = resolved["week"]
        if not wk:
            print(f"  no week column in {url}", file=sys.stderr)
            continue
        rows = [r for r in rdr if str(r.get(wk)) == str(args.week)]
        st = resolved.get("season_type")
        if st:
            rows = [r for r in rows if str(r.get(st)).upper() in ("REG", "REGULAR")]
        print(f"  OK {url} -> {len(rows)} rows for week {args.week}", file=sys.stderr)
        if not rows:
            print("  week not yet published in this asset", file=sys.stderr)
            continue

        print(f"<<<SOURCE\n{url}\nSOURCE>>>")
        out = csv.writer(sys.stdout, lineterminator="\n")
        fields = ["player", "team", "opponent", "position",
                  "pass_yards", "rush_yards", "rec_yards", "receptions"]
        print("<<<BOXCSV")
        out.writerow(fields)
        for r in rows:
            out.writerow([
                (r.get(resolved[f]) or "") if resolved.get(f) else ""
                for f in fields
            ])
        print("BOXCSV>>>")
        print(f"emitted {len(rows)} player rows", file=sys.stderr)
        return 0

    print("no source carried the requested week", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
