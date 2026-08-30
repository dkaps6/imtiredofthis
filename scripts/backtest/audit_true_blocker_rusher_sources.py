#!/usr/bin/env python3
"""M85 source audit for exact blocker-rusher assignment information."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

SOURCES = [
    {
        "source": "NFL_NGS_BLOCKING_MATCHUPS",
        "url": "https://www.nfl.com/news/next-gen-stats-introduction-to-pressure-probability",
        "exact_assignment": True,
        "machine_readable_public_contract": False,
        "multi_season_history": True,
        "in_season_deployable": False,
        "free_phase_eligible": False,
        "novel_vs_prior": True,
        "disposition": "IDEAL_BUT_PROPRIETARY",
        "notes": "NFL NGS reports automated who-blocked-who identification and historical coverage since 2018, but no stable public machine-readable historical/live feed contract is available to M85.",
    },
    {
        "source": "BIG_DATA_BOWL_PFF_BLOCKER_RUSHER",
        "url": "https://www.kaggle.com/c/nfl-big-data-bowl-2025/data",
        "exact_assignment": True,
        "machine_readable_public_contract": True,
        "multi_season_history": False,
        "in_season_deployable": False,
        "free_phase_eligible": True,
        "novel_vs_prior": True,
        "disposition": "COMPETITION_SLICE_ONLY",
        "notes": "blockedPlayerNFLId1/2/3 and pressureAllowedAsBlocker are exact assignment-level fields, but the public competition data do not provide a complete extensible multi-season + live contract.",
    },
    {
        "source": "NFLVERSE_PARTICIPATION_PASS_RUSH",
        "url": "https://nflreadr.nflverse.com/articles/dictionary_participation.html",
        "exact_assignment": False,
        "machine_readable_public_contract": True,
        "multi_season_history": True,
        "in_season_deployable": False,
        "free_phase_eligible": True,
        "novel_vs_prior": False,
        "disposition": "AGGREGATE_PLAY_CONTEXT_ONLY",
        "notes": "Participation exposes players on play and number of pass rushers but not exact blocker-rusher assignment; 2023+ participation is also postseason-only.",
    },
    {
        "source": "PUBLIC_ADVANCED_PASS_RUSH_OL_TABLES",
        "url": "https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html",
        "exact_assignment": False,
        "machine_readable_public_contract": True,
        "multi_season_history": True,
        "in_season_deployable": True,
        "free_phase_eligible": True,
        "novel_vs_prior": False,
        "disposition": "AGGREGATE_PLAYER_TABLES_NOT_ASSIGNMENTS",
        "notes": "Public advanced stats can update in-season but summarize players/teams rather than identify the specific blocker-rusher exposure required by M85.",
    },
]


def check(url: str):
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 M85-source-audit"})
        return r.status_code < 500, int(r.status_code), str(r.url)
    except Exception as exc:
        return False, None, f"{type(exc).__name__}:{exc}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/m85_blocker_rusher_source_audit"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in SOURCES:
        ok, status, final = check(spec["url"])
        r = dict(spec)
        r.update({"url_reachable": ok, "http_status": status, "resolved_url": final})
        r["complete_m85_contract"] = bool(
            r["exact_assignment"]
            and r["machine_readable_public_contract"]
            and r["multi_season_history"]
            and r["in_season_deployable"]
            and r["free_phase_eligible"]
            and r["novel_vs_prior"]
        )
        rows.append(r)

    inv = pd.DataFrame(rows)
    inv.to_csv(args.out_dir / "m85_source_inventory.csv", index=False)
    qualifying = inv.loc[inv["complete_m85_contract"]]
    if len(qualifying):
        disposition = "QUALIFIED_FOR_M86_PREDICTIVE_DEVELOPMENT"
    elif inv["novel_vs_prior"].any():
        disposition = "HOLD_SOURCE_BLOCKED_NEW_INFORMATION"
    else:
        disposition = "CLOSED_NO_MATERIALLY_NEW_SOURCE"

    decision = {
        "migration": "M85",
        "source_candidates": int(len(inv)),
        "qualifying_sources": int(len(qualifying)),
        "final_disposition": disposition,
        "qb_outcomes_read": False,
        "sportsbook_features_used": False,
        "production_actionable": False,
        "m86_allowed": bool(len(qualifying)),
        "anti_loop": "Do not substitute aggregate pressure, OL continuity, pass-rush rates, or competition-only assignments for a complete historical+live exact blocker-rusher contract.",
    }
    (args.out_dir / "m85_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print("[m85_source_inventory]")
    print(inv.to_string(index=False))
    print("[m85_decision]")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
