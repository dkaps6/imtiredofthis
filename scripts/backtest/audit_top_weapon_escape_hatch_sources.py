#!/usr/bin/env python3
"""M84 source audit for materially new top-weapon matchup information.

Source/feasibility only: no QB outcomes, no predictive fitting, no sportsbook data.
The audit deliberately separates scientific usefulness from deployable access.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

SOURCES = [
    {
        "source": "NFL_NGS_COVERAGE_RESPONSIBILITY",
        "url": "https://www.nfl.com/news/next-gen-stats-new-advanced-metrics-you-need-to-know-for-the-2025-nfl-season",
        "exact_receiver_defender_responsibility": True,
        "machine_readable_public_contract": False,
        "multi_season_research_history": False,
        "in_season_deployable": False,
        "free_phase_eligible": False,
        "novel_vs_m72_m75": True,
        "disposition": "IDEAL_BUT_PROPRIETARY",
        "notes": "NGS Coverage Responsibility identifies defender assignments/matchups frame-by-frame, but M84 has no stable public machine-readable historical/live feed contract.",
    },
    {
        "source": "BIG_DATA_BOWL_PFF_COVERAGE_ASSIGNMENT",
        "url": "https://www.kaggle.com/c/nfl-big-data-bowl-2025/data",
        "exact_receiver_defender_responsibility": True,
        "machine_readable_public_contract": True,
        "multi_season_research_history": False,
        "in_season_deployable": False,
        "free_phase_eligible": True,
        "novel_vs_m72_m75": True,
        "disposition": "COMPETITION_SLICE_ONLY",
        "notes": "PFF primary/secondary coverage matchup IDs and route labels are materially new, but the public competition slice is not a complete extensible 2023-2025 + live source contract.",
    },
    {
        "source": "NFLVERSE_PARTICIPATION_ROUTE",
        "url": "https://nflreadr.nflverse.com/articles/dictionary_participation.html",
        "exact_receiver_defender_responsibility": False,
        "machine_readable_public_contract": True,
        "multi_season_research_history": True,
        "in_season_deployable": False,
        "free_phase_eligible": True,
        "novel_vs_m72_m75": False,
        "disposition": "HISTORICAL_ROUTE_ONLY_NONDEPLOYABLE",
        "notes": "Participation contains primary-receiver route and defensive players, but not direct receiver-defender responsibility; 2023+ participation is released only after the season.",
    },
    {
        "source": "FANTASY_POINTS_VSIN_WR_CB",
        "url": "https://vsin.com/wr/",
        "exact_receiver_defender_responsibility": True,
        "machine_readable_public_contract": False,
        "multi_season_research_history": False,
        "in_season_deployable": True,
        "free_phase_eligible": True,
        "novel_vs_m72_m75": True,
        "disposition": "CURRENT_REPORT_NO_STABLE_HISTORY_CONTRACT",
        "notes": "Current WR/CB matchup reporting exposes receiver/defender alignment and advantage, but M84 cannot establish a stable machine-readable multi-season historical archive contract.",
    },
    {
        "source": "PFF_WR_CB_MATCHUP_CHART",
        "url": "https://www.pff.com/news/wr-vs-cb-matchup-chart",
        "exact_receiver_defender_responsibility": True,
        "machine_readable_public_contract": False,
        "multi_season_research_history": False,
        "in_season_deployable": True,
        "free_phase_eligible": False,
        "novel_vs_m72_m75": True,
        "disposition": "PAID_NO_FREE_RESEARCH_CONTRACT",
        "notes": "Weekly matchup product exists, but export/full access is subscription-dependent and does not satisfy the frozen free-source M84 contract.",
    },
]


def check_url(url: str) -> tuple[bool, int | None, str]:
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 M84-source-audit"})
        return bool(r.status_code < 500), int(r.status_code), str(r.url)
    except Exception as exc:
        return False, None, f"{type(exc).__name__}:{exc}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/m84_top_weapon_source_audit"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in SOURCES:
        ok, status, final = check_url(spec["url"])
        rec = dict(spec)
        rec.update({"url_reachable": ok, "http_status": status, "resolved_url": final})
        rec["complete_m84_contract"] = bool(
            rec["exact_receiver_defender_responsibility"]
            and rec["machine_readable_public_contract"]
            and rec["multi_season_research_history"]
            and rec["in_season_deployable"]
            and rec["free_phase_eligible"]
            and rec["novel_vs_m72_m75"]
        )
        rows.append(rec)

    inv = pd.DataFrame(rows)
    inv.to_csv(args.out_dir / "m84_source_inventory.csv", index=False)

    qualifying = inv.loc[inv["complete_m84_contract"]].copy()
    if len(qualifying):
        disposition = "QUALIFIED_FOR_M85_PREDICTIVE_DEVELOPMENT"
    else:
        new_but_blocked = inv.loc[inv["novel_vs_m72_m75"]].copy()
        disposition = "HOLD_SOURCE_BLOCKED_NEW_INFORMATION" if len(new_but_blocked) else "CLOSED_NO_MATERIALLY_NEW_SOURCE"

    decision = {
        "migration": "M84",
        "source_candidates": int(len(inv)),
        "qualifying_sources": int(len(qualifying)),
        "final_disposition": disposition,
        "qb_outcomes_read": False,
        "sportsbook_features_used": False,
        "production_actionable": False,
        "m82_full_stack_benchmark_mae": 56.749517,
        "m85_allowed": bool(len(qualifying)),
        "anti_loop": "Do not reuse M72/M75 aggregates or a competition/editorial/current-only source as if it satisfied the missing historical+live individual-matchup contract.",
    }
    (args.out_dir / "m84_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    print("[m84_source_inventory]")
    print(inv.to_string(index=False))
    print("[m84_decision]")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
