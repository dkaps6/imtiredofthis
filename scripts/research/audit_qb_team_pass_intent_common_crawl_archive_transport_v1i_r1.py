#!/usr/bin/env python3
"""Strict V1I-R1 archive identity recertification.

Consumes the exact preserved V1I artifact. No network retrieval.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.parse
from pathlib import Path
from typing import Any

VERSION = "QB_TEAM_PASS_INTENT_COMMON_CRAWL_ARCHIVE_TRANSPORT_V1I_R1"
EXPECTED_ROWS = 158
EXPECTED_DIRECT = 123
EXPECTED_ARCHIVE_NEEDED = 35
GATE_ROWS = 127

FRANCHISE_IDENTITY = {
    "ARI": ("cardinals",),
    "ATL": ("falcons",),
    "BAL": ("ravens",),
    "BUF": ("bills",),
    "CAR": ("panthers",),
    "CHI": ("bears",),
    "CIN": ("bengals",),
    "CLE": ("browns",),
    "DAL": ("cowboys",),
    "DEN": ("broncos",),
    "DET": ("lions",),
    "GB": ("packers",),
    "HOU": ("texans",),
    "IND": ("colts",),
    "JAX": ("jaguars", "jags"),
    "KC": ("chiefs",),
    "LV": ("raiders",),
    "LAC": ("chargers",),
    "LAR": ("rams",),
    "MIA": ("dolphins",),
    "MIN": ("vikings",),
    "NE": ("patriots",),
    "NO": ("saints",),
    "NYG": ("giants",),
    "NYJ": ("jets",),
    "PHI": ("eagles",),
    "PIT": ("steelers",),
    "SF": ("49ers", "niners"),
    "SEA": ("seahawks",),
    "TB": ("buccaneers", "bucs"),
    "TEN": ("titans",),
    "WAS": ("commanders",),
}


def _norm_path(url: str) -> str:
    path = urllib.parse.unquote(urllib.parse.urlparse(url).path).lower()
    return re.sub(r"[^a-z0-9]+", " ", path).strip()


def _matches_identity(url: str, team: str, opponent: str) -> bool:
    path = " " + _norm_path(url) + " "
    for key in (team, opponent):
        for token in FRANCHISE_IDENTITY[key]:
            t = re.sub(r"[^a-z0-9]+", " ", token.lower()).strip()
            if t and (" " + t + " ") in path:
                return True
    return False


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"row_count:{len(rows)}")
    direct = sum(bool(r.get("direct_ready")) for r in rows)
    if direct != EXPECTED_DIRECT:
        raise RuntimeError(f"direct_count:{direct}")
    if len(rows) - direct != EXPECTED_ARCHIVE_NEEDED:
        raise RuntimeError(f"archive_needed:{len(rows)-direct}")
    return rows


def _evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    direct = sum(bool(r.get("direct_ready")) for r in rows)
    rescued_rows: list[str] = []
    rescue_by_team: dict[str, int] = {}
    strict_capture_counts: dict[str, int] = {}

    for r in rows:
        if r.get("direct_ready"):
            continue
        team = str(r["team"])
        opp = str(r["opponent"])
        strict = [
            c for c in (r.get("captures") or [])
            if _matches_identity(str(c.get("url") or ""), team, opp)
        ]
        key = f'{r["season"]}|{r["week"]}|{team}'
        strict_capture_counts[key] = len(strict)
        if strict:
            rescued_rows.append(key)
            rescue_by_team[team] = rescue_by_team.get(team, 0) + 1

    rescued = len(rescued_rows)
    combined = direct + rescued
    gates = {
        "exact_rows_158": len(rows) == EXPECTED_ROWS,
        "exact_direct_123": direct == EXPECTED_DIRECT,
        "exact_archive_needed_35": len(rows) - direct == EXPECTED_ARCHIVE_NEEDED,
        "combined_ge_127": combined >= GATE_ROWS,
        "identity_vocab_32_teams": len(FRANCHISE_IDENTITY) == 32,
        "new_network_retrieval_false": True,
        "football_outcomes_read_zero": True,
        "model_residuals_read_zero": True,
        "sportsbook_inputs_used_false": True,
        "paid_oddsapi_calls_zero": True,
        "week5_2026_outcomes_read_zero": True,
        "football_predictive_models_fit_zero": True,
        "production_changed_false": True,
    }
    passes = all(bool(v) for v in gates.values())
    return {
        "version": VERSION,
        "disposition": (
            "STRICT_ARCHIVE_IDENTITY_RECERTIFICATION_PASSES_V1H_GATE"
            if passes else
            "STRICT_ARCHIVE_IDENTITY_RECERTIFICATION_FAILS_V1H_GATE"
        ),
        "rows": len(rows),
        "direct_ready_rows": direct,
        "archive_needed_rows": len(rows) - direct,
        "strict_archive_rescued_rows": rescued,
        "combined_transport_ready_rows": combined,
        "combined_transport_ready_rate": combined / len(rows) if rows else 0.0,
        "rescue_by_team": rescue_by_team,
        "rescued_row_keys": rescued_rows,
        "strict_capture_counts": strict_capture_counts,
        "gates": gates,
        "new_network_retrieval": False,
        "football_outcomes_read": 0,
        "model_residuals_read": 0,
        "sportsbook_inputs_used": False,
        "paid_oddsapi_calls": 0,
        "week5_2026_outcomes_read": 0,
        "football_predictive_models_fit": 0,
        "production_changed": False,
        "final_source_qualification_claimed": False,
        "predictive_candidate_authorized": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1i-rows", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args.v1i_rows)
    result = _evaluate(rows)
    (args.out_dir / "qb_intent_common_crawl_archive_transport_v1i_r1_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
