#!/usr/bin/env python3
"""Build the blinded V1K manual semantic precision audit packet.

No football outcomes, residuals, sportsbook data, or predictive fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from scripts.research.audit_qb_team_pass_intent_first_party_archive_transport_v1c import (
    DOMAINS,
    _first_party,
)
from scripts.research.audit_qb_team_pass_intent_first_party_gold_validation_v1d import (
    _extract_safe_publication,
    _parse_absolute,
    _read_page,
)
from scripts.research.audit_qb_team_pass_intent_gold_semantic_extractor_v1e import (
    ACTION_RE,
    OFFENSE_RE,
    POSTGAME_RE,
    TEAM_SYNONYMS,
    _fragments,
    _visible,
)

VERSION = "QB_TEAM_PASS_INTENT_BLINDED_BOUNDED_MANUAL_PRECISION_AUDIT_V1K"
EXPECTED_V1G_ROWS = 536
EXPECTED_READY = 378
SAMPLE_SIZE = 32
SAMPLED_WEEKS = (2, 5, 8, 11, 14, 17)
SEASONS = (2023, 2024, 2025)


@dataclass
class PacketRow:
    blind_id: str
    season: int
    week: int
    team: str
    opponent: str
    kickoff: str
    locator: str
    publication_time: str
    timestamp_method: str
    page_title: str
    fetch_success: bool
    first_party: bool
    timestamp_safe_recheck: bool
    excerpts: list[str]
    errors: list[str]


def _row_key(r: dict[str, Any]) -> str:
    return f'{r["season"]}|{r["week"]}|{r["team"]}|{r["locator"]}'


def _row_hash(r: dict[str, Any]) -> str:
    return hashlib.sha256(_row_key(r).encode("utf-8")).hexdigest()


def _quartiles(rows: list[dict[str, Any]]) -> dict[str, int]:
    ordered = sorted(
        rows,
        key=lambda r: (float(r["semantic_margin"]), _row_hash(r)),
    )
    out: dict[str, int] = {}
    n = len(ordered)
    for i, r in enumerate(ordered):
        q = min(3, (4 * i) // n)
        out[_row_key(r)] = q
    return out


def _select(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if len(rows) != EXPECTED_READY:
        raise RuntimeError(f"expected {EXPECTED_READY} ready rows, got {len(rows)}")
    quart = _quartiles(rows)
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    tags: set[str] = set()
    season_weeks: set[tuple[int, int]] = set()
    teams: set[str] = set()
    qs: set[int] = set()

    while len(selected) < SAMPLE_SIZE:
        best: tuple[tuple[int, str], dict[str, Any], tuple[str, tuple[int, int], str, int]] | None = None
        for r in rows:
            key = _row_key(r)
            if key in selected_keys:
                continue
            features = (
                str(r["candidate_tag"]),
                (int(r["season"]), int(r["week"])),
                str(r["team"]),
                quart[key],
            )
            score = (
                1000 * int(features[0] not in tags)
                + 300 * int(features[1] not in season_weeks)
                + 100 * int(features[2] not in teams)
                + 50 * int(features[3] not in qs)
            )
            candidate = ((-score, _row_hash(r)), r, features)
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            raise RuntimeError("selection_exhausted")
        _, r, features = best
        selected.append(r)
        selected_keys.add(_row_key(r))
        tags.add(features[0])
        season_weeks.add(features[1])
        teams.add(features[2])
        qs.add(features[3])

    return selected, quart


def _opp_in_fragment(fragment: str, opponent: str) -> bool:
    text = fragment.lower()
    return any(s.lower() in text for s in TEAM_SYNONYMS[opponent])


def _review_excerpts(text: str, opponent: str) -> list[str]:
    ranked: list[tuple[int, int, int, str]] = []
    for idx, frag in enumerate(_fragments(text)):
        if not OFFENSE_RE.search(frag) or POSTGAME_RE.search(frag):
            continue
        action = int(bool(ACTION_RE.search(frag)))
        opp = int(_opp_in_fragment(frag, opponent))
        ranked.append((-action, -opp, idx, " ".join(frag.split())))
    ranked.sort()
    return [x[3] for x in ranked[:18]]


def _packet_row(r: dict[str, Any], blind_id: str) -> PacketRow:
    errors: list[str] = []
    fetch = first = ts_safe = False
    title = ""
    excerpts: list[str] = []
    try:
        raw, final_url = _read_page(str(r["locator"]))
        fetch = True
        first = bool(
            _first_party(str(r["locator"]), DOMAINS[str(r["team"])])
            and _first_party(final_url, DOMAINS[str(r["team"])])
        )
        kickoff = _parse_absolute(str(r["kickoff"]))
        if kickoff is None:
            raise RuntimeError("non_absolute_kickoff")
        _method, _value, parseable, pre = _extract_safe_publication(raw, kickoff)
        ts_safe = bool(parseable and pre)
        title, text = _visible(raw)
        excerpts = _review_excerpts(text, str(r["opponent"]))
    except Exception as exc:
        errors.append(f"{type(exc).__name__}:{str(exc)[:180]}")

    return PacketRow(
        blind_id=blind_id,
        season=int(r["season"]),
        week=int(r["week"]),
        team=str(r["team"]),
        opponent=str(r["opponent"]),
        kickoff=str(r["kickoff"]),
        locator=str(r["locator"]),
        publication_time=str(r["publication_time"]),
        timestamp_method=str(r["timestamp_method"]),
        page_title=title,
        fetch_success=fetch,
        first_party=first,
        timestamp_safe_recheck=ts_safe,
        excerpts=excerpts,
        errors=errors,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1g-rows", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(args.v1g_rows.read_text(encoding="utf-8"))
    if len(rows) != EXPECTED_V1G_ROWS:
        raise RuntimeError(f"v1g_rows:{len(rows)}")
    ready = [r for r in rows if r["screen_state"] == "OFFICIAL_AUTO_REVIEW_READY"]
    if len(ready) != EXPECTED_READY:
        raise RuntimeError(f"ready_rows:{len(ready)}")

    selected, quart = _select(ready)
    packet: list[PacketRow] = []
    hidden: list[dict[str, Any]] = []
    for i, r in enumerate(selected, start=1):
        blind = f"B{i:02d}"
        packet.append(_packet_row(r, blind))
        hidden.append({
            "blind_id": blind,
            "row_key": _row_key(r),
            "row_hash": _row_hash(r),
            "candidate_tag": r["candidate_tag"],
            "positive_score": r["positive_score"],
            "negative_score": r["negative_score"],
            "semantic_margin": r["semantic_margin"],
            "margin_quartile": quart[_row_key(r)],
            "v1g_evidence": r["evidence"],
        })

    tags = {x["candidate_tag"] for x in hidden}
    season_weeks = {(p.season, p.week) for p in packet}
    teams = {p.team for p in packet}
    qs = {x["margin_quartile"] for x in hidden}
    summary = {
        "version": VERSION,
        "sample_size": len(packet),
        "tags_represented": sorted(tags),
        "tag_count": len(tags),
        "season_week_cells_represented": sorted(f"{s}|{w}" for s, w in season_weeks),
        "season_week_cell_count": len(season_weeks),
        "franchises_represented": sorted(teams),
        "franchise_count": len(teams),
        "margin_quartiles_represented": sorted(qs),
        "margin_quartile_count": len(qs),
        "selection_gates": {
            "exact_sample_32": len(packet) == 32,
            "all_7_tags": len(tags) == 7,
            "all_18_season_week_cells": len(season_weeks) == 18,
            "all_4_margin_quartiles": len(qs) == 4,
            "at_least_24_franchises": len(teams) >= 24,
        },
        "football_outcomes_read": 0,
        "model_residuals_read": 0,
        "sportsbook_inputs_used": False,
        "paid_oddsapi_calls": 0,
        "week5_2026_outcomes_read": 0,
        "football_predictive_models_fit": 0,
        "production_changed": False,
    }

    if not all(summary["selection_gates"].values()):
        raise RuntimeError(f"sample coverage gate failed: {summary}")

    (args.out_dir / "qb_intent_v1k_blinded_packet.json").write_text(
        json.dumps([asdict(p) for p in packet], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "qb_intent_v1k_hidden_selection_key.json").write_text(
        json.dumps(hidden, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "qb_intent_v1k_packet_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
