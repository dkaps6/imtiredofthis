#!/usr/bin/env python3
"""Validate QB First-Down Public Intent Source Audit V1 against its frozen plan.

This script is intentionally a SOURCE-QUALIFICATION validator only. It reads a
pregame source ledger plus an independently supplied regular-season schedule,
constructs the deterministic 2023-2025 sampled team-week universe, and applies
the preregistered coverage/timestamp/source-quality gates.

It MUST NOT read football outcomes, model residuals, sportsbook/game-market
fields, or construct predictive scores.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SEASONS = {2023, 2024, 2025}
SAMPLED_WEEKS = {2, 5, 8, 11, 14, 17}
DISPOSITIONS = {
    "ELIGIBLE_INTENT_SOURCE_FOUND",
    "PUBLIC_PREGAME_SOURCE_FOUND_NO_INTENT_CONTENT",
    "TIMESTAMP_UNSAFE_ONLY",
    "NO_RECONSTRUCTABLE_SOURCE",
}
TAGS = {
    "RUN_EMPHASIS",
    "PASS_EMPHASIS",
    "EARLY_DOWN_AGGRESSION",
    "TEMPO_CHANGE",
    "PROTECTION_DRIVEN_PLAN",
    "DEFENSIVE_MATCHUP_PLAN",
    "PERSONNEL_AVAILABILITY_PLAN",
    "OTHER_EXPLICIT_OFFENSIVE_INTENT",
}
STABLE_SOURCE_CLASSES = {"OFFICIAL", "LOCAL_ATTRIBUTABLE"}
FORBIDDEN_TOKENS = {
    "qb_yards", "passing_yards", "pass_yards", "qb_attempts", "attempts_actual",
    "wr_targets", "wr_receptions", "first_down_dbr", "residual", "model_error",
    "prop", "player_prop", "spread", "total", "moneyline", "sportsbook", "odds",
    "actual_result", "postgame", "final_score", "epa_result", "success_result",
}
REQUIRED_LEDGER = {
    "season", "week", "team", "opponent", "kickoff", "source_class", "publisher",
    "speaker", "publication_time", "locator", "availability_disposition",
    "semantic_tags", "evidence", "timestamp_safe",
}
REQUIRED_SCHEDULE = {"season", "week", "home_team", "away_team", "kickoff"}


def parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def parse_dt(v: str) -> datetime:
    s = str(v).strip().replace("Z", "+00:00")
    d = datetime.fromisoformat(s)
    if d.tzinfo is None:
        raise ValueError(f"timestamp lacks timezone: {v}")
    return d.astimezone(timezone.utc)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_forbidden_columns(cols: set[str]) -> list[str]:
    low = {c.lower() for c in cols}
    return sorted(c for c in low if any(tok in c for tok in FORBIDDEN_TOKENS))


def build_universe(schedule_rows: list[dict[str, str]]) -> dict[tuple[int, int, str], dict[str, str]]:
    universe: dict[tuple[int, int, str], dict[str, str]] = {}
    for r in schedule_rows:
        season = int(r["season"])
        week = int(r["week"])
        if season not in SEASONS or week not in SAMPLED_WEEKS:
            continue
        home, away = r["home_team"].strip(), r["away_team"].strip()
        kickoff = r["kickoff"].strip()
        for team, opp in ((home, away), (away, home)):
            key = (season, week, team)
            if key in universe:
                raise ValueError(f"duplicate schedule team-week: {key}")
            universe[key] = {"opponent": opp, "kickoff": kickoff}
    return universe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True, type=Path)
    ap.add_argument("--schedule", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    ledger = load_csv(args.ledger)
    schedule = load_csv(args.schedule)
    errors: list[str] = []

    ledger_cols = set(ledger[0]) if ledger else set()
    sched_cols = set(schedule[0]) if schedule else set()
    missing_ledger = REQUIRED_LEDGER - ledger_cols
    missing_sched = REQUIRED_SCHEDULE - sched_cols
    if missing_ledger:
        errors.append(f"missing ledger columns: {sorted(missing_ledger)}")
    if missing_sched:
        errors.append(f"missing schedule columns: {sorted(missing_sched)}")
    forbidden = check_forbidden_columns(ledger_cols)
    if forbidden:
        errors.append(f"forbidden outcome/market columns present: {forbidden}")
    if errors:
        result = {"disposition": "AUDIT_INVALID", "errors": errors, "production_changes": 0}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 2

    universe = build_universe(schedule)
    keyed: dict[tuple[int, int, str], dict[str, str]] = {}
    for i, r in enumerate(ledger, start=2):
        try:
            key = (int(r["season"]), int(r["week"]), r["team"].strip())
        except Exception as e:
            errors.append(f"ledger line {i}: invalid key: {e}")
            continue
        if key in keyed:
            errors.append(f"duplicate ledger team-week: {key}")
            continue
        keyed[key] = r
        if key not in universe:
            errors.append(f"ledger team-week outside frozen sample: {key}")
            continue
        if r["opponent"].strip() != universe[key]["opponent"]:
            errors.append(f"opponent mismatch {key}: {r['opponent']} != {universe[key]['opponent']}")
        disp = r["availability_disposition"].strip()
        if disp not in DISPOSITIONS:
            errors.append(f"invalid disposition {key}: {disp}")
        raw_tags = [x.strip() for x in r["semantic_tags"].split(";") if x.strip()]
        bad_tags = sorted(set(raw_tags) - TAGS)
        if bad_tags:
            errors.append(f"invalid semantic tags {key}: {bad_tags}")
        if len(r["evidence"].split()) > 25:
            errors.append(f"evidence exceeds 25 words {key}")
        if disp == "ELIGIBLE_INTENT_SOURCE_FOUND":
            if not r["locator"].strip() or not r["publisher"].strip():
                errors.append(f"eligible row missing locator/publisher {key}")
            if not parse_bool(r["timestamp_safe"]):
                errors.append(f"eligible row marked timestamp-unsafe {key}")
            try:
                pub = parse_dt(r["publication_time"])
                ko = parse_dt(universe[key]["kickoff"])
                if pub >= ko:
                    errors.append(f"publication not pre-kickoff {key}")
            except Exception as e:
                errors.append(f"timestamp parse failure {key}: {e}")

    missing = sorted(set(universe) - set(keyed))
    if missing:
        errors.append(f"deterministic sample incomplete: {len(missing)} team-weeks missing")

    total = len(universe)
    eligible_keys = [k for k, r in keyed.items() if k in universe and r["availability_disposition"].strip() == "ELIGIBLE_INTENT_SOURCE_FOUND"]
    eligible = len(eligible_keys)
    pooled_rate = eligible / total if total else 0.0

    season_total = Counter(k[0] for k in universe)
    season_elig = Counter(k[0] for k in eligible_keys)
    season_rates = {str(s): season_elig[s] / season_total[s] if season_total[s] else 0.0 for s in sorted(SEASONS)}

    week_total = Counter(k[1] for k in universe)
    week_elig = Counter(k[1] for k in eligible_keys)
    week_rates = {str(w): week_elig[w] / week_total[w] if week_total[w] else 0.0 for w in sorted(SAMPLED_WEEKS)}

    team_total = Counter(k[2] for k in universe)
    team_elig = Counter(k[2] for k in eligible_keys)
    qualifying_franchises = sum(1 for t, n in team_total.items() if n and team_elig[t] / n >= 0.50)

    eligible_rows = [keyed[k] for k in eligible_keys]
    ts_safe = sum(parse_bool(r["timestamp_safe"]) for r in eligible_rows)
    ts_rate = ts_safe / eligible if eligible else 0.0
    stable = sum(r["source_class"].strip().upper() in STABLE_SOURCE_CLASSES for r in eligible_rows)
    stable_rate = stable / eligible if eligible else 0.0

    gates = {
        "sample_complete": not missing,
        "pooled_eligible_ge_70pct": pooled_rate >= 0.70,
        "each_season_ge_60pct": all(v >= 0.60 for v in season_rates.values()),
        "each_sampled_week_ge_50pct": all(v >= 0.50 for v in week_rates.values()),
        "eligible_timestamp_safe_ge_90pct": ts_rate >= 0.90,
        "eligible_stable_source_ge_80pct": stable_rate >= 0.80,
        "franchises_ge_50pct_coverage_at_least_24": qualifying_franchises >= 24,
        "no_outcome_or_market_fields": not forbidden,
        "no_predictive_fit": True,
        "zero_production_changes": True,
    }
    qualified = not errors and all(gates.values())

    disposition_counts = Counter(r["availability_disposition"].strip() for r in keyed.values())
    tag_counts = Counter()
    for r in eligible_rows:
        tag_counts.update(x.strip() for x in r["semantic_tags"].split(";") if x.strip())

    result = {
        "disposition": "PUBLIC_INTENT_SOURCE_QUALIFIED" if qualified else "PUBLIC_INTENT_SOURCE_NOT_QUALIFIED",
        "frozen_sample_team_weeks": total,
        "ledger_rows": len(ledger),
        "eligible_team_weeks": eligible,
        "pooled_eligible_rate": pooled_rate,
        "season_rates": season_rates,
        "sampled_week_rates": week_rates,
        "eligible_timestamp_safe_rate": ts_rate,
        "eligible_stable_source_rate": stable_rate,
        "franchises_ge_50pct_coverage": qualifying_franchises,
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "semantic_tag_counts": dict(sorted(tag_counts.items())),
        "gates": gates,
        "errors": errors,
        "sportsbook_fields_used": 0,
        "predictive_models_fit": 0,
        "production_changes": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
