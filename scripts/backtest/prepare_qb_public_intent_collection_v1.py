#!/usr/bin/env python3
"""Prepare and validate working state for QB Public Intent Source Audit V1.

This helper is intentionally operational only. It never searches the web, assigns a
source disposition, reads football outcomes, or invokes the predictive audit. Its
purpose is to make the frozen collection protocol mechanically auditable:

* preserve the deterministic manifest order (season, week, team);
* distinguish PENDING_REVIEW from completed negative dispositions;
* forbid outcome/market/model fields in working state;
* require a contiguous completed prefix so rows cannot be skipped/reprioritized;
* require the frozen official->local search sequence before negative finalization;
* export a final validator ledger only when every frozen team-week is complete.

The canonical scientific validator remains
``audit_qb_first_down_public_intent_source_v1.py``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PENDING = "PENDING_REVIEW"
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
FORBIDDEN_TOKENS = {
    "qb_yards", "passing_yards", "pass_yards", "qb_attempts", "attempts_actual",
    "wr_targets", "wr_receptions", "first_down_dbr", "residual", "model_error",
    "prop", "player_prop", "spread", "total", "moneyline", "sportsbook", "odds",
    "actual_result", "postgame", "final_score", "epa_result", "success_result",
}
CORE = ["season", "week", "team", "opponent", "kickoff"]
WORK_FIELDS = CORE + [
    "review_status",
    "official_search_complete",
    "local_search_complete",
    "official_candidate_locators",
    "local_candidate_locators",
    "source_class",
    "publisher",
    "speaker",
    "publication_time",
    "locator",
    "availability_disposition",
    "semantic_tags",
    "evidence",
    "timestamp_safe",
    "duplicate_locators",
    "notes",
]
LEDGER_FIELDS = CORE + [
    "source_class", "publisher", "speaker", "publication_time", "locator",
    "availability_disposition", "semantic_tags", "evidence", "timestamp_safe",
]


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def yes(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def key(r: dict[str, str]) -> tuple[int, int, str]:
    return int(r["season"]), int(r["week"]), r["team"].strip().upper()


def reject_forbidden(fields: set[str]) -> None:
    bad = sorted(c for c in fields if any(tok in c.lower() for tok in FORBIDDEN_TOKENS))
    if bad:
        raise RuntimeError(f"forbidden outcome/market/model fields present: {bad}")


def init(manifest: Path, out: Path) -> None:
    rows = load(manifest)
    if not rows:
        raise RuntimeError("empty manifest")
    reject_forbidden(set(rows[0]))
    required = set(CORE) | {"review_status"}
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"manifest missing fields: {sorted(missing)}")
    if rows != sorted(rows, key=key):
        raise RuntimeError("manifest is not in frozen ascending (season, week, team) order")
    if len({key(r) for r in rows}) != len(rows):
        raise RuntimeError("duplicate team-week in manifest")

    out_rows: list[dict[str, str]] = []
    for r in rows:
        if r["review_status"].strip() != PENDING:
            raise RuntimeError(f"manifest row already dispositioned: {key(r)}")
        row = {f: "" for f in WORK_FIELDS}
        for f in CORE:
            row[f] = r[f].strip()
        row["review_status"] = PENDING
        row["official_search_complete"] = "false"
        row["local_search_complete"] = "false"
        out_rows.append(row)
    write(out, WORK_FIELDS, out_rows)
    print(f"initialized_rows={len(out_rows)}")
    print("sportsbook_fields_used=0 predictive_models_fit=0 production_changes=0")


def validate_row(r: dict[str, str]) -> None:
    k = key(r)
    status = r["review_status"].strip()
    if status == PENDING:
        if r["availability_disposition"].strip():
            raise RuntimeError(f"pending row has final disposition: {k}")
        return
    if status != "COMPLETE":
        raise RuntimeError(f"invalid review_status {k}: {status}")

    disp = r["availability_disposition"].strip()
    if disp not in DISPOSITIONS:
        raise RuntimeError(f"completed row lacks frozen disposition {k}: {disp}")
    if not yes(r["official_search_complete"]):
        raise RuntimeError(f"completed row lacks completed official search: {k}")

    tags = [x.strip() for x in r["semantic_tags"].split(";") if x.strip()]
    bad_tags = sorted(set(tags) - TAGS)
    if bad_tags:
        raise RuntimeError(f"invalid semantic tags {k}: {bad_tags}")
    if len(r["evidence"].split()) > 25:
        raise RuntimeError(f"evidence exceeds 25 words: {k}")

    if disp == "ELIGIBLE_INTENT_SOURCE_FOUND":
        required = ["source_class", "publisher", "publication_time", "locator"]
        missing = [f for f in required if not r[f].strip()]
        if missing:
            raise RuntimeError(f"eligible row missing {missing}: {k}")
        if not yes(r["timestamp_safe"]):
            raise RuntimeError(f"eligible row not timestamp-safe: {k}")
    else:
        # Under the frozen protocol local fallback is required after official
        # searching is exhausted without an eligible intent source. A negative
        # final disposition therefore cannot be assigned before both stages.
        if not yes(r["local_search_complete"]):
            raise RuntimeError(f"negative row finalized before local fallback completed: {k}")


def check(work: Path, export_ledger: Path | None) -> None:
    rows = load(work)
    if not rows:
        raise RuntimeError("empty working collection")
    reject_forbidden(set(rows[0]))
    missing = set(WORK_FIELDS) - set(rows[0])
    if missing:
        raise RuntimeError(f"working state missing fields: {sorted(missing)}")
    if rows != sorted(rows, key=key):
        raise RuntimeError("working rows are not in frozen ascending order")
    if len({key(r) for r in rows}) != len(rows):
        raise RuntimeError("duplicate team-week in working state")

    saw_pending = False
    completed = 0
    for r in rows:
        validate_row(r)
        is_pending = r["review_status"].strip() == PENDING
        if is_pending:
            saw_pending = True
        else:
            completed += 1
            if saw_pending:
                raise RuntimeError(
                    f"non-contiguous collection: completed row after pending row: {key(r)}"
                )

    print(f"rows={len(rows)} completed_prefix={completed} pending={len(rows)-completed}")
    print("sportsbook_fields_used=0 predictive_models_fit=0 production_changes=0")

    if export_ledger is not None:
        if completed != len(rows):
            raise RuntimeError(
                f"refusing final ledger export: {len(rows)-completed} team-weeks remain pending"
            )
        ledger = [{f: r[f] for f in LEDGER_FIELDS} for r in rows]
        write(export_ledger, LEDGER_FIELDS, ledger)
        print(f"exported_final_ledger_rows={len(ledger)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("--manifest", required=True, type=Path)
    p_init.add_argument("--out", required=True, type=Path)

    p_check = sub.add_parser("check")
    p_check.add_argument("--work", required=True, type=Path)
    p_check.add_argument("--export-ledger", type=Path)

    args = ap.parse_args()
    if args.cmd == "init":
        init(args.manifest, args.out)
    else:
        check(args.work, args.export_ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
