#!/usr/bin/env python3
"""Immutable-in-run pregame capture of a NEW public FantasyAlarm WR/CB article.

This captures *published projected alignments*, not observed coverage. No
sportsbook input, football-model fit, target-game outcomes, or editorial grades.
For durable proof beyond the GitHub artifact retention window, commit the
sanitized factual lock + hash ledger to GitHub BEFORE the eligible games.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup

from scripts.research.acquire_fantasyalarm_wr_cb_archive_v1 import (
    fetch_page, parse_page,
)
from scripts.research.audit_fantasyalarm_wr_cb_source_quality_v1 import _load_schedule


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("capture timestamps must be timezone aware")
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_url_and_title(url: str, html: str, season: int, week: int) -> None:
    u = urlparse(url)
    if (u.scheme, u.hostname) != ("https", "www.fantasyalarm.com"):
        raise ValueError("only exact HTTPS public FantasyAlarm articles allowed")
    if not u.path.startswith("/articles/nfl/wide-receivers/"):
        raise ValueError("only verified WR/CB article paths allowed")
    soup = BeautifulSoup(html, "html.parser")
    titles = [x.get_text(" ", strip=True) for x in soup.find_all("h1")]
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        titles.append(str(og["content"]))
    title = " ".join(titles)
    if not re.search(r"\b" + str(season) + r"\b", title):
        raise ValueError("source title does not verify season")
    if not re.search(r"\bweek[\s-]*0?" + str(week) + r"\b", title, re.I):
        raise ValueError("source title does not verify exact week")


def assess_capture_rows(
    rows: pd.DataFrame, schedule: pd.DataFrame, *,
    capture_started: datetime, capture_complete: datetime,
) -> pd.DataFrame:
    """Assess exact fetch timing per WR's actual kickoff, never week-level guess."""
    if capture_started.tzinfo is None or capture_complete.tzinfo is None:
        raise ValueError("capture timestamps must be timezone aware")
    if capture_complete < capture_started:
        raise ValueError("invalid reversed capture interval")
    if rows.empty:
        raise ValueError("no explicitly parsed WR/CB rows; fail closed")
    expected = {"season", "week", "wr_team", "opponent", "published_at_utc",
                "modified_at_utc", "modification_metadata_status"}
    if not expected.issubset(rows.columns):
        raise ValueError(f"source rows missing required columns: {sorted(expected - set(rows.columns))}")
    sched = schedule.rename(columns={"team": "wr_team"})
    required_schedule = {"season", "week", "wr_team", "kickoff_utc", "scheduled_opponent"}
    if not required_schedule.issubset(sched.columns):
        raise ValueError("schedule missing required team/week/kickoff columns")
    x = rows.merge(
        sched[list(required_schedule)],
        on=["season", "week", "wr_team"], how="left", validate="many_to_one",
    )
    kickoff = pd.to_datetime(x["kickoff_utc"], utc=True, errors="coerce")
    published = pd.to_datetime(x["published_at_utc"], utc=True, errors="coerce")
    modified = pd.to_datetime(x["modified_at_utc"], utc=True, errors="coerce")
    done = pd.Timestamp(capture_complete).tz_convert("UTC")
    x["schedule_match"] = (
        x["scheduled_opponent"].notna()
        & x["opponent"].astype(str).eq(x["scheduled_opponent"].astype(str))
    )
    status = pd.Series("PRE_KICKOFF_EXACT_HTML_CAPTURE", index=x.index, dtype="string")
    # Highest-severity provenance failures override any provisional evidence.
    status.loc[kickoff.isna() | ~x["schedule_match"]] = "QUARANTINE_SCHEDULE"
    status.loc[published.isna()] = "QUARANTINE_PUBLICATION_UNKNOWN"
    status.loc[published.gt(done)] = "QUARANTINE_PUBLICATION_AFTER_CAPTURE"
    status.loc[
        x["modification_metadata_status"].isin([
            "CONFLICTING_MODIFICATION_METADATA",
            "INVALID_MODIFICATION_METADATA",
        ])
    ] = "QUARANTINE_MODIFICATION_METADATA"
    status.loc[
        x["modification_metadata_status"].eq("UNAMBIGUOUS_MODIFICATION_METADATA")
        & modified.gt(done)
    ] = "QUARANTINE_METADATA_FUTURE_OF_CAPTURE"
    status.loc[kickoff.notna() & kickoff.le(done)] = "QUARANTINE_CAPTURE_NOT_PREGAME"
    x["capture_timing_status"] = status
    x["capture_started_at_utc"] = _iso(capture_started)
    x["capture_completed_at_utc"] = _iso(capture_complete)
    x["pregame_fact_snapshot_candidate"] = status.eq("PRE_KICKOFF_EXACT_HTML_CAPTURE")
    # This does NOT imply every team/receiver is covered, a verified GSIS join,
    # actual corner responsibility, model eligibility or forever-durable proof.
    x["model_feature_eligible"] = False
    return x


def run_capture(*, season: int, week: int, url: str, out_dir: Path) -> dict:
    if season < 2026 or not 1 <= week <= 18:
        raise ValueError("prospective capture restricted to 2026+ regular season")
    if out_dir.exists():
        raise FileExistsError("prospective snapshots are append-only: use unique out-dir")
    # Capture the interval *around the network fetch*, not just process time.
    started = datetime.now(timezone.utc)
    html = fetch_page(url)
    completed = datetime.now(timezone.utc)
    _validate_url_and_title(url, html, season, week)
    rows, page_audit = parse_page(html, season=season, week=week, source_url=url)
    schedule = _load_schedule([season])
    assessed = assess_capture_rows(
        rows, schedule, capture_started=started, capture_complete=completed,
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    html_bytes = html.encode("utf-8")
    source_hash = hashlib.sha256(html_bytes).hexdigest()
    (out_dir / "source_page.html").write_bytes(html_bytes)
    assessed["source_snapshot_sha256"] = source_hash
    assessed.to_csv(out_dir / "snapshot_rows_audit.csv", index=False)
    safe = assessed.loc[assessed["pregame_fact_snapshot_candidate"]].copy()
    fact_cols = [
        "season", "week", "source_url", "wr_clean_key", "wr_team",
        "opponent", "cb_clean_key", "alignment_bucket", "published_at_utc",
        "modified_at_utc", "capture_started_at_utc", "capture_completed_at_utc",
        "source_snapshot_sha256", "capture_timing_status",
    ]
    factual = safe[fact_cols].sort_values(
        ["season", "week", "wr_team", "wr_clean_key", "alignment_bucket", "cb_clean_key"]
    ).to_dict("records")
    facts_bytes = (json.dumps(factual, sort_keys=True, indent=2) + "\n").encode()
    (out_dir / "pregame_factual_lock.json").write_bytes(facts_bytes)
    summary = {
        "contract": "PUBLIC_WR_CB_PROSPECTIVE_EXACT_CAPTURE_V1",
        "season": season, "week": week, "source_url": url,
        "capture_started_at_utc": _iso(started),
        "capture_completed_at_utc": _iso(completed),
        "source_html_sha256": source_hash,
        "source_html_bytes": len(html_bytes),
        "factual_lock_sha256": hashlib.sha256(facts_bytes).hexdigest(),
        "observed_pairings": int(len(assessed)),
        "pregame_captured_pairings": int(len(safe)),
        "quarantined_pairings": int((~assessed["pregame_fact_snapshot_candidate"]).sum()),
        "row_status": assessed["capture_timing_status"].value_counts().to_dict(),
        "article_metadata": {
            "published_at_utc": page_audit["published_at_utc"],
            "modified_at_utc": page_audit["modified_at_utc"],
            "modification_metadata_status": page_audit["modification_metadata_status"],
        },
        "only_predicted_alignment_not_observed_coverage": True,
        "full_week_or_slot_coverage_certified": False,
        "identity_roster_quality_gate_cleared": False,
        "historical_model_source_gate_cleared": False,
        "editorial_grade_model_eligible": False,
        "sportsbook_inputs": False,
        "target_game_outcomes": False,
        "model_candidates_fit": 0,
        "durability_requirement": "Commit pregame_factual_lock.json and its SHA to an immutable GitHub commit BEFORE affected games. Actions artifact alone expires and is not durable.",
    }
    (out_dir / "capture_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True, indent=2))
    if safe.empty:
        raise RuntimeError("no pregame factual rows certified; only quarantined audit artifact generated")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, type=int)
    ap.add_argument("--week", required=True, type=int)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    run_capture(season=a.season, week=a.week, url=a.url, out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
