#!/usr/bin/env python3
"""Bounded FREE archive INDEX discovery, not data acquisition or model science.

Search only documented URL-index metadata for four high-risk FantasyAlarm
article vintages. A matching pregame timestamp is merely a candidate: indexed
URLs do not prove the archived body includes the originally forecast pairings.
Record every network error instead of treating a blocked index as empty.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from scripts.research.audit_fantasyalarm_wr_cb_source_quality_v1 import _load_schedule

TARGETS = ((2023, 8), (2024, 1), (2025, 4), (2025, 14))
WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
CC_COLLECTIONS = "https://index.commoncrawl.org/collinfo.json"
HEADERS = {"User-Agent": "NFL-research-source-provenance/1.0 (public metadata only)"}


def parse_ts(raw: str) -> datetime | None:
    text = str(raw).strip()
    if not re.fullmatch(r"\d{14}", text):
        return None
    try:
        return datetime.strptime(text, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def same_source_url(original: str, discovered: str) -> bool:
    """Ignore scheme and www aliases, NOT article path or article ID."""
    a, b = urlparse(original), urlparse(discovered)
    h = lambda p: (p.hostname or "").lower().removeprefix("www.")
    return h(a) == h(b) and a.path.rstrip("/") == b.path.rstrip("/")


def classify_index_ts(raw: str, first: datetime, last: datetime) -> str:
    capture = parse_ts(raw)
    if not capture:
        return "INVALID_INDEX_TIMESTAMP"
    if capture < first:
        return "INDEX_PRE_FIRST_KICKOFF_CANDIDATE_ONLY"
    if capture < last:
        return "INDEX_BETWEEN_WEEK_GAMES_REQUIRES_PER_WR_CHECK"
    return "INDEX_AFTER_FINAL_WEEK_KICKOFF_NOT_PREGAME"


def request_json(url: str, *, params: dict | None = None) -> tuple[object | None, str]:
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=12)
        if response.status_code != 200:
            return None, f"HTTP_{response.status_code}"
        return response.json(), "HTTP_200_JSON"
    except requests.RequestException as exc:
        return None, "NETWORK_" + type(exc).__name__
    except ValueError:
        return None, "NOT_JSON_RESPONSE"


def wayback_index(source_url: str, season: int) -> tuple[list[dict], str]:
    # Exact URL only, no wildcard archive scraping.
    result, status = request_json(WAYBACK_CDX, params={
        "url": source_url, "output": "json", "filter": "statuscode:200",
        "from": str(season), "to": str(season), "limit": "30",
    })
    if status != "HTTP_200_JSON":
        return [], status
    if not isinstance(result, list):
        return [], "INVALID_CDX_SCHEMA"
    if not result:
        return [], "NO_INDEX_MATCH_FOR_QUERY_NOT_PROOF_OF_ABSENCE"
    header = result[0]
    if not isinstance(header, list) or not {"timestamp", "original"}.issubset(set(header)):
        return [], "INVALID_CDX_SCHEMA"
    matches = []
    for row in result[1:]:
        if not isinstance(row, list) or len(row) != len(header):
            continue
        fields = dict(zip(header, row))
        if not same_source_url(source_url, str(fields.get("original", ""))):
            continue
        if str(fields.get("statuscode", "")) != "200":
            continue
        matches.append({"timestamp": str(fields["timestamp"]), "original": str(fields["original"]),
                        "digest": str(fields.get("digest", ""))})
    return matches, "MATCHING_INDEX_RECORDS" if matches else "NO_EXACT_INDEX_MATCH_NOT_PROOF_OF_ABSENCE"


def choose_cc_indices(collections: object, first: datetime) -> list[str]:
    if not isinstance(collections, list):
        return []
    scored = []
    for coll in collections:
        if not isinstance(coll, dict):
            continue
        cid = str(coll.get("id", ""))
        m = re.fullmatch(r"CC-MAIN-(\d{4})-(\d{2})", cid)
        if not m or int(m.group(1)) != first.year:
            continue
        distance = abs(int(m.group(2)) - first.isocalendar().week)
        if distance <= 7:
            scored.append((distance, cid))
    return [cid for _, cid in sorted(scored)[:3]]


def cc_index(source_url: str, crawl_id: str) -> tuple[list[dict], str]:
    url = f"https://index.commoncrawl.org/{crawl_id}-index"
    result, status = request_json(url, params={
        "url": source_url, "output": "json", "filter": "status:200", "limit": "30",
    })
    # CC returns newline-delimited JSON, not a JSON array. If HTTP is 200
    # but response.json() failed, fetch that *same one bounded URL* as text.
    if status == "NOT_JSON_RESPONSE":
        try:
            response = requests.get(url, params={
                "url": source_url, "output": "json", "filter": "status:200", "limit": "30",
            }, headers=HEADERS, timeout=12)
            if response.status_code != 200:
                return [], f"HTTP_{response.status_code}"
            result = [json.loads(line) for line in response.text.splitlines() if line.strip()]
            status = "HTTP_200_NDJSON"
        except requests.RequestException as exc:
            return [], "NETWORK_" + type(exc).__name__
        except ValueError:
            return [], "INVALID_CDX_NDJSON"
    if status not in {"HTTP_200_JSON", "HTTP_200_NDJSON"}:
        return [], status
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, list):
        return [], "INVALID_CDX_SCHEMA"
    matches = []
    for row in result:
        if not isinstance(row, dict):
            continue
        original = str(row.get("url", ""))
        if same_source_url(source_url, original) and str(row.get("status", "")) == "200":
            matches.append({"timestamp": str(row.get("timestamp", "")),
                            "original": original, "digest": str(row.get("digest", "")),
                            "warc_locator_present": all(k in row for k in ("filename", "offset", "length"))})
    return matches, "MATCHING_INDEX_RECORDS" if matches else "NO_EXACT_INDEX_MATCH_NOT_PROOF_OF_ABSENCE"


def run(manifest_path: Path, out_dir: Path) -> dict:
    manifest = pd.read_csv(manifest_path)
    required = {"season", "week", "url"}
    if not required.issubset(manifest.columns):
        raise ValueError("source manifest has no exact season/week/url columns")
    schedule = _load_schedule(sorted({season for season, _ in TARGETS}))
    coll, coll_status = request_json(CC_COLLECTIONS)
    results = []
    for season, week in TARGETS:
        src = manifest.loc[manifest["season"].eq(season) & manifest["week"].eq(week), "url"]
        if len(src) != 1:
            raise RuntimeError(f"manifest must identify exactly one article {season} W{week}")
        source_url = str(src.iloc[0])
        dates = pd.to_datetime(
            schedule.loc[schedule["season"].eq(season) & schedule["week"].eq(week), "kickoff_utc"],
            utc=True, errors="coerce",
        ).dropna().sort_values()
        if dates.empty:
            raise RuntimeError(f"schedule missing {season} W{week}")
        first = dates.iloc[0].to_pydatetime()
        last = dates.iloc[-1].to_pydatetime()
        matches, status = wayback_index(source_url, season)
        lookup = [{"index": "WAYBACK_CDX", "query_status": status}]
        candidate = []
        for row in matches:
            candidate.append({"index": "WAYBACK_CDX", **row,
                              "index_time_class": classify_index_ts(row["timestamp"], first, last)})
        time.sleep(0.3)
        for cid in choose_cc_indices(coll, first):
            cc_matches, cc_status = cc_index(source_url, cid)
            lookup.append({"index": cid, "query_status": cc_status})
            candidate.extend({"index": cid, **row,
                              "index_time_class": classify_index_ts(row["timestamp"], first, last)}
                             for row in cc_matches)
            time.sleep(0.3)
        results.append({
            "season": season, "week": week, "source_url": source_url,
            "first_game_utc": first.isoformat(), "last_game_utc": last.isoformat(),
            "lookups": lookup, "index_candidates": candidate,
            "pregame_index_candidate_count": sum(
                r["index_time_class"] == "INDEX_PRE_FIRST_KICKOFF_CANDIDATE_ONLY" for r in candidate),
            "partial_week_index_candidate_count": sum(
                r["index_time_class"] == "INDEX_BETWEEN_WEEK_GAMES_REQUIRES_PER_WR_CHECK" for r in candidate),
            "source_body_verified": False,
            "historical_factual_rows_verified": 0,
            "caution": "Archive index alone NEVER proves pregame table contents, identity or alignment.",
        })
    output = {
        "contract": "WR_CB_HISTORICAL_PUBLIC_INDEX_DISCOVERY_V1",
        "metadata_only": True, "bounded_articles": len(TARGETS),
        "common_crawl_collections_lookup": coll_status,
        "source_reacquisition": False, "raw_archive_html_downloaded": False,
        "nfl_outcomes_used": False, "sportsbook_inputs": False, "parameters_fit": 0,
        "source_gate_cleared": False, "index_query_failures_are_not_absence_evidence": True,
        "targets": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "snapshot_index_candidates.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    run(args.manifest, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
