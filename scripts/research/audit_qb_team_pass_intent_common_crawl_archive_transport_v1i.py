#!/usr/bin/env python3
"""V1I Common Crawl archive transport audit for unresolved QB intent rows.

Source/archive discovery only. Consumes the exact repaired V1H artifact and does
not read football outcomes, residuals, sportsbook data, or fit a football model.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scripts.research.audit_qb_team_pass_intent_gold_semantic_extractor_v1e import TEAM_SYNONYMS
from scripts.research.audit_qb_team_pass_intent_local_fallback_direct_transport_v1h import (
    EXPECTED_UNRESOLVED,
    LOCAL_SOURCES,
    _same_domain,
    _token_match,
    _norm_path,
)

VERSION = "QB_TEAM_PASS_INTENT_COMMON_CRAWL_ARCHIVE_TRANSPORT_V1I"
COLLINFO_URL = "https://index.commoncrawl.org/collinfo.json"
EXPECTED_DIRECT_READY = 123
EXPECTED_ARCHIVE_NEEDED = 35
COMBINED_GATE_ROWS = 127
WINDOW_BEFORE_DAYS = 9
WINDOW_AFTER_DAYS = 90
QUERY_LIMIT = 200
DEFAULT_INTERVAL = 0.60
USER_AGENT = (
    "NFL-source-audit/1.0 archive-transport-research "
    "(https://github.com/dkaps6/imtiredofthis)"
)
RETRY_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Collection:
    id: str
    cdx_api: str
    start: str
    end: str


@dataclass(frozen=True)
class ArchiveCapture:
    collection: str
    timestamp: str
    url: str
    mime: str
    mime_detected: str
    status: str
    filename: str
    offset: int
    length: int


@dataclass
class ArchiveRow:
    season: int
    week: int
    team: str
    opponent: str
    kickoff: str
    direct_ready: bool
    mapped_publishers: list[str]
    archive_ready: bool
    rescued_by_archive: bool
    capture_count: int
    captures: list[dict[str, Any]]
    query_errors: list[str]


class ThrottledHTTP:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self.last_request_at = 0.0
        self.calls = 0

    def _sleep(self) -> None:
        elapsed = time.monotonic() - self.last_request_at
        need = self.min_interval - elapsed
        if need > 0:
            time.sleep(need)

    def get(self, url: str, *, timeout: float = 25.0) -> bytes:
        backoffs = (0.0, 2.0, 5.0, 10.0)
        last_exc: Exception | None = None
        for attempt, backoff in enumerate(backoffs):
            if backoff:
                time.sleep(backoff)
            self._sleep()
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json,text/plain;q=0.9,*/*;q=0.5",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read(8 * 1024 * 1024 + 1)
                    if len(raw) > 8 * 1024 * 1024:
                        raise RuntimeError("response_too_large")
                    self.calls += 1
                    self.last_request_at = time.monotonic()
                    return raw
            except urllib.error.HTTPError as exc:
                self.calls += 1
                self.last_request_at = time.monotonic()
                last_exc = exc
                if exc.code not in RETRY_CODES or attempt == len(backoffs) - 1:
                    raise
            except (urllib.error.URLError, TimeoutError) as exc:
                self.calls += 1
                self.last_request_at = time.monotonic()
                last_exc = exc
                if attempt == len(backoffs) - 1:
                    raise
        assert last_exc is not None
        raise last_exc


def _dt(value: str) -> datetime:
    s = str(value).strip().replace("Z", "+00:00")
    out = datetime.fromisoformat(s)
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def _capture_dt(value: str) -> datetime | None:
    s = str(value or "").strip()
    if not re.fullmatch(r"\d{14}", s):
        return None
    try:
        return datetime.strptime(s, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _load_v1h(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != EXPECTED_UNRESOLVED:
        raise RuntimeError(f"v1h_unresolved_count:{len(rows)}")
    direct = sum(bool(r.get("has_reachable_publisher")) for r in rows)
    needed = len(rows) - direct
    if direct != EXPECTED_DIRECT_READY or needed != EXPECTED_ARCHIVE_NEEDED:
        raise RuntimeError(f"v1h_transport_counts:{direct}:{needed}")
    for r in rows:
        team = str(r["team"])
        if list(r.get("mapped_publishers") or []) != list(LOCAL_SOURCES[team]):
            raise RuntimeError(f"publisher_map_drift:{team}")
    return rows


def _load_collections(http: ThrottledHTTP) -> list[Collection]:
    raw = http.get(COLLINFO_URL)
    payload = json.loads(raw.decode("utf-8"))
    out: list[Collection] = []
    for item in payload:
        ident = str(item.get("id") or "")
        api = str(item.get("cdx-api") or "")
        start = str(item.get("from") or "")
        end = str(item.get("to") or "")
        if not ident or not api or not start or not end:
            continue
        if not api.startswith("https://index.commoncrawl.org/"):
            continue
        out.append(Collection(ident, api, start, end))
    if not out:
        raise RuntimeError("no_common_crawl_collections")
    return out


def _overlapping_collections(
    collections: list[Collection],
    start: datetime,
    end: datetime,
) -> list[Collection]:
    rows = [
        c for c in collections
        if _dt(c.start) <= end and _dt(c.end) >= start
    ]
    rows.sort(key=lambda c: _dt(c.start))
    return rows


def _query_tokens(team: str, opponent: str) -> list[str]:
    tokens: set[str] = set()
    for value in (*TEAM_SYNONYMS[team], *TEAM_SYNONYMS[opponent]):
        for token in re.findall(r"[a-z0-9]+", str(value).lower()):
            if len(token) >= 4 and not token.isdigit():
                tokens.add(token)
    return sorted(tokens)


def _url_regex(team: str, opponent: str) -> str:
    tokens = _query_tokens(team, opponent)
    if not tokens:
        raise RuntimeError(f"no_query_tokens:{team}:{opponent}")
    return r".*(?:" + "|".join(re.escape(t) for t in tokens) + r").*"


def _query_url(
    collection: Collection,
    domain: str,
    start: datetime,
    end: datetime,
    team: str,
    opponent: str,
    *,
    collapse: bool = True,
) -> str:
    params: list[tuple[str, str]] = [
        ("url", domain),
        ("matchType", "domain"),
        ("output", "json"),
        ("from", start.strftime("%Y%m%d%H%M%S")),
        ("to", end.strftime("%Y%m%d%H%M%S")),
        ("filter", "=status:200"),
        ("filter", "~url:" + _url_regex(team, opponent)),
        ("limit", str(QUERY_LIMIT)),
    ]
    if collapse:
        params.append(("collapse", "urlkey"))
    return collection.cdx_api + "?" + urllib.parse.urlencode(params)


def _parse_ndjson(raw: bytes) -> list[dict[str, Any]]:
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _mime_ok(record: dict[str, Any]) -> bool:
    vals = [
        str(record.get("mime") or "").lower(),
        str(record.get("mime-detected") or record.get("mime_detected") or "").lower(),
    ]
    vals = [v for v in vals if v]
    if not vals:
        return False
    return any("html" in v or "xhtml" in v for v in vals)


def _capture_from_record(
    record: dict[str, Any],
    *,
    collection: str,
    domain: str,
    team: str,
    opponent: str,
    start: datetime,
    end: datetime,
) -> ArchiveCapture | None:
    url = str(record.get("url") or "")
    if not url or not _same_domain(url, domain):
        return None
    ts = str(record.get("timestamp") or "")
    capture_dt = _capture_dt(ts)
    if capture_dt is None or not (start <= capture_dt <= end):
        return None
    if str(record.get("status") or "") != "200":
        return None
    if not _mime_ok(record):
        return None
    path = _norm_path(url)
    if not (
        _token_match(path, TEAM_SYNONYMS[team])
        or _token_match(path, TEAM_SYNONYMS[opponent])
    ):
        return None
    filename = str(record.get("filename") or "")
    try:
        offset = int(record.get("offset"))
        length = int(record.get("length"))
    except (TypeError, ValueError):
        return None
    if not filename or offset < 0 or length <= 0:
        return None
    return ArchiveCapture(
        collection=collection,
        timestamp=ts,
        url=url,
        mime=str(record.get("mime") or ""),
        mime_detected=str(record.get("mime-detected") or record.get("mime_detected") or ""),
        status=str(record.get("status") or ""),
        filename=filename,
        offset=offset,
        length=length,
    )


def _query_collection(
    http: ThrottledHTTP,
    collection: Collection,
    domain: str,
    start: datetime,
    end: datetime,
    team: str,
    opponent: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    url = _query_url(collection, domain, start, end, team, opponent, collapse=True)
    try:
        return _parse_ndjson(http.get(url)), errors
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            errors.append(f"{collection.id}:{domain}:HTTP_{exc.code}")
            return [], errors
    except Exception as exc:
        errors.append(f"{collection.id}:{domain}:{type(exc).__name__}:{str(exc)[:120]}")
        return [], errors

    # Generic compatibility retry when an index does not support collapse.
    try:
        url = _query_url(collection, domain, start, end, team, opponent, collapse=False)
        return _parse_ndjson(http.get(url)), errors
    except Exception as exc:
        errors.append(f"{collection.id}:{domain}:no_collapse:{type(exc).__name__}:{str(exc)[:120]}")
        return [], errors


def _audit_row(
    row: dict[str, Any],
    collections: list[Collection],
    http: ThrottledHTTP,
) -> ArchiveRow:
    direct = bool(row.get("has_reachable_publisher"))
    mapped = list(row["mapped_publishers"])
    if direct:
        return ArchiveRow(
            season=int(row["season"]), week=int(row["week"]), team=str(row["team"]),
            opponent=str(row["opponent"]), kickoff=str(row["kickoff"]),
            direct_ready=True, mapped_publishers=mapped, archive_ready=False,
            rescued_by_archive=False, capture_count=0, captures=[], query_errors=[],
        )

    kickoff = _dt(str(row["kickoff"]))
    start = kickoff - timedelta(days=WINDOW_BEFORE_DAYS)
    end = kickoff + timedelta(days=WINDOW_AFTER_DAYS)
    relevant_collections = _overlapping_collections(collections, start, end)

    captures: dict[tuple[str, str, str], ArchiveCapture] = {}
    errors: list[str] = []
    for domain in mapped:
        for coll in relevant_collections:
            records, errs = _query_collection(
                http, coll, domain, start, end, str(row["team"]), str(row["opponent"])
            )
            errors.extend(errs)
            for rec in records:
                cap = _capture_from_record(
                    rec, collection=coll.id, domain=domain,
                    team=str(row["team"]), opponent=str(row["opponent"]),
                    start=start, end=end,
                )
                if cap is None:
                    continue
                key = (cap.url, cap.timestamp, cap.collection)
                captures.setdefault(key, cap)

    ordered = sorted(captures.values(), key=lambda x: (x.timestamp, x.url, x.collection))
    return ArchiveRow(
        season=int(row["season"]), week=int(row["week"]), team=str(row["team"]),
        opponent=str(row["opponent"]), kickoff=str(row["kickoff"]),
        direct_ready=False, mapped_publishers=mapped,
        archive_ready=bool(ordered), rescued_by_archive=bool(ordered),
        capture_count=len(ordered),
        captures=[asdict(x) for x in ordered[:20]],
        query_errors=errors[:30],
    )


def _evaluate(rows: list[ArchiveRow], *, http_calls: int) -> dict[str, Any]:
    direct = sum(r.direct_ready for r in rows)
    archive_needed = sum(not r.direct_ready for r in rows)
    rescued = sum(r.rescued_by_archive for r in rows)
    combined = sum(r.direct_ready or r.archive_ready for r in rows)

    archive_domains_valid = True
    for r in rows:
        for cap in r.captures:
            if not any(_same_domain(cap["url"], d) for d in r.mapped_publishers):
                archive_domains_valid = False

    gates = {
        "exact_unresolved_universe_158": len(rows) == EXPECTED_UNRESOLVED,
        "exact_direct_ready_baseline_123": direct == EXPECTED_DIRECT_READY,
        "exact_archive_needed_35": archive_needed == EXPECTED_ARCHIVE_NEEDED,
        "combined_transport_ready_ge_127": combined >= COMBINED_GATE_ROWS,
        "frozen_publisher_map_unchanged": all(
            r.mapped_publishers == list(LOCAL_SOURCES[r.team]) for r in rows
        ),
        "archive_records_on_frozen_domains": archive_domains_valid,
        "publisher_access_control_bypass_attempted_false": True,
        "football_outcomes_read_zero": True,
        "model_residuals_read_zero": True,
        "sportsbook_inputs_used_false": True,
        "paid_oddsapi_calls_zero": True,
        "week5_2026_outcomes_read_zero": True,
        "football_predictive_models_fit_zero": True,
        "production_changed_false": True,
    }
    ready = all(bool(v) for v in gates.values())
    rescue_by_team: dict[str, int] = {}
    for r in rows:
        if r.rescued_by_archive:
            rescue_by_team[r.team] = rescue_by_team.get(r.team, 0) + 1

    return {
        "version": VERSION,
        "disposition": (
            "COMMON_CRAWL_ARCHIVE_TRANSPORT_RESCUES_V1H_GATE"
            if ready else
            "COMMON_CRAWL_ARCHIVE_TRANSPORT_DOES_NOT_RESCUE_V1H_GATE"
        ),
        "unresolved_rows": len(rows),
        "direct_ready_rows": direct,
        "archive_needed_rows": archive_needed,
        "archive_rescued_rows": rescued,
        "combined_transport_ready_rows": combined,
        "combined_transport_ready_rate": combined / len(rows) if rows else 0.0,
        "rescue_by_team": rescue_by_team,
        "common_crawl_http_calls": http_calls,
        "gates": gates,
        "publisher_access_control_bypass_attempted": False,
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
    ap.add_argument("--v1h-rows", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-request-interval", type=float, default=DEFAULT_INTERVAL)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    input_rows = _load_v1h(args.v1h_rows)
    http = ThrottledHTTP(args.min_request_interval)
    collections = _load_collections(http)

    rows: list[ArchiveRow] = []
    for row in input_rows:
        rows.append(_audit_row(row, collections, http))
    rows.sort(key=lambda r: (r.season, r.week, r.team))

    summary = _evaluate(rows, http_calls=http.calls)
    (args.out_dir / "qb_intent_common_crawl_archive_transport_v1i_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "qb_intent_common_crawl_archive_transport_v1i_rows.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
