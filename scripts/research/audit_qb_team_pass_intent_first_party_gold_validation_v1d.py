#!/usr/bin/env python3
"""V1D operational gold validation for QB first-party intent sources.

Source/schema only. No football outcomes, model residuals, sportsbook inputs, or
predictive fitting.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import json
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from scripts.research.audit_qb_team_pass_intent_first_party_archive_transport_v1c import (
    DOMAINS,
    MAX_BODY,
    MAX_SITEMAPS_PER_TEAM,
    MAX_URLS_PER_TEAM,
    USER_AGENT,
    _first_party,
    _parse_sitemap,
    _read_url,
    _robots,
)

VERSION = "QB_TEAM_PASS_INTENT_FIRST_PARTY_GOLD_VALIDATION_V1D"
GOLD_PATH = Path("docs/research/QB_TEAM_PASS_INTENT_V1D_GOLD_PREFIX.csv")
LOCAL_EXPECTED_TEAM = "ATL"


@dataclass
class GoldRowResult:
    team: str
    source_class: str
    canonical_url: str
    expected_timestamp_safe: bool
    route: str
    sitemap_transport_reachable: bool
    parsed_sitemaps: int
    enumerated_urls: int
    canonical_reacquired: bool
    direct_fetch_success: bool
    final_url: str
    official_source_class_correct: bool
    timestamp_method: str
    timestamp_value: str
    timestamp_parseable_absolute: bool
    timestamp_pre_kickoff: bool
    timestamp_safe_correct: bool
    errors: list[str]


class _MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: list[dict[str, str]] = []
        self.times: list[str] = []
        self.ldjson: list[str] = []
        self._in_ldjson = False
        self._script_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = {str(k).lower(): (v or "") for k, v in attrs}
        if tag.lower() == "meta":
            self.meta.append(d)
        elif tag.lower() == "time" and d.get("datetime"):
            self.times.append(d["datetime"].strip())
        elif tag.lower() == "script":
            typ = d.get("type", "").lower().split(";", 1)[0].strip()
            if typ == "application/ld+json":
                self._in_ldjson = True
                self._script_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_ldjson:
            self._script_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._in_ldjson:
            self.ldjson.append("".join(self._script_parts).strip())
            self._in_ldjson = False
            self._script_parts = []


def _norm_url(url: str) -> str:
    p = urllib.parse.urlsplit(url.strip())
    host = (p.hostname or "").lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    path = urllib.parse.unquote(p.path or "/")
    path = re.sub(r"/+", "/", path)
    if path != "/":
        path = path.rstrip("/")
    return host + path


def _enumerate_first_party_urls(domain: str) -> tuple[bool, int, set[str], list[str]]:
    robots_status, declared, errors = _robots(domain)
    _ = robots_status
    queue: list[str] = []
    seen_sitemaps: set[str] = set()
    urls: set[str] = set()
    for u in [*declared, f"https://www.{domain}/sitemap-index.xml", f"https://www.{domain}/sitemap.xml"]:
        if u not in queue and _first_party(u, domain):
            queue.append(u)

    parsed = 0
    while queue and parsed < MAX_SITEMAPS_PER_TEAM and len(urls) < MAX_URLS_PER_TEAM:
        sitemap = queue.pop(0)
        if sitemap in seen_sitemaps:
            continue
        seen_sitemaps.add(sitemap)
        if not _first_party(sitemap, domain):
            continue
        try:
            raw = _read_url(sitemap)
            kind, rows = _parse_sitemap(raw)
            parsed += 1
        except Exception as exc:
            errors.append(f"sitemap:{sitemap}:{type(exc).__name__}:{str(exc)[:160]}")
            continue
        if kind == "sitemapindex":
            for loc, _lastmod in rows:
                if (
                    _first_party(loc, domain)
                    and loc not in seen_sitemaps
                    and loc not in queue
                    and len(seen_sitemaps) + len(queue) < MAX_SITEMAPS_PER_TEAM * 3
                ):
                    queue.append(loc)
        else:
            for loc, _lastmod in rows:
                if len(urls) >= MAX_URLS_PER_TEAM:
                    break
                if _first_party(loc, domain):
                    urls.add(_norm_url(loc))
    return parsed > 0, parsed, urls, errors[:12]


def _read_page(url: str, timeout: float = 12.0) -> tuple[bytes, str]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(MAX_BODY + 1)
        if len(raw) > MAX_BODY:
            raise RuntimeError("body_too_large")
        encoding = (resp.headers.get("Content-Encoding") or "").lower()
        if encoding == "gzip":
            try:
                raw = gzip.decompress(raw)
            except OSError:
                pass
        return raw, resp.geturl()


def _walk_dates(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            k = str(key).lower()
            if k in {"datepublished", "date_published"} and isinstance(val, (str, int, float)):
                out.append(str(val).strip())
            out.extend(_walk_dates(val))
    elif isinstance(obj, list):
        for val in obj:
            out.extend(_walk_dates(val))
    return out


def _timestamp_candidates(raw: bytes) -> list[tuple[str, str]]:
    text = raw.decode("utf-8", errors="replace")
    parser = _MetaParser()
    try:
        parser.feed(text)
    except Exception:
        pass

    out: list[tuple[str, str]] = []
    for blob in parser.ldjson:
        if not blob:
            continue
        try:
            obj = json.loads(blob)
            for value in _walk_dates(obj):
                out.append(("JSON_LD_datePublished", value))
        except Exception:
            for m in re.finditer(
                r'["\\\']datePublished["\\\']\s*:\s*["\\\']([^"\\\']+)["\\\']',
                blob,
                flags=re.I,
            ):
                out.append(("JSON_LD_datePublished_regex", m.group(1).strip()))

    preferred_meta = {
        "article:published_time",
        "datepublished",
        "date_published",
        "publishdate",
        "publish_date",
        "publication_date",
        "parsely-pub-date",
        "sailthru.date",
    }
    for d in parser.meta:
        key = (d.get("property") or d.get("name") or d.get("itemprop") or "").strip().lower()
        value = (d.get("content") or "").strip()
        if value and key in preferred_meta:
            out.append((f"META_{key}", value))

    for value in parser.times:
        if value:
            out.append(("TIME_datetime", value))

    dedup: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in out:
        if pair not in seen:
            seen.add(pair)
            dedup.append(pair)
    return dedup


def _parse_absolute(value: str) -> datetime | None:
    s = value.strip()
    if not s:
        return None
    if re.fullmatch(r"\d{10}(?:\.\d+)?", s):
        try:
            return datetime.fromtimestamp(float(s)).astimezone()
        except Exception:
            return None
    if re.fullmatch(r"\d{13}", s):
        try:
            return datetime.fromtimestamp(int(s) / 1000.0).astimezone()
        except Exception:
            return None
    s = s.replace("Z", "+00:00").replace("z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt


def _extract_safe_publication(raw: bytes, kickoff: datetime) -> tuple[str, str, bool, bool]:
    for method, value in _timestamp_candidates(raw):
        dt = _parse_absolute(value)
        if dt is None:
            continue
        return method, value, True, dt < kickoff
    return "", "", False, False


def _bool(s: str) -> bool:
    return str(s).strip().lower() == "true"


def _load_gold(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if len(rows) != 13:
        raise RuntimeError(f"expected exact 13-row gold fixture, got {len(rows)}")
    teams = [r["team"] for r in rows]
    if len(set(teams)) != 13:
        raise RuntimeError("duplicate team in gold fixture")
    return rows


def audit_official(row: dict[str, str]) -> GoldRowResult:
    team = row["team"]
    domain = DOMAINS[team]
    canonical = row["locator"].strip()
    errors: list[str] = []
    transport = False
    parsed = 0
    urls: set[str] = set()
    try:
        transport, parsed, urls, enum_errors = _enumerate_first_party_urls(domain)
        errors.extend(enum_errors)
    except Exception as exc:
        errors.append(f"enumeration:{type(exc).__name__}:{str(exc)[:180]}")

    reacquired = _norm_url(canonical) in urls
    fetch_ok = False
    final_url = ""
    class_ok = False
    method = ""
    ts_value = ""
    parseable = False
    pre = False
    try:
        raw, final_url = _read_page(canonical)
        fetch_ok = True
        class_ok = _first_party(canonical, domain) and _first_party(final_url, domain)
        kickoff = _parse_absolute(row["kickoff"])
        if kickoff is None:
            raise RuntimeError("gold kickoff is not absolute")
        method, ts_value, parseable, pre = _extract_safe_publication(raw, kickoff)
    except Exception as exc:
        errors.append(f"fetch_or_timestamp:{type(exc).__name__}:{str(exc)[:180]}")

    expected_safe = _bool(row["timestamp_safe"])
    correct = expected_safe == bool(parseable and pre)
    return GoldRowResult(
        team=team,
        source_class=row["source_class"],
        canonical_url=canonical,
        expected_timestamp_safe=expected_safe,
        route="OFFICIAL_FIRST_PARTY",
        sitemap_transport_reachable=transport,
        parsed_sitemaps=parsed,
        enumerated_urls=len(urls),
        canonical_reacquired=reacquired,
        direct_fetch_success=fetch_ok,
        final_url=final_url,
        official_source_class_correct=class_ok,
        timestamp_method=method,
        timestamp_value=ts_value,
        timestamp_parseable_absolute=parseable,
        timestamp_pre_kickoff=pre,
        timestamp_safe_correct=correct,
        errors=errors[:12],
    )


def evaluate(rows: list[dict[str, str]], audited: list[GoldRowResult]) -> dict[str, Any]:
    official_gold = [r for r in rows if r["source_class"] == "OFFICIAL"]
    local_gold = [r for r in rows if r["source_class"] == "LOCAL_ATTRIBUTABLE"]
    by_team = {r.team: r for r in audited}
    official_results = [by_team[r["team"]] for r in official_gold]

    reacquired = sum(r.canonical_reacquired for r in official_results)
    fetch_ok = sum(r.direct_fetch_success for r in official_results)
    timestamp_correct = sum(r.timestamp_safe_correct for r in official_results)
    source_correct = sum(r.official_source_class_correct for r in official_results)
    atl_route = (
        len(local_gold) == 1
        and local_gold[0]["team"] == LOCAL_EXPECTED_TEAM
        and LOCAL_EXPECTED_TEAM not in by_team
    )

    gates = {
        "exact_gold_rows_13": len(rows) == 13,
        "exact_source_split_12_official_1_local": len(official_gold) == 12 and len(local_gold) == 1,
        "official_canonical_reacquired_ge_11_of_12": reacquired >= 11,
        "official_direct_fetch_12_of_12": fetch_ok == 12,
        "timestamp_safe_correct_12_of_12": timestamp_correct == 12,
        "official_source_class_correct_12_of_12": source_correct == 12,
        "atl_local_fallback_routed": atl_route,
        "external_search_engine_html_contacted": False,
        "football_outcomes_read": False,
        "model_residuals_read": False,
        "sportsbook_inputs_used": False,
        "week5_2026_outcomes_read": False,
        "predictive_models_fit_zero": True,
        "production_changed": False,
    }
    required = [
        "exact_gold_rows_13",
        "exact_source_split_12_official_1_local",
        "official_canonical_reacquired_ge_11_of_12",
        "official_direct_fetch_12_of_12",
        "timestamp_safe_correct_12_of_12",
        "official_source_class_correct_12_of_12",
        "atl_local_fallback_routed",
    ]
    qualified = all(bool(gates[k]) for k in required)
    return {
        "version": VERSION,
        "disposition": (
            "FIRST_PARTY_GOLD_VALIDATION_QUALIFIED_FOR_SEMANTIC_COVERAGE_AUDIT"
            if qualified
            else "FIRST_PARTY_GOLD_VALIDATION_NOT_QUALIFIED"
        ),
        "gold_rows": len(rows),
        "official_gold_rows": len(official_gold),
        "local_gold_rows": len(local_gold),
        "official_canonical_reacquired": reacquired,
        "official_direct_fetch_success": fetch_ok,
        "timestamp_safe_correct": timestamp_correct,
        "official_source_class_correct": source_correct,
        "local_fallback_teams": [r["team"] for r in local_gold],
        "gates": gates,
        "external_search_engine_html_contacted": False,
        "football_outcomes_read": 0,
        "model_residuals_read": 0,
        "sportsbook_inputs_used": False,
        "paid_oddsapi_calls": 0,
        "week5_2026_outcomes_read": 0,
        "predictive_models_fit": 0,
        "production_changed": False,
        "predictive_candidate_authorized": False,
        "semantic_coverage_audit_authorized": qualified,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", type=Path, default=GOLD_PATH)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_gold(args.gold)
    official = [r for r in rows if r["source_class"] == "OFFICIAL"]
    audited: list[GoldRowResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(audit_official, r): r["team"] for r in official}
        for fut in concurrent.futures.as_completed(futures):
            audited.append(fut.result())
    audited.sort(key=lambda x: x.team)

    result = evaluate(rows, audited)
    (args.out_dir / "qb_first_party_gold_validation_v1d_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "qb_first_party_gold_validation_v1d_rows.json").write_text(
        json.dumps([asdict(r) for r in audited], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    for r in audited:
        print(
            r.team,
            f"reacquired={r.canonical_reacquired}",
            f"fetch={r.direct_fetch_success}",
            f"source={r.official_source_class_correct}",
            f"timestamp={r.timestamp_safe_correct}",
            f"method={r.timestamp_method}",
            f"errors={len(r.errors)}",
        )
    print(f"{LOCAL_EXPECTED_TEAM} route=LOCAL_FALLBACK_REQUIRED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
