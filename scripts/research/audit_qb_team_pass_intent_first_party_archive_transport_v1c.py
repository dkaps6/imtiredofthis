#!/usr/bin/env python3
"""Source-only audit for direct first-party NFL club archive transport V1C.

No football outcomes, model residuals, sportsbook data, or predictive fitting.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

VERSION = "QB_TEAM_PASS_INTENT_FIRST_PARTY_ARCHIVE_TRANSPORT_V1C"
YEARS = (2023, 2024, 2025)
TOKENS = (
    "transcript",
    "press-conference",
    "press-conferences",
    "media-availability",
    "media_availability",
    "what-they-said",
    "quotes",
)
DOMAINS = {
    "ARI": "azcardinals.com",
    "ATL": "atlantafalcons.com",
    "BAL": "baltimoreravens.com",
    "BUF": "buffalobills.com",
    "CAR": "panthers.com",
    "CHI": "chicagobears.com",
    "CIN": "bengals.com",
    "CLE": "clevelandbrowns.com",
    "DAL": "dallascowboys.com",
    "DEN": "denverbroncos.com",
    "DET": "detroitlions.com",
    "GB": "packers.com",
    "HOU": "houstontexans.com",
    "IND": "colts.com",
    "JAX": "jaguars.com",
    "KC": "chiefs.com",
    "LV": "raiders.com",
    "LAC": "chargers.com",
    "LAR": "therams.com",
    "MIA": "miamidolphins.com",
    "MIN": "vikings.com",
    "NE": "patriots.com",
    "NO": "neworleanssaints.com",
    "NYG": "giants.com",
    "NYJ": "newyorkjets.com",
    "PHI": "philadelphiaeagles.com",
    "PIT": "steelers.com",
    "SF": "49ers.com",
    "SEA": "seahawks.com",
    "TB": "buccaneers.com",
    "TEN": "titansonline.com",
    "WAS": "commanders.com",
}
USER_AGENT = (
    "Mozilla/5.0 (compatible; NFL-research-source-audit/1.0; "
    "+https://github.com/dkaps6/imtiredofthis)"
)
MAX_BODY = 16 * 1024 * 1024
MAX_SITEMAPS_PER_TEAM = 40
MAX_URLS_PER_TEAM = 150_000


@dataclass
class TeamResult:
    team: str
    domain: str
    robots_status: str
    robots_sitemaps: list[str]
    sitemap_transport_reachable: bool
    parsed_sitemaps: int
    enumerated_urls: int
    candidate_urls: int
    candidate_years: list[int]
    candidate_examples: list[str]
    errors: list[str]


def _first_party(url: str, domain: str) -> bool:
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    domain = domain.lower().rstrip(".")
    return host == domain or host.endswith("." + domain)


def _read_url(url: str, timeout: float = 8.0) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/xml,text/xml,text/plain,text/html;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(MAX_BODY + 1)
        if len(raw) > MAX_BODY:
            raise RuntimeError("body_too_large")
        encoding = (resp.headers.get("Content-Encoding") or "").lower()
        if encoding == "gzip" or url.lower().endswith(".gz"):
            try:
                raw = gzip.decompress(raw)
            except OSError:
                pass
        return raw


def _robots(domain: str) -> tuple[str, list[str], list[str]]:
    url = f"https://www.{domain}/robots.txt"
    errors: list[str] = []
    try:
        raw = _read_url(url)
        text = raw.decode("utf-8", errors="replace")
        found = []
        for line in text.splitlines():
            if line.lower().startswith("sitemap:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate and _first_party(candidate, domain):
                    found.append(candidate)
        return "OK", sorted(set(found)), errors
    except Exception as exc:  # audit records fail-closed reason
        errors.append(f"robots:{type(exc).__name__}:{str(exc)[:180]}")
        return "ERROR", [], errors


def _tag_name(elem: ET.Element) -> str:
    return elem.tag.rsplit("}", 1)[-1].lower()


def _text_child(elem: ET.Element, name: str) -> str:
    for child in elem:
        if _tag_name(child) == name and child.text:
            return child.text.strip()
    return ""


def _parse_sitemap(raw: bytes) -> tuple[str, list[tuple[str, str]]]:
    root = ET.fromstring(raw)
    kind = _tag_name(root)
    if kind not in {"sitemapindex", "urlset"}:
        raise RuntimeError(f"unsupported_xml_root:{kind}")
    rows: list[tuple[str, str]] = []
    for child in root:
        loc = _text_child(child, "loc")
        if not loc:
            continue
        lastmod = _text_child(child, "lastmod")
        rows.append((loc, lastmod))
    return kind, rows


def _candidate(url: str) -> bool:
    path = urllib.parse.unquote(urllib.parse.urlparse(url).path).lower()
    return any(token in path for token in TOKENS)


def _discovery_year(url: str, lastmod: str) -> int | None:
    m = re.match(r"^(2023|2024|2025)(?:-|$)", (lastmod or "").strip())
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|[^0-9])(2023|2024|2025)(?:[^0-9]|$)", url)
    if m:
        return int(m.group(1))
    return None


def audit_team(team: str, domain: str) -> TeamResult:
    robots_status, robots_sitemaps, errors = _robots(domain)
    fallbacks = [
        f"https://www.{domain}/sitemap-index.xml",
        f"https://www.{domain}/sitemap.xml",
    ]
    queue: list[str] = []
    seen_sitemaps: set[str] = set()
    for url in [*robots_sitemaps, *fallbacks]:
        if url not in queue and _first_party(url, domain):
            queue.append(url)

    parsed = 0
    enumerated = 0
    candidates: dict[str, str] = {}
    transport = False

    while queue and parsed < MAX_SITEMAPS_PER_TEAM and enumerated < MAX_URLS_PER_TEAM:
        sitemap_url = queue.pop(0)
        if sitemap_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sitemap_url)
        if not _first_party(sitemap_url, domain):
            continue
        try:
            raw = _read_url(sitemap_url)
            kind, rows = _parse_sitemap(raw)
            parsed += 1
            transport = True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ET.ParseError, RuntimeError, OSError) as exc:
            errors.append(f"sitemap:{sitemap_url}:{type(exc).__name__}:{str(exc)[:160]}")
            continue
        except Exception as exc:
            errors.append(f"sitemap:{sitemap_url}:{type(exc).__name__}:{str(exc)[:160]}")
            continue

        if kind == "sitemapindex":
            for loc, _ in rows:
                if (
                    loc not in seen_sitemaps
                    and loc not in queue
                    and _first_party(loc, domain)
                    and len(seen_sitemaps) + len(queue) < MAX_SITEMAPS_PER_TEAM * 3
                ):
                    queue.append(loc)
            continue

        for loc, lastmod in rows:
            if enumerated >= MAX_URLS_PER_TEAM:
                break
            if not _first_party(loc, domain):
                continue
            enumerated += 1
            if _candidate(loc):
                candidates.setdefault(loc, lastmod)

    years = sorted(
        {
            year
            for url, lastmod in candidates.items()
            if (year := _discovery_year(url, lastmod)) in YEARS
        }
    )
    examples = sorted(candidates)[:8]
    return TeamResult(
        team=team,
        domain=domain,
        robots_status=robots_status,
        robots_sitemaps=robots_sitemaps,
        sitemap_transport_reachable=transport,
        parsed_sitemaps=parsed,
        enumerated_urls=enumerated,
        candidate_urls=len(candidates),
        candidate_years=years,
        candidate_examples=examples,
        errors=errors[:12],
    )


def evaluate(rows: Iterable[TeamResult]) -> dict:
    rows = list(rows)
    attempted = len(rows)
    transport = sum(r.sitemap_transport_reachable for r in rows)
    candidate_teams = sum(r.candidate_urls > 0 for r in rows)
    two_year = sum(len(r.candidate_years) >= 2 for r in rows)
    three_year = sum(len(r.candidate_years) == 3 for r in rows)
    gates = {
        "all_32_clubs_attempted": attempted == 32,
        "sitemap_transport_reachable_ge_24": transport >= 24,
        "candidate_url_teams_ge_24": candidate_teams >= 24,
        "candidate_two_of_three_year_teams_ge_20": two_year >= 20,
        "candidate_all_three_year_teams_ge_12": three_year >= 12,
        "external_search_engine_html_contacted": False,
        "football_outcomes_read": False,
        "model_residuals_read": False,
        "sportsbook_inputs_used": False,
        "week5_2026_outcomes_read": False,
        "predictive_models_fit_zero": True,
        "production_changed": False,
    }
    qualification_keys = [
        "all_32_clubs_attempted",
        "sitemap_transport_reachable_ge_24",
        "candidate_url_teams_ge_24",
        "candidate_two_of_three_year_teams_ge_20",
        "candidate_all_three_year_teams_ge_12",
    ]
    qualified = all(bool(gates[k]) for k in qualification_keys)
    return {
        "version": VERSION,
        "disposition": (
            "FIRST_PARTY_ARCHIVE_TRANSPORT_QUALIFIED_FOR_SEMANTIC_AUDIT"
            if qualified
            else "FIRST_PARTY_ARCHIVE_TRANSPORT_NOT_QUALIFIED"
        ),
        "attempted_clubs": attempted,
        "transport_reachable_clubs": transport,
        "candidate_url_clubs": candidate_teams,
        "candidate_two_of_three_year_clubs": two_year,
        "candidate_all_three_year_clubs": three_year,
        "candidate_years": list(YEARS),
        "candidate_tokens": list(TOKENS),
        "gates": gates,
        "sportsbook_inputs_used": False,
        "paid_oddsapi_calls": 0,
        "football_outcomes_read": 0,
        "model_residuals_read": 0,
        "week5_2026_outcomes_read": 0,
        "predictive_models_fit": 0,
        "production_changed": False,
        "timestamp_safe_rows_certified": 0,
        "predictive_candidate_authorized": False,
        "semantic_audit_authorized": qualified,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    results: list[TeamResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(audit_team, team, domain): team
            for team, domain in sorted(DOMAINS.items())
        }
        for fut in concurrent.futures.as_completed(futures):
            team = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append(
                    TeamResult(
                        team=team,
                        domain=DOMAINS[team],
                        robots_status="UNHANDLED_ERROR",
                        robots_sitemaps=[],
                        sitemap_transport_reachable=False,
                        parsed_sitemaps=0,
                        enumerated_urls=0,
                        candidate_urls=0,
                        candidate_years=[],
                        candidate_examples=[],
                        errors=[f"unhandled:{type(exc).__name__}:{str(exc)[:180]}"],
                    )
                )
    results.sort(key=lambda r: r.team)
    result = evaluate(results)
    result["elapsed_seconds"] = round(time.time() - started, 3)

    (args.out_dir / "qb_first_party_archive_transport_v1c_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "qb_first_party_archive_transport_v1c_teams.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2, sort_keys=True))
    for row in results:
        print(
            row.team,
            f"transport={row.sitemap_transport_reachable}",
            f"candidates={row.candidate_urls}",
            f"years={row.candidate_years}",
            f"errors={len(row.errors)}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
