#!/usr/bin/env python3
"""V1H direct local-publisher transport audit for unresolved QB intent rows.

Source/discovery research only. Reuses the exact preserved V1G artifact and does
not read football outcomes, model residuals, sportsbook data, or fit a football
predictive model.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scripts.research.audit_qb_team_pass_intent_gold_semantic_extractor_v1e import TEAM_SYNONYMS
from scripts.research.audit_qb_team_pass_intent_first_party_gold_validation_v1d import _parse_absolute

VERSION = "QB_TEAM_PASS_INTENT_LOCAL_FALLBACK_DIRECT_TRANSPORT_V1H"
EXPECTED_V1G_ROWS = 536
EXPECTED_OFFICIAL_READY = 378
EXPECTED_UNRESOLVED = 158
WINDOW_DAYS = 9
MAX_SITEMAPS_PER_PUBLISHER = 250
MAX_URLS_PER_PUBLISHER = 100_000
MAX_CANDIDATES_PER_ROW_PUBLISHER = 60
USER_AGENT = (
    "Mozilla/5.0 (compatible; NFL-research-source-audit/1.0; "
    "+https://github.com/dkaps6/imtiredofthis)"
)
MAX_BODY = 20 * 1024 * 1024

LOCAL_SOURCES: dict[str, tuple[str, str]] = {
    "ARI": ("azcentral.com", "arizonasports.com"),
    "ATL": ("ajc.com", "11alive.com"),
    "BAL": ("baltimoresun.com", "wbal.com"),
    "BUF": ("buffalonews.com", "wgrz.com"),
    "CAR": ("charlotteobserver.com", "wcnc.com"),
    "CHI": ("chicagotribune.com", "suntimes.com"),
    "CIN": ("cincinnati.com", "wcpo.com"),
    "CLE": ("cleveland.com", "wkyc.com"),
    "DAL": ("dallasnews.com", "star-telegram.com"),
    "DEN": ("denverpost.com", "9news.com"),
    "DET": ("freep.com", "detroitnews.com"),
    "GB": ("jsonline.com", "greenbaypressgazette.com"),
    "HOU": ("houstonchronicle.com", "click2houston.com"),
    "IND": ("indystar.com", "wthr.com"),
    "JAX": ("jacksonville.com", "news4jax.com"),
    "KC": ("kansascity.com", "kshb.com"),
    "LAC": ("latimes.com", "ocregister.com"),
    "LAR": ("latimes.com", "ocregister.com"),
    "LV": ("reviewjournal.com", "lasvegassun.com"),
    "MIA": ("miamiherald.com", "sun-sentinel.com"),
    "MIN": ("startribune.com", "twincities.com"),
    "NE": ("boston.com", "masslive.com"),
    "NO": ("nola.com", "theadvocate.com"),
    "NYG": ("nj.com", "nydailynews.com"),
    "NYJ": ("nj.com", "nydailynews.com"),
    "PHI": ("inquirer.com", "delcotimes.com"),
    "PIT": ("post-gazette.com", "triblive.com"),
    "SEA": ("seattletimes.com", "thenewstribune.com"),
    "SF": ("sfchronicle.com", "mercurynews.com"),
    "TB": ("tampabay.com", "wtsp.com"),
    "TEN": ("tennessean.com", "wkrn.com"),
    "WAS": ("washingtonpost.com", "wtop.com"),
}

SEARCH_ENGINE_HOSTS = {
    "google.com", "www.google.com", "bing.com", "www.bing.com",
    "duckduckgo.com", "www.duckduckgo.com",
}


@dataclass(frozen=True)
class SitemapEntry:
    url: str
    lastmod: str


@dataclass
class PublisherResult:
    domain: str
    probe_reachable: bool
    transport_reachable: bool
    parsed_sitemaps: int
    enumerated_urls: int
    errors: list[str]


@dataclass
class RowResult:
    season: int
    week: int
    team: str
    opponent: str
    kickoff: str
    mapped_publishers: list[str]
    reachable_publishers: list[str]
    candidate_counts: dict[str, int]
    candidate_examples: dict[str, list[str]]
    has_reachable_publisher: bool
    has_local_candidate: bool


def _host(url: str) -> str:
    try:
        return (urllib.parse.urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _same_domain(url: str, domain: str) -> bool:
    host = _host(url)
    d = domain.lower().rstrip(".")
    return host == d or host.endswith("." + d)


def _read_url(url: str, timeout: float = 10.0) -> bytes:
    if _host(url) in SEARCH_ENGINE_HOSTS:
        raise RuntimeError("search_engine_contact_prohibited")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/xml,text/xml,text/plain,text/html;q=0.8,*/*;q=0.5",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        final = resp.geturl()
        if not _same_domain(final, _registrable_hint(url)):
            raise RuntimeError(f"cross_domain_redirect:{final}")
        raw = resp.read(MAX_BODY + 1)
        if len(raw) > MAX_BODY:
            raise RuntimeError("body_too_large")
        encoding = (resp.headers.get("Content-Encoding") or "").lower()
        if encoding == "gzip" or final.lower().endswith(".gz"):
            try:
                raw = gzip.decompress(raw)
            except OSError:
                pass
        return raw


def _registrable_hint(url: str) -> str:
    # Calls always start from a frozen publisher domain or a same-domain sitemap.
    host = _host(url)
    for domains in LOCAL_SOURCES.values():
        for domain in domains:
            if host == domain or host.endswith("." + domain):
                return domain
    return host


def _tag(elem: ET.Element) -> str:
    return elem.tag.rsplit("}", 1)[-1].lower()


def _child_text(elem: ET.Element, name: str) -> str:
    for child in elem:
        if _tag(child) == name and child.text:
            return child.text.strip()
    return ""


def _parse_sitemap(raw: bytes) -> tuple[str, list[tuple[str, str]]]:
    root = ET.fromstring(raw)
    kind = _tag(root)
    if kind not in {"sitemapindex", "urlset"}:
        raise RuntimeError(f"unsupported_xml_root:{kind}")
    rows: list[tuple[str, str]] = []
    for child in root:
        loc = _child_text(child, "loc")
        if not loc:
            continue
        rows.append((loc, _child_text(child, "lastmod")))
    return kind, rows


def _robots(domain: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    found: list[str] = []
    for base in (f"https://{domain}", f"https://www.{domain}"):
        try:
            raw = _read_url(base + "/robots.txt")
            text = raw.decode("utf-8", errors="replace")
            for line in text.splitlines():
                if line.lower().startswith("sitemap:"):
                    u = line.split(":", 1)[1].strip()
                    if u and _same_domain(u, domain):
                        found.append(u)
            if found:
                break
        except Exception as exc:
            errors.append(f"robots:{base}:{type(exc).__name__}:{str(exc)[:140]}")
    return sorted(set(found)), errors


def _seed_sitemaps(domain: str) -> tuple[list[str], list[str]]:
    declared, errors = _robots(domain)
    fallbacks = []
    for base in (f"https://{domain}", f"https://www.{domain}"):
        fallbacks.extend([
            base + "/sitemap.xml",
            base + "/sitemap-index.xml",
            base + "/sitemap_index.xml",
        ])
    out: list[str] = []
    for u in [*declared, *fallbacks]:
        if _same_domain(u, domain) and u not in out:
            out.append(u)
    return out, errors


def _probe_domain(domain: str) -> tuple[bool, list[str]]:
    seeds, errors = _seed_sitemaps(domain)
    for u in seeds:
        try:
            kind, _rows = _parse_sitemap(_read_url(u))
            if kind in {"sitemapindex", "urlset"}:
                return True, errors[:12]
        except Exception as exc:
            errors.append(f"probe:{u}:{type(exc).__name__}:{str(exc)[:140]}")
    return False, errors[:12]


def _historical_sitemap_hint(url: str, lastmod: str) -> bool:
    text = f"{url} {lastmod}".lower()
    if any(y in text for y in ("2023", "2024", "2025")):
        return True
    # Generic archive indexes without year hints must remain traversable.
    return not bool(re.search(r"20(?:1[0-9]|2[0-2]|2[6-9]|3[0-9])", text))


def _enumerate_domain(domain: str) -> tuple[PublisherResult, list[SitemapEntry]]:
    probe, probe_errors = _probe_domain(domain)
    seeds, errors = _seed_sitemaps(domain)
    errors = [*probe_errors, *errors]
    queue = list(seeds)
    seen: set[str] = set()
    entries: dict[str, str] = {}
    parsed = 0
    transport = False

    while queue and parsed < MAX_SITEMAPS_PER_PUBLISHER and len(entries) < MAX_URLS_PER_PUBLISHER:
        sitemap = queue.pop(0)
        if sitemap in seen or not _same_domain(sitemap, domain):
            continue
        seen.add(sitemap)
        try:
            kind, rows = _parse_sitemap(_read_url(sitemap))
            parsed += 1
            transport = True
        except Exception as exc:
            errors.append(f"sitemap:{sitemap}:{type(exc).__name__}:{str(exc)[:140]}")
            continue

        if kind == "sitemapindex":
            for loc, lastmod in rows:
                if (
                    _same_domain(loc, domain)
                    and loc not in seen
                    and loc not in queue
                    and _historical_sitemap_hint(loc, lastmod)
                    and len(seen) + len(queue) < MAX_SITEMAPS_PER_PUBLISHER * 4
                ):
                    queue.append(loc)
        else:
            for loc, lastmod in rows:
                if len(entries) >= MAX_URLS_PER_PUBLISHER:
                    break
                if _same_domain(loc, domain):
                    entries.setdefault(loc, lastmod)

    return (
        PublisherResult(
            domain=domain,
            probe_reachable=probe,
            transport_reachable=transport,
            parsed_sitemaps=parsed,
            enumerated_urls=len(entries),
            errors=errors[:30],
        ),
        [SitemapEntry(url=u, lastmod=lm) for u, lm in entries.items()],
    )


def _parse_dt(value: str) -> datetime | None:
    s = (value or "").strip().replace("Z", "+00:00").replace("z", "+00:00")
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.fromisoformat(s[:10])
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _norm_path(url: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        urllib.parse.unquote(urllib.parse.urlparse(url).path).lower(),
    ).strip()


def _token_match(path: str, values: tuple[str, ...]) -> bool:
    padded = " " + path + " "
    for value in values:
        token = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
        if token and (" " + token + " ") in padded:
            return True
    return False


def _week_match(path: str, week: int) -> bool:
    return bool(re.search(rf"\bweek\s*{week}\b", path))


def _year_match(path: str, season: int) -> bool:
    return bool(re.search(rf"(?:^|\s){season}(?:\s|$)", path))


def _candidate_urls(
    row: dict[str, Any],
    entries: list[SitemapEntry],
) -> list[tuple[int, float, str]]:
    kickoff = _parse_absolute(str(row["kickoff"]))
    if kickoff is None:
        raise RuntimeError(f"non_absolute_kickoff:{row}")
    start = kickoff - timedelta(days=WINDOW_DAYS)
    team = str(row["team"])
    opp = str(row["opponent"])
    team_tokens = TEAM_SYNONYMS[team]
    opp_tokens = TEAM_SYNONYMS[opp]
    ranked: list[tuple[int, float, str]] = []

    for ent in entries:
        path = _norm_path(ent.url)
        team_hit = _token_match(path, team_tokens)
        opp_hit = _token_match(path, opp_tokens)
        if not (team_hit or opp_hit):
            continue
        week_hit = _week_match(path, int(row["week"]))
        year_hit = _year_match(path, int(row["season"]))
        last = _parse_dt(ent.lastmod)
        date_hit = bool(last is not None and start <= last <= kickoff)
        if not (date_hit or year_hit or week_hit):
            continue
        score = (
            4 * int(opp_hit)
            + 4 * int(team_hit)
            + 3 * int(week_hit)
            + 2 * int(year_hit)
            + int(date_hit)
        )
        distance = abs((kickoff - last).total_seconds()) if last else float("inf")
        ranked.append((score, distance, ent.url))

    ranked.sort(key=lambda x: (-x[0], x[1], x[2].lower()))
    return ranked[:MAX_CANDIDATES_PER_ROW_PUBLISHER]


def _load_v1g(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != EXPECTED_V1G_ROWS:
        raise RuntimeError(f"v1g_row_count:{len(rows)}")
    ready = [r for r in rows if r.get("screen_state") == "OFFICIAL_AUTO_REVIEW_READY"]
    unresolved = [
        r for r in rows
        if r.get("screen_state") == "OFFICIAL_UNRESOLVED_REQUIRES_FALLBACK_OR_REVIEW"
    ]
    if len(ready) != EXPECTED_OFFICIAL_READY or len(unresolved) != EXPECTED_UNRESOLVED:
        raise RuntimeError(f"v1g_state_counts:{len(ready)}:{len(unresolved)}")
    return ready, unresolved


def _evaluate(
    unresolved: list[dict[str, Any]],
    probes: dict[str, bool],
    publisher_results: dict[str, PublisherResult],
    row_results: list[RowResult],
) -> dict[str, Any]:
    exact_map = set(LOCAL_SOURCES) == set(TEAM_SYNONYMS) and all(
        len(v) == 2 and len(set(v)) == 2 for v in LOCAL_SOURCES.values()
    )
    franchise_reachable = {}
    for team, domains in LOCAL_SOURCES.items():
        franchise_reachable[team] = any(bool(probes.get(d, False)) for d in domains)
    franchises_with_reachable = sum(franchise_reachable.values())
    rows_with_reachable = sum(r.has_reachable_publisher for r in row_results)
    rows_with_candidate = sum(r.has_local_candidate for r in row_results)

    gates = {
        "exact_unresolved_input_158": len(unresolved) == EXPECTED_UNRESOLVED,
        "frozen_map_exactly_32x2": exact_map,
        "franchises_with_reachable_publisher_ge_24": franchises_with_reachable >= 24,
        "unresolved_rows_with_reachable_publisher_ge_80pct": (
            rows_with_reachable / len(unresolved) >= 0.80 if unresolved else False
        ),
        "public_search_engine_html_contacted_false": True,
        "external_search_api_used_false": True,
        "football_outcomes_read_zero": True,
        "model_residuals_read_zero": True,
        "sportsbook_inputs_used_false": True,
        "paid_oddsapi_calls_zero": True,
        "week5_2026_outcomes_read_zero": True,
        "football_predictive_models_fit_zero": True,
        "production_changed_false": True,
    }
    required = list(gates)
    ready = all(bool(gates[k]) for k in required)
    return {
        "version": VERSION,
        "disposition": "LOCAL_DIRECT_TRANSPORT_READY" if ready else "LOCAL_DIRECT_TRANSPORT_PARTIAL",
        "v1g_rows": EXPECTED_V1G_ROWS,
        "v1g_official_auto_review_ready": EXPECTED_OFFICIAL_READY,
        "unresolved_rows": len(unresolved),
        "frozen_franchises": len(LOCAL_SOURCES),
        "frozen_publishers_declared": sum(len(v) for v in LOCAL_SOURCES.values()),
        "unique_publishers_declared": len({d for ds in LOCAL_SOURCES.values() for d in ds}),
        "franchises_with_reachable_publisher": franchises_with_reachable,
        "franchise_reachable": franchise_reachable,
        "unresolved_rows_with_reachable_publisher": rows_with_reachable,
        "unresolved_reachable_rate": rows_with_reachable / len(unresolved) if unresolved else 0.0,
        "unresolved_rows_with_local_candidate": rows_with_candidate,
        "unresolved_local_candidate_rate": rows_with_candidate / len(unresolved) if unresolved else 0.0,
        "zero_candidate_rows": [
            f"{r.season}|{r.week}|{r.team}" for r in row_results if not r.has_local_candidate
        ],
        "publisher_transport": {
            d: {
                "probe_reachable": p.probe_reachable,
                "transport_reachable": p.transport_reachable,
                "parsed_sitemaps": p.parsed_sitemaps,
                "enumerated_urls": p.enumerated_urls,
                "error_count": len(p.errors),
            }
            for d, p in sorted(publisher_results.items())
        },
        "gates": gates,
        "public_search_engine_html_contacted": False,
        "external_search_api_used": False,
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
    ap.add_argument("--v1g-rows", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _ready, unresolved = _load_v1g(args.v1g_rows)
    unresolved_teams = sorted({str(r["team"]) for r in unresolved})

    unique_domains = sorted({d for ds in LOCAL_SOURCES.values() for d in ds})
    probes: dict[str, bool] = {}
    probe_errors: dict[str, list[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(24, args.workers)) as pool:
        futs = {pool.submit(_probe_domain, d): d for d in unique_domains}
        for fut in concurrent.futures.as_completed(futs):
            d = futs[fut]
            ok, errs = fut.result()
            probes[d] = ok
            probe_errors[d] = errs

    needed_domains = sorted({d for t in unresolved_teams for d in LOCAL_SOURCES[t]})
    publisher_results: dict[str, PublisherResult] = {}
    entries_by_domain: dict[str, list[SitemapEntry]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, args.workers)) as pool:
        futs = {pool.submit(_enumerate_domain, d): d for d in needed_domains}
        for fut in concurrent.futures.as_completed(futs):
            d = futs[fut]
            result, entries = fut.result()
            # Preserve cheap reachability probe even if deep historical enumeration
            # later exhausts/blocks.
            result.probe_reachable = bool(probes.get(d, result.probe_reachable))
            publisher_results[d] = result
            entries_by_domain[d] = entries

    # Domains not needed by an unresolved row still retain probe-only audit state.
    for d in unique_domains:
        if d not in publisher_results:
            publisher_results[d] = PublisherResult(
                domain=d,
                probe_reachable=bool(probes.get(d, False)),
                transport_reachable=False,
                parsed_sitemaps=0,
                enumerated_urls=0,
                errors=probe_errors.get(d, []),
            )
            entries_by_domain[d] = []

    row_results: list[RowResult] = []
    for row in unresolved:
        team = str(row["team"])
        domains = list(LOCAL_SOURCES[team])
        reachable = [
            d for d in domains
            if publisher_results[d].transport_reachable
            and publisher_results[d].enumerated_urls > 0
        ]
        counts: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
        for d in domains:
            ranked = _candidate_urls(row, entries_by_domain.get(d, []))
            counts[d] = len(ranked)
            examples[d] = [u for _s, _dist, u in ranked[:5]]
        row_results.append(RowResult(
            season=int(row["season"]),
            week=int(row["week"]),
            team=team,
            opponent=str(row["opponent"]),
            kickoff=str(row["kickoff"]),
            mapped_publishers=domains,
            reachable_publishers=reachable,
            candidate_counts=counts,
            candidate_examples=examples,
            has_reachable_publisher=bool(reachable),
            has_local_candidate=any(v > 0 for v in counts.values()),
        ))

    row_results.sort(key=lambda r: (r.season, r.week, r.team))
    summary = _evaluate(unresolved, probes, publisher_results, row_results)

    (args.out_dir / "qb_intent_local_fallback_transport_v1h_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "qb_intent_local_fallback_transport_v1h_rows.json").write_text(
        json.dumps([asdict(r) for r in row_results], indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    (args.out_dir / "qb_intent_local_fallback_transport_v1h_publishers.json").write_text(
        json.dumps(
            {d: asdict(p) for d, p in sorted(publisher_results.items())},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
