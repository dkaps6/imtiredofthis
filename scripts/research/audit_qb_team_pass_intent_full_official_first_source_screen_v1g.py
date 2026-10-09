#!/usr/bin/env python3
"""V1G full official-first historical source screen for QB pregame intent.

Source/retrieval research only. No football outcomes, model residuals, sportsbook
information, or predictive football fitting.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import urllib.parse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.build._schedule_utils import get_nfl_schedule
from scripts._opponent_map import canon_team
from scripts.research.audit_qb_team_pass_intent_first_party_archive_transport_v1c import (
    DOMAINS,
    MAX_SITEMAPS_PER_TEAM,
    MAX_URLS_PER_TEAM,
    _first_party,
    _parse_sitemap,
    _read_url,
    _robots,
)
from scripts.research.audit_qb_team_pass_intent_first_party_gold_validation_v1d import (
    _extract_safe_publication,
    _parse_absolute,
    _read_page,
)
from scripts.research.audit_qb_team_pass_intent_gold_semantic_extractor_v1e import (
    FROZEN_TAGS,
    OFFENSE_RE,
    POSTGAME_RE,
    TEAM_SYNONYMS,
    _fragments,
    _opponent_relevant,
    _visible,
)
from scripts.research.audit_qb_team_pass_intent_pretrained_semantic_router_v1f import (
    MIN_MARGIN,
    MIN_POSITIVE_SCORE,
    MODEL_ID,
    MODEL_REVISION,
    NEGATIVE_PROTOTYPES,
    POSITIVE_PROTOTYPES,
)

VERSION = "QB_TEAM_PASS_INTENT_FULL_OFFICIAL_FIRST_SOURCE_SCREEN_V1G"
SEASONS = (2023, 2024, 2025)
SAMPLED_WEEKS = (2, 5, 8, 11, 14, 17)
WINDOW_DAYS = 9
MAX_CANDIDATES = 20
SOURCE_TOKENS = (
    "transcript",
    "press-conference",
    "press-conferences",
    "media-availability",
    "media_availability",
    "what-they-said",
    "quotes",
    "game-preview",
)
EXPECTED_UNIVERSE = 536


@dataclass(frozen=True)
class TeamWeek:
    season: int
    week: int
    team: str
    opponent: str
    kickoff: str


@dataclass(frozen=True)
class SitemapEntry:
    url: str
    lastmod: str


@dataclass(frozen=True)
class Candidate:
    key: str
    season: int
    week: int
    team: str
    opponent: str
    kickoff: str
    rank: int
    discovery_score: int
    url: str
    lastmod: str
    opponent_url_match: bool
    week_url_match: bool
    source_token_match: bool
    year_url_match: bool
    date_window_match: bool


@dataclass
class PagePayload:
    candidate: Candidate
    fetch_success: bool
    first_party: bool
    timestamp_safe: bool
    timestamp_method: str
    publication_time: str
    target_relevant: bool
    fragments: list[str]
    errors: list[str]


@dataclass
class ScreenRow:
    season: int
    week: int
    team: str
    opponent: str
    kickoff: str
    screen_state: str
    candidate_count: int
    candidates_attempted: int
    accepted_rank: int
    locator: str
    publication_time: str
    timestamp_method: str
    candidate_tag: str
    positive_score: float
    negative_score: float
    semantic_margin: float
    evidence: str


def _key(x: TeamWeek) -> str:
    return f"{x.season}|{x.week}|{x.team}"


def _build_universe() -> list[TeamWeek]:
    rows: list[TeamWeek] = []
    seen: set[tuple[int, int, str]] = set()
    for season in SEASONS:
        df = get_nfl_schedule(season)
        df = df[df["week"].astype(int).isin(SAMPLED_WEEKS)].copy()
        for _, r in df.iterrows():
            week = int(r["week"])
            home = canon_team(str(r["home"]))
            away = canon_team(str(r["away"]))
            ko = r["kickoff_utc"]
            kickoff = ko.isoformat() if getattr(ko, "isoformat", None) else str(ko)
            if not kickoff or kickoff.lower() in {"nat", "nan", "none"}:
                raise RuntimeError(f"missing kickoff {season} W{week} {away}@{home}")
            for team, opp in ((home, away), (away, home)):
                k = (season, week, team)
                if k in seen:
                    raise RuntimeError(f"duplicate team-week {k}")
                seen.add(k)
                rows.append(TeamWeek(season, week, team, opp, kickoff))
    rows.sort(key=lambda x: (x.season, x.week, x.team))
    if len(rows) != EXPECTED_UNIVERSE:
        raise RuntimeError(f"frozen universe mismatch {len(rows)} != {EXPECTED_UNIVERSE}")
    return rows


def _enumerate_domain(domain: str) -> tuple[list[SitemapEntry], list[str]]:
    _status, declared, errors = _robots(domain)
    queue: list[str] = []
    seen_sitemaps: set[str] = set()
    entries: dict[str, str] = {}
    for u in [*declared, f"https://www.{domain}/sitemap-index.xml", f"https://www.{domain}/sitemap.xml"]:
        if u not in queue and _first_party(u, domain):
            queue.append(u)

    parsed = 0
    while queue and parsed < MAX_SITEMAPS_PER_TEAM and len(entries) < MAX_URLS_PER_TEAM:
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
            errors.append(f"sitemap:{sitemap}:{type(exc).__name__}:{str(exc)[:150]}")
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
            for loc, lastmod in rows:
                if len(entries) >= MAX_URLS_PER_TEAM:
                    break
                if _first_party(loc, domain):
                    entries.setdefault(loc, lastmod)
    return [SitemapEntry(u, lm) for u, lm in entries.items()], errors[:20]


def _parse_discovery_dt(value: str) -> datetime | None:
    s = (value or "").strip()
    if not s:
        return None
    s = s.replace("Z", "+00:00").replace("z", "+00:00")
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


def _path(url: str) -> str:
    return urllib.parse.unquote(urllib.parse.urlparse(url).path).lower()


def _url_token_match(path: str, values: tuple[str, ...]) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", path.lower())
    return any(re.sub(r"[^a-z0-9]+", " ", v.lower()).strip() in normalized for v in values)


def _week_match(path: str, week: int) -> bool:
    p = re.sub(r"[_-]+", " ", path.lower())
    return bool(re.search(rf"\bweek\s*{week}\b", p))


def _source_match(path: str) -> bool:
    return any(tok in path for tok in SOURCE_TOKENS)


def _year_match(path: str, season: int) -> bool:
    return bool(re.search(rf"(?:^|[^0-9]){season}(?:[^0-9]|$)", path))


def _candidate_rows(tw: TeamWeek, entries: list[SitemapEntry]) -> list[Candidate]:
    kickoff = _parse_absolute(tw.kickoff)
    if kickoff is None:
        raise RuntimeError(f"non-absolute kickoff {_key(tw)}")
    start = kickoff - timedelta(days=WINDOW_DAYS)
    opp_syn = TEAM_SYNONYMS[tw.opponent]
    selected: list[tuple[int, float, str, SitemapEntry, tuple[bool, bool, bool, bool, bool]]] = []
    for ent in entries:
        p = _path(ent.url)
        opp = _url_token_match(p, opp_syn)
        week = _week_match(p, tw.week)
        source = _source_match(p)
        year = _year_match(p, tw.season)
        last = _parse_discovery_dt(ent.lastmod)
        date_window = bool(last is not None and start <= last <= kickoff)
        strong = opp or week or (source and year)
        if not (date_window or strong):
            continue
        score = 4 * int(opp) + 3 * int(week) + 2 * int(source) + int(date_window)
        distance = abs((kickoff - last).total_seconds()) if last is not None else float("inf")
        selected.append((score, distance, ent.url.lower(), ent, (opp, week, source, year, date_window)))

    selected.sort(key=lambda x: (-x[0], x[1], x[2]))
    out: list[Candidate] = []
    for rank, item in enumerate(selected[:MAX_CANDIDATES], start=1):
        score, _distance, _u, ent, flags = item
        opp, week, source, year, date_window = flags
        out.append(Candidate(
            key=_key(tw), season=tw.season, week=tw.week, team=tw.team,
            opponent=tw.opponent, kickoff=tw.kickoff, rank=rank,
            discovery_score=score, url=ent.url, lastmod=ent.lastmod,
            opponent_url_match=opp, week_url_match=week,
            source_token_match=source, year_url_match=year,
            date_window_match=date_window,
        ))
    return out


def _fetch_candidate(c: Candidate) -> PagePayload:
    errors: list[str] = []
    fetch = first = safe = relevant = False
    method = pub_value = ""
    fragments: list[str] = []
    try:
        raw, final_url = _read_page(c.url)
        fetch = True
        first = _first_party(c.url, DOMAINS[c.team]) and _first_party(final_url, DOMAINS[c.team])
        kickoff = _parse_absolute(c.kickoff)
        if kickoff is None:
            raise RuntimeError("non-absolute kickoff")
        method, pub_value, parseable, pre = _extract_safe_publication(raw, kickoff)
        pub_dt = _parse_absolute(pub_value) if parseable else None
        safe = bool(
            pub_dt is not None
            and pre
            and pub_dt >= kickoff - timedelta(days=WINDOW_DAYS)
        )
        title, text = _visible(raw)
        relevant = _opponent_relevant(c.url, title, text, c.opponent)
        fragments = [
            f for f in _fragments(text)
            if OFFENSE_RE.search(f) and not POSTGAME_RE.search(f)
        ]
    except Exception as exc:
        errors.append(f"{type(exc).__name__}:{str(exc)[:180]}")
    return PagePayload(
        candidate=c, fetch_success=fetch, first_party=first,
        timestamp_safe=safe, timestamp_method=method,
        publication_time=pub_value, target_relevant=relevant,
        fragments=fragments, errors=errors,
    )


def _score_pages(model, pages: list[PagePayload], pos_emb: np.ndarray, neg_emb: np.ndarray) -> dict[str, dict[str, Any]]:
    tag_names = list(POSITIVE_PROTOTYPES)
    valid_pages = [
        p for p in pages
        if p.fetch_success and p.first_party and p.timestamp_safe and p.target_relevant and p.fragments
    ]
    flat: list[str] = []
    owner: list[tuple[str, int]] = []
    for p in valid_pages:
        for i, frag in enumerate(p.fragments):
            flat.append(frag)
            owner.append((p.candidate.key, i))

    best: dict[str, tuple[float, float, float, str, str]] = {}
    if flat:
        emb = model.encode(
            flat,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=128,
        )
        pos_scores = emb @ pos_emb.T
        neg_scores = emb @ neg_emb.T
        page_lookup = {p.candidate.key: p for p in valid_pages}
        for idx, (key, frag_i) in enumerate(owner):
            p_idx = int(np.argmax(pos_scores[idx]))
            p_score = float(pos_scores[idx, p_idx])
            n_score = float(np.max(neg_scores[idx]))
            margin = p_score - n_score
            if p_score < MIN_POSITIVE_SCORE or margin < MIN_MARGIN:
                continue
            frag = page_lookup[key].fragments[frag_i]
            candidate = (margin, p_score, n_score, tag_names[p_idx], frag)
            if key not in best or candidate[:2] > best[key][:2]:
                best[key] = candidate

    out: dict[str, dict[str, Any]] = {}
    for p in pages:
        b = best.get(p.candidate.key)
        if b is None:
            out[p.candidate.key] = {
                "accepted": False, "tag": "", "positive_score": 0.0,
                "negative_score": 0.0, "margin": 0.0, "evidence": "",
            }
        else:
            margin, p_score, n_score, tag, frag = b
            out[p.candidate.key] = {
                "accepted": True,
                "tag": tag,
                "positive_score": round(p_score, 6),
                "negative_score": round(n_score, 6),
                "margin": round(margin, 6),
                "evidence": " ".join(frag.split()[:25]),
            }
    return out


def _evaluate(rows: list[ScreenRow]) -> dict[str, Any]:
    accepted = [r for r in rows if r.screen_state == "OFFICIAL_AUTO_REVIEW_READY"]
    total = len(rows)
    pooled = len(accepted) / total if total else 0.0

    season_total = Counter(r.season for r in rows)
    season_ok = Counter(r.season for r in accepted)
    season_rates = {str(s): season_ok[s] / season_total[s] for s in SEASONS}

    week_total = Counter(r.week for r in rows)
    week_ok = Counter(r.week for r in accepted)
    week_rates = {str(w): week_ok[w] / week_total[w] for w in SAMPLED_WEEKS}

    team_total = Counter(r.team for r in rows)
    team_ok = Counter(r.team for r in accepted)
    team_rates = {t: team_ok[t] / team_total[t] for t in sorted(team_total)}
    franchises_50 = sum(v >= 0.50 for v in team_rates.values())

    bad_tags = sorted({r.candidate_tag for r in accepted if r.candidate_tag not in FROZEN_TAGS})
    long_evidence = [f"{r.season}|{r.week}|{r.team}" for r in accepted if len(r.evidence.split()) > 25]

    gates = {
        "exact_universe_536": total == EXPECTED_UNIVERSE,
        "pooled_official_auto_ready_ge_70pct": pooled >= 0.70,
        "each_season_ge_60pct": all(v >= 0.60 for v in season_rates.values()),
        "each_sampled_week_ge_50pct": all(v >= 0.50 for v in week_rates.values()),
        "franchises_ge_50pct_at_least_24": franchises_50 >= 24,
        "accepted_first_party_and_timestamp_safe": True,
        "evidence_le_25_words": not long_evidence,
        "only_frozen_semantic_tags": not bad_tags,
        "exact_v1f_model_revision_frozen": True,
        "supervised_fit_or_finetune_used": False,
        "external_search_engine_html_contacted": False,
        "football_outcomes_read": False,
        "model_residuals_read": False,
        "sportsbook_inputs_used": False,
        "week5_2026_outcomes_read": False,
        "football_predictive_models_fit_zero": True,
        "production_changed": False,
    }
    required = [
        "exact_universe_536",
        "pooled_official_auto_ready_ge_70pct",
        "each_season_ge_60pct",
        "each_sampled_week_ge_50pct",
        "franchises_ge_50pct_at_least_24",
        "accepted_first_party_and_timestamp_safe",
        "evidence_le_25_words",
        "only_frozen_semantic_tags",
    ]
    clears = all(bool(gates[k]) for k in required)
    return {
        "version": VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "min_positive_score": MIN_POSITIVE_SCORE,
        "min_margin": MIN_MARGIN,
        "window_days": WINDOW_DAYS,
        "max_candidates_per_team_week": MAX_CANDIDATES,
        "disposition": (
            "OFFICIAL_AUTO_READY_LOWER_BOUND_CLEARS_PARENT_DENSITY_GATES"
            if clears else
            "OFFICIAL_AUTO_READY_LOWER_BOUND_DOES_NOT_CLEAR_PARENT_DENSITY_GATES"
        ),
        "team_weeks": total,
        "official_auto_review_ready": len(accepted),
        "official_auto_review_ready_rate": pooled,
        "unresolved_requires_fallback_or_review": total - len(accepted),
        "season_rates": season_rates,
        "sampled_week_rates": week_rates,
        "franchise_rates": team_rates,
        "franchises_ge_50pct": franchises_50,
        "bad_tags": bad_tags,
        "long_evidence_rows": long_evidence,
        "gates": gates,
        "official_rows_finalized_as_no_source": 0,
        "external_search_engine_html_contacted": False,
        "football_outcomes_read": 0,
        "model_residuals_read": 0,
        "sportsbook_inputs_used": False,
        "paid_oddsapi_calls": 0,
        "week5_2026_outcomes_read": 0,
        "supervised_fit_or_finetune_used": False,
        "football_predictive_models_fit": 0,
        "production_changed": False,
        "predictive_candidate_authorized": False,
        "final_source_qualification_claimed": False,
        "final_source_qualification_stage_authorized": clears,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    universe = _build_universe()
    by_team: dict[str, list[TeamWeek]] = {}
    for tw in universe:
        by_team.setdefault(tw.team, []).append(tw)

    domain_entries: dict[str, list[SitemapEntry]] = {}
    domain_errors: dict[str, list[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, args.workers)) as pool:
        futures = {
            pool.submit(_enumerate_domain, DOMAINS[team]): team
            for team in sorted(by_team)
        }
        for fut in concurrent.futures.as_completed(futures):
            team = futures[fut]
            entries, errors = fut.result()
            domain_entries[team] = entries
            domain_errors[team] = errors

    candidates: dict[str, list[Candidate]] = {}
    for tw in universe:
        candidates[_key(tw)] = _candidate_rows(tw, domain_entries.get(tw.team, []))

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION)
    pos_emb = model.encode(
        list(POSITIVE_PROTOTYPES.values()),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    neg_emb = model.encode(
        list(NEGATIVE_PROTOTYPES),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    resolved: dict[str, ScreenRow] = {}
    attempted = Counter()
    for rank in range(1, MAX_CANDIDATES + 1):
        jobs: list[Candidate] = []
        for tw in universe:
            key = _key(tw)
            if key in resolved:
                continue
            cs = candidates[key]
            if len(cs) >= rank:
                jobs.append(cs[rank - 1])
        if not jobs:
            continue

        pages: list[PagePayload] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_fetch_candidate, c): c.key for c in jobs}
            for fut in concurrent.futures.as_completed(futures):
                page = fut.result()
                pages.append(page)
                attempted[page.candidate.key] += 1

        scored = _score_pages(model, pages, pos_emb, neg_emb)
        by_key_page = {p.candidate.key: p for p in pages}
        for key, score in scored.items():
            if not score["accepted"] or key in resolved:
                continue
            p = by_key_page[key]
            c = p.candidate
            resolved[key] = ScreenRow(
                season=c.season, week=c.week, team=c.team, opponent=c.opponent,
                kickoff=c.kickoff, screen_state="OFFICIAL_AUTO_REVIEW_READY",
                candidate_count=len(candidates[key]), candidates_attempted=attempted[key],
                accepted_rank=c.rank, locator=c.url,
                publication_time=p.publication_time,
                timestamp_method=p.timestamp_method,
                candidate_tag=score["tag"],
                positive_score=score["positive_score"],
                negative_score=score["negative_score"],
                semantic_margin=score["margin"],
                evidence=score["evidence"],
            )

    rows: list[ScreenRow] = []
    for tw in universe:
        key = _key(tw)
        if key in resolved:
            rows.append(resolved[key])
        else:
            rows.append(ScreenRow(
                season=tw.season, week=tw.week, team=tw.team, opponent=tw.opponent,
                kickoff=tw.kickoff,
                screen_state="OFFICIAL_UNRESOLVED_REQUIRES_FALLBACK_OR_REVIEW",
                candidate_count=len(candidates[key]), candidates_attempted=attempted[key],
                accepted_rank=0, locator="", publication_time="", timestamp_method="",
                candidate_tag="", positive_score=0.0, negative_score=0.0,
                semantic_margin=0.0, evidence="",
            ))
    rows.sort(key=lambda r: (r.season, r.week, r.team))

    summary = _evaluate(rows)
    summary["domain_error_counts"] = {t: len(domain_errors.get(t, [])) for t in sorted(by_team)}
    summary["teams_with_zero_enumerated_urls"] = sorted(
        t for t in by_team if not domain_entries.get(t)
    )
    summary["team_weeks_with_zero_candidates"] = sum(r.candidate_count == 0 for r in rows)

    (args.out_dir / "qb_intent_full_official_screen_v1g_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "qb_intent_full_official_screen_v1g_rows.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    for s in SEASONS:
        print("season", s, summary["season_rates"][str(s)])
    for w in SAMPLED_WEEKS:
        print("week", w, summary["sampled_week_rates"][str(w)])
    print("resolved", summary["official_auto_review_ready"], "of", summary["team_weeks"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
