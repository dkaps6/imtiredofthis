#!/usr/bin/env python3
"""Gold semantic candidate-extractor validation for QB pregame intent V1E.

Operational/source-only research. No football outcomes, model residuals,
sportsbook inputs, or predictive fitting.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
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

VERSION = "QB_TEAM_PASS_INTENT_GOLD_SEMANTIC_EXTRACTOR_V1E"
GOLD_PATH = Path("docs/research/QB_TEAM_PASS_INTENT_V1D_GOLD_PREFIX.csv")
FROZEN_TAGS = {
    "RUN_EMPHASIS",
    "PASS_EMPHASIS",
    "EARLY_DOWN_AGGRESSION",
    "TEMPO_CHANGE",
    "PROTECTION_DRIVEN_PLAN",
    "DEFENSIVE_MATCHUP_PLAN",
    "PERSONNEL_AVAILABILITY_PLAN",
    "OTHER_EXPLICIT_OFFENSIVE_INTENT",
}

TEAM_SYNONYMS = {
    "ARI": ("cardinals", "arizona"),
    "ATL": ("falcons", "atlanta"),
    "BAL": ("ravens", "baltimore"),
    "BUF": ("bills", "buffalo"),
    "CAR": ("panthers", "carolina"),
    "CHI": ("bears", "chicago"),
    "CIN": ("bengals", "cincinnati"),
    "CLE": ("browns", "cleveland"),
    "DAL": ("cowboys", "dallas"),
    "DEN": ("broncos", "denver"),
    "DET": ("lions", "detroit"),
    "GB": ("packers", "green bay"),
    "HOU": ("texans", "houston"),
    "IND": ("colts", "indianapolis"),
    "JAX": ("jaguars", "jacksonville", "jags"),
    "KC": ("chiefs", "kansas city"),
    "LV": ("raiders", "las vegas"),
    "LAC": ("chargers", "los angeles chargers"),
    "LAR": ("rams", "los angeles rams"),
    "MIA": ("dolphins", "miami"),
    "MIN": ("vikings", "minnesota"),
    "NE": ("patriots", "new england"),
    "NO": ("saints", "new orleans"),
    "NYG": ("giants", "new york giants"),
    "NYJ": ("jets", "new york jets"),
    "PHI": ("eagles", "philadelphia"),
    "PIT": ("steelers", "pittsburgh"),
    "SF": ("49ers", "niners", "san francisco"),
    "SEA": ("seahawks", "seattle"),
    "TB": ("buccaneers", "bucs", "tampa bay"),
    "TEN": ("titans", "tennessee"),
    "WAS": ("commanders", "washington"),
}

ACTION_RE = re.compile(
    r"\b(?:game\s*plan|plan(?:ned|ning)?|approach|want(?:ed)?\s+to|need\s+to|"
    r"going\s+to|we(?:'|’)ll|will\s+(?:try|look|lean|attack|run|throw|use|give|"
    r"create|take|get|open)|look(?:ing)?\s+to|emphasis|focus(?:ed|ing)?\s+on|"
    r"goal|aim|attack|establish|stick\s+with|lean\s+on|more\s+(?:touches|"
    r"opportunities|targets)|open(?:ing)?\s+up|create\s+(?:more\s+)?"
    r"opportunities|take\s+(?:more\s+)?shots|be\s+aggressive)\b",
    re.I,
)
OFFENSE_RE = re.compile(
    r"\b(?:offense|offensive|pass(?:ing)?|throw(?:ing)?|quarterback|qb|receiver|"
    r"playmaker|run(?:ning)?|ground\s+game|backfield|running\s+back|ball|touches|"
    r"targets|downfield|protection|coverage|front|tempo|motion|formation|explosive|"
    r"shot(?:s)?|third\s+down|early\s+down|first\s+down)\b",
    re.I,
)
POSTGAME_RE = re.compile(
    r"\b(?:postgame|after\s+(?:the|sunday'?s|monday'?s|thursday'?s)\s+game|"
    r"final\s+score|victory\s+over|win\s+over|loss\s+to|defeated\s+the)\b",
    re.I,
)

TAG_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("RUN_EMPHASIS", re.compile(
        r"\b(?:establish\s+(?:the\s+)?run|run(?:ning)?\s+game|run\s+the\s+ball|"
        r"ground\s+game|stick\s+with\s+(?:the\s+)?run|lean\s+on\s+(?:the\s+)?run)\b",
        re.I,
    )),
    ("PASS_EMPHASIS", re.compile(
        r"\b(?:passing\s+game|throw(?:ing)?\s+(?:the\s+)?ball|let\s+(?:it|the\s+ball)\s+fly|"
        r"air\s+it\s+out|through\s+the\s+air|downfield\s+(?:shot|pass)|vertical\s+pass|"
        r"quick-release\s+pass)\b",
        re.I,
    )),
    ("EARLY_DOWN_AGGRESSION", re.compile(
        r"\b(?:early[-\s]down|first[-\s]down|attack\s+early|aggressive\s+early)\b",
        re.I,
    )),
    ("TEMPO_CHANGE", re.compile(
        r"\b(?:tempo|no[-\s]?huddle|hurry[-\s]?up|pace\s+of\s+play|play\s+faster|speed\s+up)\b",
        re.I,
    )),
    ("PROTECTION_DRIVEN_PLAN", re.compile(
        r"\b(?:pass\s+protection|protection|protect\s+(?:the\s+)?(?:quarterback|qb)|"
        r"offensive\s+line|pass\s+pro)\b",
        re.I,
    )),
    ("DEFENSIVE_MATCHUP_PLAN", re.compile(
        r"\b(?:matchup|coverage|man\s+coverage|zone\s+coverage|defensive\s+front|"
        r"front\s+seven|blitz|pressure|defensive\s+line|secondary)\b",
        re.I,
    )),
    ("PERSONNEL_AVAILABILITY_PLAN", re.compile(
        r"\b(?:without\s+\w+|return(?:ing)?|availability|available|injur(?:y|ed)|replacement|"
        r"more\s+(?:touches|opportunities|targets)|workload|two[-\s]back|packages?)\b",
        re.I,
    )),
]


class VisibleParser(HTMLParser):
    BLOCKED = {"script", "style", "noscript", "svg", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth_blocked = 0
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in self.BLOCKED:
            self.depth_blocked += 1
        elif t == "title":
            self.in_title = True

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self.BLOCKED and self.depth_blocked:
            self.depth_blocked -= 1
        elif t == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.depth_blocked:
            return
        s = re.sub(r"\s+", " ", data).strip()
        if not s:
            return
        self.parts.append(s)
        if self.in_title:
            self.title_parts.append(s)


@dataclass
class SemanticResult:
    team: str
    opponent: str
    expected_positive: bool
    url: str
    fetch_success: bool
    first_party: bool
    timestamp_safe: bool
    timestamp_method: str
    target_relevant: bool
    evidence_found: bool
    candidate_tags: list[str]
    evidence: str
    auto_review_ready: bool
    errors: list[str]


def _visible(raw: bytes) -> tuple[str, str]:
    p = VisibleParser()
    p.feed(raw.decode("utf-8", errors="replace"))
    title = " ".join(p.title_parts)
    text = " ".join(p.parts)
    return title, text


def _fragments(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\s*[\n\r]+\s*|\s*[|•]\s*", text)
    out: list[str] = []
    for c in chunks:
        c = re.sub(r"\s+", " ", c).strip(" -\t")
        if 25 <= len(c) <= 420 and 5 <= len(c.split()) <= 65:
            out.append(c)
    return out


def _opponent_relevant(url: str, title: str, text: str, opponent: str) -> bool:
    hay = (urllib_parse_unquote(url) + " " + title + " " + text).lower()
    return any(s.lower() in hay for s in TEAM_SYNONYMS[opponent])


def urllib_parse_unquote(value: str) -> str:
    import urllib.parse
    return urllib.parse.unquote(value)


def _tags(fragment: str) -> list[str]:
    found = [tag for tag, pat in TAG_PATTERNS if pat.search(fragment)]
    if not found and ACTION_RE.search(fragment) and OFFENSE_RE.search(fragment):
        found = ["OTHER_EXPLICIT_OFFENSIVE_INTENT"]
    return found


def _candidate(fragment: str) -> bool:
    if POSTGAME_RE.search(fragment):
        return False
    tags = _tags(fragment)
    if not tags:
        return False
    return bool(ACTION_RE.search(fragment) and OFFENSE_RE.search(fragment))


def _best_evidence(text: str) -> tuple[str, list[str]]:
    scored: list[tuple[int, int, str, list[str]]] = []
    for frag in _fragments(text):
        if not _candidate(frag):
            continue
        tags = _tags(frag)
        specificity = sum(t != "OTHER_EXPLICIT_OFFENSIVE_INTENT" for t in tags)
        scored.append((specificity, -len(frag.split()), frag, tags))
    if not scored:
        return "", []
    scored.sort(reverse=True)
    frag, tags = scored[0][2], scored[0][3]
    words = frag.split()
    evidence = " ".join(words[:25])
    return evidence, tags


def _first_locator(raw: str) -> str:
    return next((x.strip() for x in str(raw).split(";") if x.strip()), "")


def _audit_page(row: dict[str, str], expected_positive: bool, url: str) -> SemanticResult:
    team = row["team"]
    opponent = row["opponent"]
    errors: list[str] = []
    fetch = False
    first_party = False
    ts_safe = False
    ts_method = ""
    relevant = False
    evidence = ""
    tags: list[str] = []
    try:
        raw, final_url = _read_page(url)
        fetch = True
        first_party = _first_party(url, DOMAINS[team]) and _first_party(final_url, DOMAINS[team])
        kickoff = _parse_absolute(row["kickoff"])
        if kickoff is None:
            raise RuntimeError("non-absolute frozen kickoff")
        ts_method, _value, parseable, pre = _extract_safe_publication(raw, kickoff)
        ts_safe = bool(parseable and pre)
        title, text = _visible(raw)
        relevant = _opponent_relevant(url, title, text, opponent)
        evidence, tags = _best_evidence(text)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}:{str(exc)[:180]}")

    auto = bool(fetch and first_party and ts_safe and relevant and evidence)
    return SemanticResult(
        team=team,
        opponent=opponent,
        expected_positive=expected_positive,
        url=url,
        fetch_success=fetch,
        first_party=first_party,
        timestamp_safe=ts_safe,
        timestamp_method=ts_method,
        target_relevant=relevant,
        evidence_found=bool(evidence),
        candidate_tags=tags,
        evidence=evidence,
        auto_review_ready=auto,
        errors=errors,
    )


def _load_gold(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 13:
        raise RuntimeError(f"expected 13 gold rows, got {len(rows)}")
    return rows


def evaluate(results: list[SemanticResult]) -> dict[str, Any]:
    pos = [r for r in results if r.expected_positive]
    neg = [r for r in results if not r.expected_positive]
    tp = sum(r.auto_review_ready for r in pos)
    fp = sum(r.auto_review_ready for r in neg)
    rel = sum(r.target_relevant for r in pos)
    ev = sum(r.evidence_found for r in pos)
    precision = tp / (tp + fp) if tp + fp else 0.0
    bad_tags = sorted({t for r in results for t in r.candidate_tags if t not in FROZEN_TAGS})
    long_evidence = [r.team for r in results if len(r.evidence.split()) > 25]

    gates = {
        "exact_positive_official_gold_12": len(pos) == 12,
        "exact_negative_official_gold_1": len(neg) == 1 and neg[0].team == "ATL",
        "positive_auto_review_ready_ge_11_of_12": tp >= 11,
        "positive_target_relevance_ge_11_of_12": rel >= 11,
        "positive_evidence_found_ge_11_of_12": ev >= 11,
        "atl_official_negative_not_auto_ready": len(neg) == 1 and not neg[0].auto_review_ready,
        "auto_review_ready_precision_ge_085": precision >= 0.85,
        "evidence_le_25_words": not long_evidence,
        "only_frozen_semantic_tags": not bad_tags,
        "external_search_engine_html_contacted": False,
        "football_outcomes_read": False,
        "model_residuals_read": False,
        "sportsbook_inputs_used": False,
        "week5_2026_outcomes_read": False,
        "predictive_models_fit_zero": True,
        "production_changed": False,
    }
    required = [
        "exact_positive_official_gold_12",
        "exact_negative_official_gold_1",
        "positive_auto_review_ready_ge_11_of_12",
        "positive_target_relevance_ge_11_of_12",
        "positive_evidence_found_ge_11_of_12",
        "atl_official_negative_not_auto_ready",
        "auto_review_ready_precision_ge_085",
        "evidence_le_25_words",
        "only_frozen_semantic_tags",
    ]
    qualified = all(bool(gates[k]) for k in required)
    return {
        "version": VERSION,
        "disposition": (
            "GOLD_SEMANTIC_EXTRACTOR_QUALIFIED_FOR_FULL_SOURCE_SCREEN"
            if qualified else "GOLD_SEMANTIC_EXTRACTOR_NOT_QUALIFIED"
        ),
        "positive_rows": len(pos),
        "negative_rows": len(neg),
        "positive_auto_review_ready": tp,
        "negative_auto_review_ready": fp,
        "positive_target_relevant": rel,
        "positive_evidence_found": ev,
        "auto_review_ready_precision": precision,
        "bad_tags": bad_tags,
        "long_evidence_teams": long_evidence,
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
        "full_source_screen_authorized": qualified,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", type=Path, default=GOLD_PATH)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gold = _load_gold(args.gold)

    jobs: list[tuple[dict[str, str], bool, str]] = []
    for row in gold:
        if row["source_class"] == "OFFICIAL":
            jobs.append((row, True, row["locator"].strip()))
        elif row["team"] == "ATL" and row["source_class"] == "LOCAL_ATTRIBUTABLE":
            official = _first_locator(row.get("official_candidate_locators", ""))
            if not official:
                raise RuntimeError("ATL frozen negative lacks official candidate locator")
            jobs.append((row, False, official))

    results: list[SemanticResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(_audit_page, row, positive, url): (row["team"], positive)
            for row, positive, url in jobs
        }
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    results.sort(key=lambda r: (not r.expected_positive, r.team))

    summary = evaluate(results)
    (args.out_dir / "qb_intent_gold_semantic_v1e_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "qb_intent_gold_semantic_v1e_rows.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    for r in results:
        print(
            r.team,
            f"positive={r.expected_positive}",
            f"auto={r.auto_review_ready}",
            f"relevant={r.target_relevant}",
            f"evidence={r.evidence_found}",
            f"tags={','.join(r.candidate_tags)}",
            f"words={len(r.evidence.split())}",
            f"errors={len(r.errors)}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
