#!/usr/bin/env python3
"""V1F pretrained semantic source router gold validation.

This is source-routing research only. The pretrained language model is frozen and
is not trained/fine-tuned on football outcomes, gold rows, model residuals, or
sportsbook information.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.audit_qb_team_pass_intent_first_party_archive_transport_v1c import (
    DOMAINS,
    _first_party,
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
    _first_locator,
    _fragments,
    _opponent_relevant,
    _visible,
)

VERSION = "QB_TEAM_PASS_INTENT_PRETRAINED_SEMANTIC_ROUTER_V1F"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MIN_POSITIVE_SCORE = 0.38
MIN_MARGIN = 0.05
GOLD_PATH = Path("docs/research/QB_TEAM_PASS_INTENT_V1D_GOLD_PREFIX.csv")

POSITIVE_PROTOTYPES = {
    "RUN_EMPHASIS": "The speaker describes a specific plan to emphasize the running game or stay with the run in the upcoming game.",
    "PASS_EMPHASIS": "The speaker describes a specific plan to emphasize the passing game, throw more, or create more passing opportunities in the upcoming game.",
    "EARLY_DOWN_AGGRESSION": "The speaker describes a specific plan to be more aggressive on early downs in the upcoming game.",
    "TEMPO_CHANGE": "The speaker describes a specific plan to change offensive tempo, pace, or no-huddle usage in the upcoming game.",
    "PROTECTION_DRIVEN_PLAN": "The speaker describes a specific offensive protection plan or adjustment for pressure in the upcoming game.",
    "DEFENSIVE_MATCHUP_PLAN": "The speaker describes a specific offensive plan to respond to or exploit the opponent's defensive matchup, coverage, front, or pressure.",
    "PERSONNEL_AVAILABILITY_PLAN": "The speaker describes a specific planned change in offensive personnel usage, role, workload, touches, packages, or injury replacement for the upcoming game.",
    "OTHER_EXPLICIT_OFFENSIVE_INTENT": "The speaker describes a specific offensive game plan, approach, intended adjustment, or intended opportunity for the upcoming game.",
}
NEGATIVE_PROTOTYPES = (
    "This is general evaluation or preparation commentary without a specific offensive plan for the upcoming game.",
    "This is retrospective or descriptive football commentary rather than an explicit intended offensive action for the upcoming game.",
)


@dataclass
class PagePayload:
    team: str
    opponent: str
    expected_positive: bool
    url: str
    fetch_success: bool
    first_party: bool
    timestamp_safe: bool
    timestamp_method: str
    target_relevant: bool
    fragments: list[str]
    errors: list[str]


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
    semantic_candidate_found: bool
    candidate_tag: str
    positive_score: float
    negative_score: float
    semantic_margin: float
    evidence: str
    auto_review_ready: bool
    errors: list[str]


def _load_gold(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 13:
        raise RuntimeError(f"expected 13 gold rows, got {len(rows)}")
    return rows


def _page(row: dict[str, str], expected_positive: bool, url: str) -> PagePayload:
    team = row["team"]
    opponent = row["opponent"]
    errors: list[str] = []
    fetch = first_party = ts_safe = relevant = False
    method = ""
    fragments: list[str] = []
    try:
        raw, final_url = _read_page(url)
        fetch = True
        first_party = _first_party(url, DOMAINS[team]) and _first_party(final_url, DOMAINS[team])
        kickoff = _parse_absolute(row["kickoff"])
        if kickoff is None:
            raise RuntimeError("non-absolute frozen kickoff")
        method, _value, parseable, pre = _extract_safe_publication(raw, kickoff)
        ts_safe = bool(parseable and pre)
        title, text = _visible(raw)
        relevant = _opponent_relevant(url, title, text, opponent)
        fragments = [
            f for f in _fragments(text)
            if OFFENSE_RE.search(f) and not POSTGAME_RE.search(f)
        ]
    except Exception as exc:
        errors.append(f"{type(exc).__name__}:{str(exc)[:180]}")
    return PagePayload(
        team=team,
        opponent=opponent,
        expected_positive=expected_positive,
        url=url,
        fetch_success=fetch,
        first_party=first_party,
        timestamp_safe=ts_safe,
        timestamp_method=method,
        target_relevant=relevant,
        fragments=fragments,
        errors=errors,
    )


def _semantic_score(model, page: PagePayload, pos_emb: np.ndarray, neg_emb: np.ndarray) -> SemanticResult:
    tag_names = list(POSITIVE_PROTOTYPES)
    best: tuple[float, float, str, str] | None = None  # margin, pos, tag, fragment
    if page.fragments:
        emb = model.encode(
            page.fragments,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
        )
        pos_scores = emb @ pos_emb.T
        neg_scores = emb @ neg_emb.T
        for i, frag in enumerate(page.fragments):
            p_idx = int(np.argmax(pos_scores[i]))
            p_score = float(pos_scores[i, p_idx])
            n_score = float(np.max(neg_scores[i]))
            margin = p_score - n_score
            if p_score >= MIN_POSITIVE_SCORE and margin >= MIN_MARGIN:
                candidate = (margin, p_score, tag_names[p_idx], frag)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate

    found = best is not None
    if found:
        margin, p_score, tag, frag = best
        frag_emb = model.encode(
            [frag],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        n_score = float(np.max(frag_emb @ neg_emb.T))
        evidence = " ".join(frag.split()[:25])
    else:
        margin = p_score = n_score = 0.0
        tag = ""
        evidence = ""

    auto = bool(
        page.fetch_success
        and page.first_party
        and page.timestamp_safe
        and page.target_relevant
        and found
    )
    return SemanticResult(
        team=page.team,
        opponent=page.opponent,
        expected_positive=page.expected_positive,
        url=page.url,
        fetch_success=page.fetch_success,
        first_party=page.first_party,
        timestamp_safe=page.timestamp_safe,
        timestamp_method=page.timestamp_method,
        target_relevant=page.target_relevant,
        semantic_candidate_found=found,
        candidate_tag=tag,
        positive_score=round(p_score, 6),
        negative_score=round(n_score, 6),
        semantic_margin=round(margin, 6),
        evidence=evidence,
        auto_review_ready=auto,
        errors=page.errors,
    )


def evaluate(rows: list[SemanticResult]) -> dict[str, Any]:
    pos = [r for r in rows if r.expected_positive]
    neg = [r for r in rows if not r.expected_positive]
    tp = sum(r.auto_review_ready for r in pos)
    fp = sum(r.auto_review_ready for r in neg)
    rel = sum(r.target_relevant for r in pos)
    sem = sum(r.semantic_candidate_found for r in pos)
    precision = tp / (tp + fp) if tp + fp else 0.0
    bad_tags = sorted({r.candidate_tag for r in rows if r.candidate_tag and r.candidate_tag not in FROZEN_TAGS})
    long_evidence = [r.team for r in rows if len(r.evidence.split()) > 25]

    gates = {
        "exact_positive_official_gold_12": len(pos) == 12,
        "exact_negative_official_gold_1": len(neg) == 1 and neg[0].team == "ATL",
        "positive_auto_review_ready_ge_11_of_12": tp >= 11,
        "positive_target_relevance_ge_11_of_12": rel >= 11,
        "positive_semantic_evidence_ge_11_of_12": sem >= 11,
        "atl_official_negative_not_auto_ready": len(neg) == 1 and not neg[0].auto_review_ready,
        "auto_review_ready_precision_ge_085": precision >= 0.85,
        "evidence_le_25_words": not long_evidence,
        "only_frozen_semantic_tags": not bad_tags,
        "exact_model_id_revision_frozen": True,
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
        "exact_positive_official_gold_12",
        "exact_negative_official_gold_1",
        "positive_auto_review_ready_ge_11_of_12",
        "positive_target_relevance_ge_11_of_12",
        "positive_semantic_evidence_ge_11_of_12",
        "atl_official_negative_not_auto_ready",
        "auto_review_ready_precision_ge_085",
        "evidence_le_25_words",
        "only_frozen_semantic_tags",
    ]
    qualified = all(bool(gates[k]) for k in required)
    return {
        "version": VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "min_positive_score": MIN_POSITIVE_SCORE,
        "min_margin": MIN_MARGIN,
        "disposition": (
            "PRETRAINED_SEMANTIC_ROUTER_QUALIFIED_FOR_FULL_SOURCE_SCREEN"
            if qualified else "PRETRAINED_SEMANTIC_ROUTER_NOT_QUALIFIED"
        ),
        "positive_rows": len(pos),
        "negative_rows": len(neg),
        "positive_auto_review_ready": tp,
        "negative_auto_review_ready": fp,
        "positive_target_relevant": rel,
        "positive_semantic_evidence": sem,
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
        "supervised_fit_or_finetune_used": False,
        "football_predictive_models_fit": 0,
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
            url = _first_locator(row.get("official_candidate_locators", ""))
            if not url:
                raise RuntimeError("ATL frozen negative lacks official candidate locator")
            jobs.append((row, False, url))

    pages: list[PagePayload] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(_page, row, positive, url) for row, positive, url in jobs]
        for fut in concurrent.futures.as_completed(futures):
            pages.append(fut.result())
    pages.sort(key=lambda p: (not p.expected_positive, p.team))

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION)
    pos_texts = list(POSITIVE_PROTOTYPES.values())
    pos_emb = model.encode(
        pos_texts,
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

    results = [_semantic_score(model, p, pos_emb, neg_emb) for p in pages]
    summary = evaluate(results)

    (args.out_dir / "qb_intent_pretrained_semantic_v1f_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "qb_intent_pretrained_semantic_v1f_rows.json").write_text(
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
            f"semantic={r.semantic_candidate_found}",
            f"tag={r.candidate_tag}",
            f"pos={r.positive_score:.3f}",
            f"neg={r.negative_score:.3f}",
            f"margin={r.semantic_margin:.3f}",
            f"words={len(r.evidence.split())}",
            f"errors={len(r.errors)}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
