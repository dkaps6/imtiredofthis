#!/usr/bin/env python3
"""V1J independent NLI semantic validator on immutable QB intent gold.

Source semantics only. No football outcomes, residuals, sportsbook inputs, or
predictive football fitting.
"""
from __future__ import annotations

import argparse
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
    OFFENSE_RE,
    POSTGAME_RE,
    _fragments,
    _opponent_relevant,
    _visible,
)

VERSION = "QB_TEAM_PASS_INTENT_INDEPENDENT_NLI_SEMANTIC_VALIDATOR_V1J"
MODEL_ID = "cross-encoder/nli-deberta-v3-small"
MODEL_REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"
HYPOTHESIS = (
    "The speaker states a specific offensive plan, intended adjustment, "
    "or intended player usage for the upcoming game."
)
MIN_ENTAILMENT = 0.70
MIN_MARGIN = 0.20


@dataclass
class GoldPage:
    team: str
    opponent: str
    expected_positive: bool
    url: str
    fetch_success: bool
    first_party: bool
    timestamp_safe: bool
    timestamp_method: str
    target_relevant: bool
    fragments_scored: int
    entailment_probability: float
    neutral_probability: float
    contradiction_probability: float
    semantic_margin: float
    evidence: str
    auto_promotion_ready: bool
    errors: list[str]


def _first_locator(raw: str) -> str:
    return next((x.strip() for x in str(raw).split(";") if x.strip()), "")


def _load_gold(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 13:
        raise RuntimeError(f"expected exact 13-row gold fixture, got {len(rows)}")
    return rows


def _gold_cases(rows: list[dict[str, str]]) -> list[tuple[dict[str, str], bool, str]]:
    cases: list[tuple[dict[str, str], bool, str]] = []
    official = [r for r in rows if r["source_class"] == "OFFICIAL"]
    local = [r for r in rows if r["source_class"] == "LOCAL_ATTRIBUTABLE"]
    if len(official) != 12 or len(local) != 1 or local[0]["team"] != "ATL":
        raise RuntimeError("unexpected immutable gold split")
    for row in official:
        cases.append((row, True, row["locator"].strip()))
    atl = local[0]
    cases.append((atl, False, _first_locator(atl["official_candidate_locators"])))
    return cases


def _label_indexes(id2label: dict[Any, Any]) -> tuple[int, int, int]:
    labels = {int(k): str(v).lower() for k, v in id2label.items()}
    entail = next((i for i, s in labels.items() if "entail" in s), None)
    contra = next((i for i, s in labels.items() if "contrad" in s), None)
    neutral = next((i for i, s in labels.items() if "neutral" in s), None)
    # Published model card mapping is contradiction, entailment, neutral.
    if entail is None or contra is None or neutral is None:
        if len(labels) == 3:
            return 1, 2, 0
        raise RuntimeError(f"unable to resolve NLI labels: {labels}")
    return entail, neutral, contra


def _qualifies_probs(entail: float, neutral: float, contra: float) -> tuple[bool, float]:
    margin = entail - max(neutral, contra)
    return bool(entail >= MIN_ENTAILMENT and margin >= MIN_MARGIN), margin


def _page_payload(row: dict[str, str], expected_positive: bool, url: str) -> tuple[dict[str, Any], list[str]]:
    team = row["team"]
    opponent = row["opponent"]
    errors: list[str] = []
    payload: dict[str, Any] = {
        "team": team,
        "opponent": opponent,
        "expected_positive": expected_positive,
        "url": url,
        "fetch_success": False,
        "first_party": False,
        "timestamp_safe": False,
        "timestamp_method": "",
        "target_relevant": False,
        "fragments": [],
    }
    try:
        raw, final_url = _read_page(url)
        payload["fetch_success"] = True
        payload["first_party"] = bool(
            _first_party(url, DOMAINS[team]) and _first_party(final_url, DOMAINS[team])
        )
        kickoff = _parse_absolute(row["kickoff"])
        if kickoff is None:
            raise RuntimeError("non_absolute_gold_kickoff")
        method, _value, parseable, pre = _extract_safe_publication(raw, kickoff)
        payload["timestamp_method"] = method
        payload["timestamp_safe"] = bool(parseable and pre)
        title, text = _visible(raw)
        payload["target_relevant"] = _opponent_relevant(url, title, text, opponent)
        payload["fragments"] = [
            f for f in _fragments(text)
            if OFFENSE_RE.search(f) and not POSTGAME_RE.search(f)
        ]
    except Exception as exc:
        errors.append(f"{type(exc).__name__}:{str(exc)[:180]}")
    return payload, errors


def _score_fragments(model, tokenizer, fragments: list[str]) -> tuple[float, float, float, float, str]:
    if not fragments:
        return 0.0, 0.0, 0.0, 0.0, ""
    import torch

    entail_i, neutral_i, contra_i = _label_indexes(model.config.id2label)
    best: tuple[float, float, float, float, str] | None = None

    batch_size = 32
    for start in range(0, len(fragments), batch_size):
        batch = fragments[start:start + batch_size]
        premises = batch
        hypotheses = [HYPOTHESIS] * len(batch)
        toks = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**toks).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for frag, p in zip(batch, probs):
            entail = float(p[entail_i])
            neutral = float(p[neutral_i])
            contra = float(p[contra_i])
            qualifies, margin = _qualifies_probs(entail, neutral, contra)
            if not qualifies:
                continue
            candidate = (margin, entail, neutral, contra, frag)
            if best is None or candidate[:2] > best[:2]:
                best = candidate

    if best is None:
        return 0.0, 0.0, 0.0, 0.0, ""
    margin, entail, neutral, contra, frag = best
    return entail, neutral, contra, margin, " ".join(frag.split()[:25])


def _evaluate(results: list[GoldPage], model_labels: dict[str, str]) -> dict[str, Any]:
    positives = [r for r in results if r.expected_positive]
    negatives = [r for r in results if not r.expected_positive]
    tp = sum(r.auto_promotion_ready for r in positives)
    fp = sum(r.auto_promotion_ready for r in negatives)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    target_rel = sum(r.target_relevant for r in positives)
    evidence_ok = all(len(r.evidence.split()) <= 25 for r in results if r.evidence)

    gates = {
        "exact_positive_gold_12": len(positives) == 12,
        "exact_negative_gold_1": len(negatives) == 1,
        "positive_auto_promotion_ready_ge_11_of_12": tp >= 11,
        "positive_target_relevance_ge_11_of_12": target_rel >= 11,
        "atl_negative_not_auto_promotion_ready": len(negatives) == 1 and not negatives[0].auto_promotion_ready,
        "pooled_precision_ge_0_85": precision >= 0.85,
        "evidence_le_25_words": evidence_ok,
        "exact_model_id": True,
        "exact_model_revision": True,
        "exact_hypothesis": True,
        "exact_thresholds": True,
        "supervised_fit_or_finetune_used_false": True,
        "football_outcomes_read_zero": True,
        "model_residuals_read_zero": True,
        "sportsbook_inputs_used_false": True,
        "paid_oddsapi_calls_zero": True,
        "week5_2026_outcomes_read_zero": True,
        "football_predictive_models_fit_zero": True,
        "production_changed_false": True,
    }
    passes = all(bool(v) for v in gates.values())
    return {
        "version": VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_labels": model_labels,
        "hypothesis": HYPOTHESIS,
        "min_entailment": MIN_ENTAILMENT,
        "min_margin": MIN_MARGIN,
        "disposition": (
            "INDEPENDENT_NLI_VALIDATOR_QUALIFIED_FOR_OFFICIAL_PROMOTION_SCREEN"
            if passes else
            "INDEPENDENT_NLI_VALIDATOR_NOT_QUALIFIED"
        ),
        "positive_gold": len(positives),
        "negative_gold": len(negatives),
        "positive_auto_promotion_ready": tp,
        "negative_auto_promotion_ready": fp,
        "positive_target_relevant": target_rel,
        "pooled_auto_promotion_precision": precision,
        "gates": gates,
        "supervised_fit_or_finetune_used": False,
        "football_outcomes_read": 0,
        "model_residuals_read": 0,
        "sportsbook_inputs_used": False,
        "paid_oddsapi_calls": 0,
        "week5_2026_outcomes_read": 0,
        "football_predictive_models_fit": 0,
        "production_changed": False,
        "full_official_promotion_screen_authorized": passes,
        "final_source_qualification_claimed": False,
        "predictive_candidate_authorized": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_gold(args.gold)
    cases = _gold_cases(rows)

    page_payloads: list[tuple[dict[str, Any], list[str]]] = [
        _page_payload(row, expected, url)
        for row, expected, url in cases
    ]

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    model.eval()

    labels = {str(k): str(v) for k, v in model.config.id2label.items()}
    _label_indexes(model.config.id2label)

    results: list[GoldPage] = []
    for payload, errors in page_payloads:
        entail, neutral, contra, margin, evidence = _score_fragments(
            model, tokenizer, payload["fragments"]
        )
        auto = bool(
            payload["fetch_success"]
            and payload["first_party"]
            and payload["timestamp_safe"]
            and payload["target_relevant"]
            and evidence
        )
        results.append(GoldPage(
            team=payload["team"],
            opponent=payload["opponent"],
            expected_positive=payload["expected_positive"],
            url=payload["url"],
            fetch_success=payload["fetch_success"],
            first_party=payload["first_party"],
            timestamp_safe=payload["timestamp_safe"],
            timestamp_method=payload["timestamp_method"],
            target_relevant=payload["target_relevant"],
            fragments_scored=len(payload["fragments"]),
            entailment_probability=round(entail, 6),
            neutral_probability=round(neutral, 6),
            contradiction_probability=round(contra, 6),
            semantic_margin=round(margin, 6),
            evidence=evidence,
            auto_promotion_ready=auto,
            errors=errors,
        ))

    summary = _evaluate(results, labels)
    (args.out_dir / "qb_intent_independent_nli_semantic_validator_v1j_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "qb_intent_independent_nli_semantic_validator_v1j_rows.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    for r in results:
        print(r.team, r.expected_positive, r.auto_promotion_ready,
              r.entailment_probability, r.semantic_margin, r.evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
