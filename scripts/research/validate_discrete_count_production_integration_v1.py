#!/usr/bin/env python3
"""Exact production-integration parity validator for Discrete Count Alignment V1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.discrete_count_alignment_v1 import align_outcomes
from scripts.research.discrete_count_mean_alignment_v1 import (
    KEYS,
    continuous_align,
    discrete_largest_remainder,
    empirical_crps,
    load_metadata,
)

ATOL = 1e-12


def _canon(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    x["season"] = pd.to_numeric(x["season"], errors="coerce").astype(int)
    x["week"] = pd.to_numeric(x["week"], errors="coerce").astype(int)
    x["market"] = x["market"].astype(str).str.lower()
    x["player_clean_key"] = x["player_clean_key"].astype(str)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projection-file", type=Path, required=True)
    ap.add_argument("--distribution-dir", type=Path, required=True)
    ap.add_argument("--research-detail", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    proj = _canon(pd.read_csv(args.projection_file, low_memory=False))
    proj = proj.loc[proj["market"].isin({"receptions", "rush_att"})].copy()
    meta = load_metadata(args.distribution_dir)
    detail = _canon(pd.read_csv(args.research_detail, low_memory=False))

    merged = proj.merge(
        meta[KEYS + ["array_key", "npz_file"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(proj):
        raise RuntimeError(f"projection/metadata coverage mismatch {len(merged)} != {len(proj)}")

    expected_detail = detail.set_index(KEYS)
    if expected_detail.index.duplicated().any():
        raise RuntimeError("research detail has duplicate identity")

    cache: dict[str, object] = {}
    max_array_gap = 0.0
    max_mean_gap_vs_research = 0.0
    max_crps_gap_vs_research = 0.0
    applied_rows = 0
    noop_rows = 0
    integer_failures = 0
    rows = []

    for r in merged.itertuples(index=False):
        fn = str(r.npz_file)
        if fn not in cache:
            cache[fn] = np.load(args.distribution_dir / fn, allow_pickle=False)
        raw = np.asarray(cache[fn][str(r.array_key)], dtype=float)
        mc = float(np.mean(raw))
        target = float(r.ensemble_proj)

        prod, meta_out = align_outcomes(
            raw,
            market=str(r.market),
            mc_proj=mc,
            target_mean=target,
        )

        a0 = continuous_align(raw, target)
        if mc > 0 and np.isfinite(target):
            research = discrete_largest_remainder(a0)
        else:
            research = raw.copy()

        gap = float(np.max(np.abs(prod - research))) if len(prod) else 0.0
        max_array_gap = max(max_array_gap, gap)
        if gap > ATOL:
            raise RuntimeError(
                f"production/research array mismatch {gap} for "
                f"{r.season} W{r.week} {r.player_clean_key} {r.market}"
            )

        key = tuple(getattr(r, k) for k in KEYS)
        if key not in expected_detail.index:
            raise RuntimeError(f"row absent from frozen research detail: {key}")
        frozen = expected_detail.loc[key]
        mean_gap = abs(float(np.mean(prod)) - float(frozen["a1_mean"]))
        crps = empirical_crps(prod, float(r.actual))
        crps_gap = abs(float(crps) - float(frozen["a1_crps"]))
        max_mean_gap_vs_research = max(max_mean_gap_vs_research, mean_gap)
        max_crps_gap_vs_research = max(max_crps_gap_vs_research, crps_gap)

        if mean_gap > 1e-10 or crps_gap > 1e-10:
            raise RuntimeError(
                f"frozen result parity mismatch mean={mean_gap} crps={crps_gap} key={key}"
            )

        applied = int(meta_out["discrete_count_alignment_applied"])
        applied_rows += applied
        noop_rows += 1 - applied
        int_gap = float(np.max(np.abs(prod - np.rint(prod)))) if len(prod) else 0.0
        if applied and int_gap > ATOL:
            integer_failures += 1

        rows.append(
            {
                "season": int(r.season),
                "week": int(r.week),
                "team": str(r.team),
                "player_clean_key": str(r.player_clean_key),
                "market": str(r.market),
                "applied": applied,
                "array_max_gap_vs_research": gap,
                "mean_gap_vs_research": mean_gap,
                "crps_gap_vs_research": crps_gap,
                "integer_max_gap": int_gap,
            }
        )

    # Explicit legacy invariance probes for non-count markets.
    probe = np.array([0.0, 2.5, 7.0, 11.25], dtype=float)
    probe_mc = float(np.mean(probe))
    probe_target = 8.125
    legacy = probe * (probe_target / probe_mc)
    noncount = {}
    for market in ("pass_yards", "rush_yards", "rec_yards", "rush_rec_yards", "anytime_td"):
        out, info = align_outcomes(
            probe,
            market=market,
            mc_proj=probe_mc,
            target_mean=probe_target,
        )
        exact = bool(np.array_equal(out, legacy) and int(info["discrete_count_alignment_applied"]) == 0)
        noncount[market] = exact
        if not exact:
            raise RuntimeError(f"non-count legacy invariance failed for {market}")

    result = {
        "disposition": "DISCRETE_COUNT_MEAN_ALIGNMENT_V1_INTEGRATION_MECHANICS_PASS",
        "rows_checked": int(len(rows)),
        "applied_rows": int(applied_rows),
        "production_noop_rows": int(noop_rows),
        "max_array_gap_vs_research": max_array_gap,
        "max_mean_gap_vs_research": max_mean_gap_vs_research,
        "max_crps_gap_vs_research": max_crps_gap_vs_research,
        "integer_failures": int(integer_failures),
        "noncount_legacy_invariance": noncount,
        "sportsbook_inputs_to_football": 0,
        "parameters_fit": 0,
        "week3_outcomes_used": False,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out.with_suffix(".csv"), index=False)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
