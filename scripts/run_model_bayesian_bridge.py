#!/usr/bin/env python3
"""Materialize leakage-safe empirical-Bayesian player baselines."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.modeling.bayesian_v2 import load_bayesian_baseline

OUT = Path("data/model_bayesian_diagnostics.csv")


def main() -> int:
    df = load_bayesian_baseline()
    if df.empty:
        raise RuntimeError("Bayesian baseline produced 0 rows")
    if int(pd.to_numeric(df["bayes_available"], errors="coerce").fillna(0).sum()) != len(df):
        raise RuntimeError("Bayesian baseline is unavailable for one or more active players")
    if df.duplicated(["team", "player_clean_key"]).any():
        raise RuntimeError("Bayesian baseline contains duplicate player/team rows")
    compatibility = int(pd.to_numeric(df["bayes_compatibility_prior_used"], errors="coerce").fillna(0).max())
    if compatibility:
        raise RuntimeError(
            "Bayesian baseline had to reconstruct priors from already-blended PlayerForm metrics. "
            "Production requires preserved *_prior/*_current evidence columns."
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    states = df["bayes_evidence_state"].value_counts(dropna=False).to_dict()
    print(f"[bayesian_v2] players={len(df)} evidence_states={states}")
    print(f"[bayesian_v2] wrote {len(df)} rows -> {OUT}")
    print("[bayesian_v2] market-independent and pregame-only; preserved prior/current evidence verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
