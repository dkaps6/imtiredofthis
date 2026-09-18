#!/usr/bin/env python3
"""Build leakage-safe stability evidence for football-context candidate signals.

Engineering/QA only. Given a historical feature table, compare each entity's prior-period
feature value with its next observed period. No target-game outcomes are read or scored.
The output is designed to feed Football Context Signal Qualification V1.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--entity-key", required=True)
    ap.add_argument("--season-col", default="season")
    ap.add_argument("--period-col", default="week")
    ap.add_argument("--features", required=True, help="comma-separated numeric feature columns")
    ap.add_argument("--min-pairs", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    features = [x.strip() for x in args.features.split(",") if x.strip()]
    required = [args.entity_key, args.season_col, args.period_col, *features]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"missing required columns: {missing}")
    if df.duplicated([args.entity_key, args.season_col, args.period_col]).any():
        raise SystemExit("duplicate entity-season-period keys; refusing ambiguous stability evidence")

    df = df.sort_values([args.entity_key, args.season_col, args.period_col]).copy()
    rows = []
    for feature in features:
        x = pd.to_numeric(df[feature], errors="coerce")
        prior = x.groupby([df[args.entity_key], df[args.season_col]], sort=False).shift(1)
        valid = x.notna() & prior.notna()
        n = int(valid.sum())
        rho = float(prior[valid].corr(x[valid], method="spearman")) if n >= 2 else float("nan")
        rows.append({
            "feature_name": feature,
            "stability_stat": "strict_prior_adjacent_period_spearman",
            "stability_value": rho,
            "stability_pairs": n,
            "stability_support_ok": bool(n >= args.min_pairs),
            "entity_key": args.entity_key,
            "season_col": args.season_col,
            "period_col": args.period_col,
        })

    out = pd.DataFrame(rows).sort_values("feature_name")
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
