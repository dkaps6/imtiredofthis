#!/usr/bin/env python3
"""Authoritative M70 entrypoint with official-style PBP pass attempts.

nflfastR's play-by-play `pass_attempt` indicator includes sacks. M70 compares
against official player passing attempts/YPA, so using that raw flag changes the
denominator and breaks both passer matching and the physical YPA decomposition.

This wrapper preserves the frozen M70 scientific design and thresholds while
normalizing the PBP attempt flag before the forensic aggregation:
- sacks are not official pass attempts;
- two-point conversion throws are not official pass attempts.

No M59-M65 reconstruction occurs here; the immutable canonical v1 snapshot is
still the only historical projection input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.backtest.audit_qb_extreme_ypa_mechanisms as m

_original_load_pbp = m.load_pbp


def load_pbp_official_attempts(seasons):
    pbp, manifest = _original_load_pbp(seasons)
    x = pbp.copy()

    raw_pass = pd.to_numeric(x.get("pass_attempt", 0), errors="coerce").fillna(0).eq(1)
    sack = pd.to_numeric(
        x.get("sack", pd.Series(0, index=x.index)), errors="coerce"
    ).fillna(0).eq(1)
    two_point = pd.to_numeric(
        x.get("two_point_attempt", pd.Series(0, index=x.index)), errors="coerce"
    ).fillna(0).eq(1)

    official_attempt = raw_pass & ~sack & ~two_point
    x["pass_attempt"] = official_attempt.astype(int)

    manifest = manifest.copy()
    manifest = pd.concat(
        [
            manifest,
            pd.DataFrame(
                [
                    {
                        "season": ",".join(map(str, sorted(set(int(s) for s in seasons)))),
                        "family": "m70_official_attempt_normalization",
                        "status": "sacks_and_two_point_attempts_excluded",
                        "rows": int(official_attempt.sum()),
                    }
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    return x, manifest


m.load_pbp = load_pbp_official_attempts


if __name__ == "__main__":
    raise SystemExit(m.main())
