#!/usr/bin/env python3
"""Report on a real RB PD2 shadow capture session (plan section 7 validation).

Every section 7 proof so far runs on a synthetic fixture. This reports what a
capture session looks like against a real slate, so the questions the fixture
cannot answer get answered before sections 8 and 9 are built on the contract:

- are football keys actually unique per player-game, or does the real
  book/line expansion produce a mismatch?
- is `commence_time` populated at this seam at all?
- how many eligible player-games does the loop drop before the seam?
- does the real alias table remap anyone on a live slate?
- do the persisted NPZ arrays round-trip, and can section 8's frozen transform
  actually run on them?

Read-only. It changes nothing and prints to stdout, because artifact download
out of CI is blocked by the org egress policy -- the job log is the channel.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.research import rb_pd2_shadow_capture_v1 as shadow


def widen_mean_neutral(draws: np.ndarray, width_mult: float) -> np.ndarray:
    """The frozen section 8 transform, reproduced here only to prove it runs."""
    mu = float(np.mean(draws))
    raw = np.maximum(0.0, mu + width_mult * (draws - mu))
    mean_raw = float(np.mean(raw))
    if mean_raw <= 0:
        raise RuntimeError("candidate mean collapsed to zero")
    return raw * (mu / mean_raw)


def latest_session(root: Path) -> Path:
    sessions = sorted(p for p in root.iterdir() if p.is_dir())
    if not sessions:
        raise SystemExit(f"no capture session under {root}")
    return sessions[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(shadow.DEFAULT_ROOT))
    ap.add_argument("--session", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    session_dir = Path(args.session) if args.session else latest_session(root)
    records, receipt = shadow.load_session(session_dir)

    print("=" * 78)
    print("RB PD2 SHADOW CAPTURE -- REAL SLATE SESSION REPORT")
    print("=" * 78)
    print(f"session_dir                : {session_dir}")
    print(f"session_id                 : {receipt['session_id']}")
    print(f"season                     : {receipt['season']}")
    print(f"valid                      : {receipt['valid']}")
    print(f"pricing_rows_seen          : {receipt['pricing_rows_seen']}")
    print(f"duplicate_rows_collapsed   : {receipt['duplicate_rows_collapsed']}")
    print(f"rows_written               : {receipt['rows_written']}")
    print(f"lock_eligible_rows         : {receipt['lock_eligible_rows']}")
    print(f"expected_key_count         : {receipt['expected_key_count']}")
    print(f"missing_expected_keys      : {len(receipt['missing_expected_keys'])}")
    print(f"pre_seam_eligible_count    : {receipt['pre_seam_eligible_count']}")
    print(f"dropped_before_seam_keys   : {len(receipt['dropped_before_seam_keys'])}")
    print(f"sentinels                  : {len(receipt['sentinels'])}")
    for s in receipt["sentinels"][:10]:
        print(f"  - {s['kind']}: {s['football_key']} :: {s['detail'][:120]}")
    if receipt["dropped_before_seam_keys"][:10]:
        print("  dropped before seam (sample):")
        for k in receipt["dropped_before_seam_keys"][:10]:
            print(f"    {k}")

    print("-" * 78)
    print("Q1. Football key uniqueness under real book/line expansion")
    keys = [r["football_key"] for r in records]
    dupes = [k for k, n in Counter(keys).items() if n > 1]
    per_key = [r["duplicate_pricing_rows"] for r in records]
    print(f"  distinct football keys   : {len(set(keys))} of {len(records)} records")
    print(f"  duplicate keys in output : {len(dupes)}  (must be 0)")
    if per_key:
        print(f"  books per player-game    : min={min(per_key)} max={max(per_key)} "
              f"mean={sum(per_key)/len(per_key):.2f}")
    mismatch = [s for s in receipt["sentinels"] if s["kind"] == "duplicate_football_key_mismatch"]
    print(f"  digest mismatches        : {len(mismatch)}  (must be 0)")

    print("-" * 78)
    print("Q2. commence_time availability at this seam")
    have = sum(1 for r in records if str(r.get("commence_time") or "").strip())
    print(f"  populated                : {have} / {len(records)}")
    if have:
        sample = next(r["commence_time"] for r in records if str(r.get("commence_time") or "").strip())
        print(f"  sample value             : {sample!r}")
    else:
        print("  NONE -- section 9 cannot derive a kickoff boundary from the capture alone")
        # Isolate WHICH link drops it rather than guessing: the field is in the
        # carry list at materialize_pricing_offers_v1.py:102, so either the
        # compact frame never had it, or it is lost between the offers file and
        # metrics_ready.
        for label, path in (("props_raw_compact", "outputs/props_raw_compact.csv"),
                            ("props_pricing_offers", "outputs/props_pricing_offers.csv"),
                            ("metrics_ready", "data/metrics_ready.csv")):
            f = Path(path)
            if not f.exists():
                print(f"    {label:<22}: (file absent)")
                continue
            import pandas as pd
            df = pd.read_csv(f, low_memory=False)
            if "commence_time" not in df.columns:
                print(f"    {label:<22}: column ABSENT  (rows={len(df)})")
            else:
                nn = int(df["commence_time"].notna().sum())
                print(f"    {label:<22}: column present, non-null {nn}/{len(df)}")

    print("-" * 78)
    print("Q3. Alias remapping on a real slate")
    remapped = [r for r in records if r.get("player_clean_key_remapped")]
    print(f"  remapped player-games    : {len(remapped)} / {len(records)}")
    for r in remapped[:10]:
        print(f"    {r['player']!r}: {r['player_clean_key_source']!r} -> {r['player_clean_key']!r}")
    prov = receipt["provenance"]
    for label, field in (("overrides", "manual_name_overrides_sha256"), ("roles", "roles_ourlads_sha256")):
        digest = prov.get(field) or ""
        if digest:
            print(f"  fingerprint {label:<9}    : {digest[:16]}…")
        else:
            # An empty digest means the file was absent when the session opened.
            # Section 9 must treat that as UNPROVABLE, never as a match against
            # another empty digest, or two runs with no alias table would join.
            print(f"  fingerprint {label:<9}    : (EMPTY -- file absent; section 9 must not treat this as a match)")

    print("-" * 78)
    print("Q4. NPZ round-trip and section 8 feasibility on the real arrays")
    checked = failed = 0
    for r in records:
        try:
            draws = shadow.load_draws(session_dir, r)
            cand = widen_mean_neutral(draws, 1.30)
            assert cand.size == draws.size
            assert np.isfinite(cand).all() and (cand >= 0).all()
            assert abs(float(np.mean(cand)) - float(np.mean(draws))) <= 1e-8
            checked += 1
        except Exception as exc:
            failed += 1
            print(f"  FAIL {r['football_key']}: {type(exc).__name__}: {exc}")
    print(f"  round-tripped + transformed: {checked} / {len(records)}   failures: {failed}")
    if records:
        d = shadow.load_draws(session_dir, records[0])
        print(f"  sample draw_count        : {d.size}")
        print(f"  sample mean/sd           : {float(np.mean(d)):.4f} / {float(np.std(d, ddof=1)):.4f}")

    print("-" * 78)
    print("Q5. Weeks and teams covered")
    print(f"  weeks                    : {sorted({int(r['week']) for r in records})}")
    print(f"  teams                    : {len(sorted({r['team'] for r in records}))}")
    print(f"  positions                : {sorted({r['position'] for r in records})}")

    print("=" * 78)
    verdict = "PASS" if (not dupes and not mismatch and failed == 0 and receipt["valid"]) else "ATTENTION"
    print(f"VERDICT: {verdict}")
    print("=" * 78)
    print(json.dumps({"session_id": receipt["session_id"], "verdict": verdict,
                      "records": len(records), "valid": receipt["valid"],
                      "duplicate_keys": len(dupes), "digest_mismatches": len(mismatch),
                      "roundtrip_failures": failed,
                      "commence_time_populated": have,
                      "dropped_before_seam": len(receipt["dropped_before_seam_keys"])},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
