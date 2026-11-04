"""Allocator shape V2: mass-preserving concentration on top of E1.
Preregistration: docs/research/PRISM_ALLOCATOR_SHAPE_PREREGISTRATION_V2.md (frozen first).
Scores the ALLOCATION RULE in expectation. No MC run, no production code modified.

V1 was mis-specified: share^g x A with no renormalisation deflates every player because the
production normaliser only scales DOWN. See PRISM_ALLOCATOR_SHAPE_RESULTS_V1_MISSPECIFIED.md.
Here the team sum is restored to E1's before the production normaliser, matching the contract
_sharpen_wr_target_shares already states.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

sys.path.insert(0, "scripts/research")
from evaluate_mc_leak_fixes_v1 import build, c1_normalise  # noqa: E402

GRID = np.round(np.arange(1.00, 2.51, 0.05), 2)
TIERS = [(0.0, 0.05, "<5%"), (0.05, 0.12, "5-12%"), (0.12, 0.20, "12-20%"), (0.20, 1.01, "20%+")]
S2_GAMMA, MIN_PLAYERS, MIN_WEEKS, ITERS, SEED = 1.25, 100, 2, 10000, 47


def shape(e: pd.Series, team: pd.Series, gamma: float) -> pd.Series:
    """e^gamma renormalised so each team sum equals sum(e). MASS-PRESERVING."""
    w = np.power(e.clip(lower=0.0).astype(float), float(gamma))
    se = e.groupby(team).transform("sum")
    sw = w.groupby(team).transform("sum")
    scaled = w * (se / sw.where(sw > 0, np.nan))
    return scaled.fillna(0.0)


def per_player(df: pd.DataFrame, col: str) -> pd.Series:
    """bias_p = mean over the player's evaluated weeks of (alloc - actual)."""
    return (df[col] - df.actual).groupby(df.k).mean()


def main() -> int:
    a = build()
    key = a.team + "_" + a.wk.astype(str)
    a["E1"] = c1_normalise(a.share * a.A, key)
    e_raw = (a.share * a.A).clip(lower=0.0)

    # ---- mechanical precondition: mass preservation on every team-week, every gamma ----
    worst = 0.0
    for g in GRID:
        w = shape(e_raw, key, g)
        d = (w.groupby(key).sum() - e_raw.groupby(key).sum()).abs().max()
        worst = max(worst, float(d))
    print(f"[check] mass preservation, worst |team sum - E1 team sum| over grid: {worst:.3e}")
    if worst > 1e-9:
        print("K8 VOID: mass not preserved"); return 1

    # ---- mechanical precondition: nesting at gamma = 1.00 ----
    nest = float((c1_normalise(shape(e_raw, key, 1.00), key) - a.E1).abs().max())
    print(f"[check] nesting at gamma=1.00, worst |arm - E1| per row: {nest:.3e}")
    if nest > 1e-12:
        print("K8 VOID: gamma=1 does not reproduce E1"); return 1

    # ---- panel ----
    cnt = a.groupby("k").wk.nunique()
    panel = set(cnt[cnt >= MIN_WEEKS].index)
    tgt = a.groupby("k").act_tgt.sum()
    evaluable = sorted(panel & set(tgt[tgt > 0].index))
    print(f"[panel] {len(panel)} players with >={MIN_WEEKS} weeks; "
          f"{len(evaluable)} ever-targeted (decision population)")
    if len(panel) < MIN_PLAYERS:
        print("K-sample: below the preregistered 100-player minimum"); return 1

    # ---- S1': gamma fitted leave-one-week-out ----
    fits = {}
    for fit_wk, app_wk in ((7, 8), (8, 7)):
        m = a.wk == fit_wk
        sub, ksub = a[m], key[m]
        e_sub = e_raw[m]
        best, bg = np.inf, None
        for g in GRID:
            col = c1_normalise(shape(e_sub, ksub, g), ksub)
            tmp = sub.assign(_x=col)
            b = per_player(tmp, "_x")
            b = b[b.index.isin(tgt[tgt > 0].index)]
            v = float(b.abs().mean())
            if v < best: best, bg = v, g
        fits[app_wk] = bg
        print(f"[S1'] fit on W{fit_wk} -> gamma={bg:.2f} (applied to W{app_wk}), "
              f"fitting-week M1={best:.5f}")

    a["S1"] = 0.0
    for wk, g in fits.items():
        m = a.wk == wk
        a.loc[m, "S1"] = c1_normalise(shape(e_raw[m], key[m], g), key[m])
    a["S2"] = c1_normalise(shape(e_raw, key, S2_GAMMA), key)

    arms = ["C1", "E1", "S1", "S2"]
    a["C1"] = c1_normalise(a.share, key)
    bias = {c: per_player(a, c) for c in arms}
    dec = [k for k in evaluable]

    # ---- M1 + G1 ----
    print("\n=== M1  mean over players of |bias_p|, ever-targeted only "
          f"(n={len(dec)}) ===")
    rng = np.random.default_rng(SEED)
    idx = np.array(dec)
    draws = rng.integers(0, len(idx), (ITERS, len(idx)))
    base = bias["E1"].reindex(idx).to_numpy()
    for c in arms:
        v = bias[c].reindex(idx).to_numpy()
        gain = np.abs(base) - np.abs(v)            # >0 means the arm beats E1
        bs = np.abs(v)[draws].mean(1)
        gb = gain[draws].mean(1)
        lo, hi = np.percentile(gb, [2.5, 97.5])
        tag = "" if c == "E1" else (
            "PASS" if lo > 0 else "FAIL")
        print(f"  {c:3s} M1={np.abs(v).mean():.5f} "
              f"[{np.percentile(bs,2.5):.5f},{np.percentile(bs,97.5):.5f}]  "
              f"gain vs E1={gain.mean():+.5f} [{lo:+.5f},{hi:+.5f}]  G1 {tag}")

    # ---- M2 breadth + G2 ----
    print("\n=== M2  breadth vs E1 (ever-targeted) ===")
    for c in ("S1", "S2"):
        v = bias[c].reindex(idx).to_numpy()
        imp = (np.abs(v) < np.abs(base)).astype(float)
        bs = imp[draws].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {c}: {imp.mean()*100:.1f}% improved [{lo*100:.1f}%,{hi*100:.1f}%]  "
              f"G2 {'PASS' if imp.mean()>=0.60 and lo>0.50 else 'FAIL'}")

    # ---- G3 starved starters ----
    print("\n=== G3  players who drew >=1 target but got zero allocation ===")
    drew = a[a.act_tgt >= 1]
    for c in arms:
        z = float((drew[c] <= 0).mean())
        print(f"  {c}: {z*100:.2f}%  {'PASS' if z<=0.05 else 'FAIL'}")

    # ---- M3 tier table + G4/G6 ----
    print("\n=== M3  tier table, binned on ACTUAL share (matches the E1 table) ===")
    print(f"  {'tier':8s} {'n':>5s} {'actual':>8s} " + " ".join(f"{c:>9s}" for c in arms)
          + "   gain(S1)  gain(S2)")
    g4 = {"S1": [], "S2": []}
    for lo_t, hi_t, lab in TIERS:
        m = (a.actual >= lo_t) & (a.actual < hi_t)
        sub = a[m]
        if sub.empty: continue
        row = f"  {lab:8s} {len(sub):5d} {sub.actual.mean():8.4f} "
        row += " ".join(f"{(sub[c]-sub.actual).mean():+9.4f}" for c in arms)
        gE = (sub.E1 - sub.actual).abs().mean()
        gains = {}
        for c in ("S1", "S2"):
            gains[c] = gE - (sub[c] - sub.actual).abs().mean()
            g4[c].append((lab, gains[c]))
        print(row + f"   {gains['S1']:+8.4f}  {gains['S2']:+8.4f}")
    print("\n  G4 (no tier harmed, gain >= -0.002 vs E1):")
    for c in ("S1", "S2"):
        bad = [(l, v) for l, v in g4[c] if v < -0.002]
        print(f"    {c}: {'PASS' if not bad else 'FAIL on ' + ', '.join(f'{l} {v:+.4f}' for l,v in bad)}")
    print("\n  G6 (20%+ bias toward zero, no overshoot):")
    m = a.actual >= 0.20
    bE = (a[m].E1 - a[m].actual).mean()
    for c in ("S1", "S2"):
        bA = (a[m][c] - a[m].actual).mean()
        ok = abs(bA) < abs(bE) and bA <= 0.0
        print(f"    {c}: E1 {bE:+.4f} -> {bA:+.4f}  {'PASS' if ok else 'FAIL'}")

    # ---- G5 week agreement ----
    print("\n=== G5  per-week, scored separately (ever-targeted) ===")
    for wk in (7, 8):
        sub = a[a.wk == wk]
        bw = {c: per_player(sub, c) for c in arms}
        ks = [k for k in bw["E1"].index if k in set(dec)]
        bE = bw["E1"].reindex(ks).abs().to_numpy()
        out = []
        for c in ("S1", "S2"):
            out.append(f"{c} gain={bE.mean()-bw[c].reindex(ks).abs().to_numpy().mean():+.5f}")
        print(f"  W{wk} (n={len(ks)}): " + "  ".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
