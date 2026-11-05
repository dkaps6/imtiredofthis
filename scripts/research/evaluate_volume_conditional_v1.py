"""PRISM — volume-conditional recalibration of ensemble_proj.
Preregistered: docs/research/PRISM_VOLUME_CONDITIONAL_PREREGISTRATION_V1.md

Fit 2024 / blind 2025, per market. Conditioner is the model's OWN output
(AGENTS.md section 3 forbids vegas_line in the football projection).
pass_yards is an absolute positive control.
"""
from __future__ import annotations
import json, sys
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

RNG = np.random.default_rng(20261104)
HIST = 'docs/research/overnight/clean_v1_full_stack_vegas_benchmark_detail.csv'
MARKETS = ['rec_yards', 'receptions', 'rush_yards', 'rush_rec_yards', 'pass_yards']
CONTROL = 'pass_yards'
FORBIDDEN = ['line', 'over_odds', 'under_odds', 'p_over', 'p_under', 'ev_over', 'ev_under',
             'edge_pct', 'edge_abs', 'over_novig', 'under_novig', 'side', 'signal',
             'bet_result', 'unit_result', 'vegas_error', 'best_ev', 'market_prob', 'fair_prob']
gates = {}


def g(tag, ok, msg):
    gates[tag] = {'pass': bool(ok), 'msg': msg}
    print(f'  [{tag}] {"PASS" if ok else "FAIL"} — {msg}')
    return ok


h = pd.read_csv(HIST, low_memory=False)
h = h[h['market'].isin(MARKETS)].copy()
for c in ['ensemble_proj', 'actual']:
    h[c] = pd.to_numeric(h[c], errors='coerce')
opp = pd.to_numeric(h.get('actual_opportunities'), errors='coerce')
h = h[~((opp.fillna(0) <= 0) & (h['actual'].fillna(0) <= 0))]
h = h.dropna(subset=['ensemble_proj', 'actual'])
h['player'] = h['player_clean_key'].astype(str)

used = ['ensemble_proj', 'actual', 'market', 'season', 'player']
g('M1', not [c for c in FORBIDDEN if c in used],
  f'conditioner=ensemble_proj target=actual; {len(FORBIDDEN)} forbidden columns absent from {used}')
g('M2', set(h[h.season == 2024]['season']) == {2024}, 'fits are 2024 only; 2025 is blind')


def spreads(proj, act, cuts, strat=None):
    """Three readings of the compression spread. Risk 1 said the robust one governs.

    `strat` is the BASELINE projection that defines the volume stratum. Using
    the corrected projection to assign strata would move rows between bins and
    measure something else entirely.
    """
    qt = np.digitize(proj if strat is None else strat, cuts)
    lo_m = qt == 0
    hi_m = qt == 3
    out = {}
    pos = act > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(pos, proj / np.where(pos, act, 1), np.nan)
    out['mean_of_ratios'] = float(np.nanmean(ratio[hi_m]) - np.nanmean(ratio[lo_m]))
    out['median_of_ratios'] = float(np.nanmedian(ratio[hi_m]) - np.nanmedian(ratio[lo_m]))
    out['mean_over_mean'] = float(proj[hi_m].mean() / max(act[hi_m].mean(), 1e-9)
                                  - proj[lo_m].mean() / max(act[lo_m].mean(), 1e-9))
    return out


def boot(pa, pb, idx, up, y, reps=2500):
    da = {u: np.abs(pa[idx[u]] - y[idx[u]]).mean() for u in up}
    db = {u: np.abs(pb[idx[u]] - y[idx[u]]).mean() for u in up}
    dv = np.array([da[u] - db[u] for u in up])
    bs = np.array([dv[RNG.integers(0, len(up), len(up))].mean() for _ in range(reps)])
    return float(dv.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


results = {}
for m in MARKETS:
    f = h[(h.market == m) & (h.season == 2024)]
    v = h[(h.market == m) & (h.season == 2025)]
    if len(v) < 100 or v['player'].nunique() < 40:
        print(f'\n--- {m}: blind n={len(v)} players={v["player"].nunique()} -> K5, not promoted')
        continue
    xf, yf = f['ensemble_proj'].to_numpy(float), f['actual'].to_numpy(float)
    xv, yv = v['ensemble_proj'].to_numpy(float), v['actual'].to_numpy(float)
    cuts = np.quantile(xf, [.25, .5, .75])

    lin = LinearRegression().fit(xf.reshape(-1, 1), yf)
    iso = IsotonicRegression(out_of_bounds='clip').fit(xf, yf)
    qf = np.digitize(xf, cuts)
    mult = {q: (yf[qf == q].mean() / max(xf[qf == q].mean(), 1e-9)) if (qf == q).sum() > 20 else 1.0
            for q in range(4)}
    qv = np.digitize(xv, cuts)

    pred = {'V0_incumbent': xv,
            'V1_linear': lin.predict(xv.reshape(-1, 1)),
            'V2_tiered': xv * np.array([mult[q] for q in qv]),
            'V3_isotonic': iso.predict(xv)}

    pl = v['player'].to_numpy(); up = np.unique(pl)
    idx = {u: np.where(pl == u)[0] for u in up}
    ppm = lambda p: float(np.mean([np.abs(p[idx[u]] - yv[idx[u]]).mean() for u in up]))

    print(f'\n--- {m}   fit {len(f)} / blind {len(v)} ({len(up)} players)   '
          f'V1: a={lin.intercept_:+.3f} b={lin.coef_[0]:.4f}   V2 mult={[round(mult[q],3) for q in range(4)]}')
    s0 = spreads(xv, yv, cuts, strat=xv)
    print(f'    {"arm":16s}{"ppMAE":>9s}{"level":>8s}{"spread(mean)":>14s}{"spread(med)":>13s}{"spread(m/m)":>13s}')
    rec = {}
    for k, p in pred.items():
        s = spreads(p, yv, cuts, strat=xv)
        print(f'    {k:16s}{ppm(p):9.4f}{p.mean()/yv.mean():8.4f}'
              f'{s["mean_of_ratios"]:14.4f}{s["median_of_ratios"]:13.4f}{s["mean_over_mean"]:13.4f}')
        rec[k] = {'ppmae': ppm(p), 'level': float(p.mean() / yv.mean()), **s}

    for arm in ['V1_linear', 'V2_tiered', 'V3_isotonic']:
        d = boot(pred[arm], pred['V0_incumbent'], idx, up, yv)
        # S1 on the robust (median) reading, per named risk 1
        s1 = (abs(rec[arm]['median_of_ratios']) <= 0.6 * abs(rec['V0_incumbent']['median_of_ratios']))
        s2 = not (d[1] > 0)
        s4 = abs(rec[arm]['level'] - 1.0) <= 0.05
        worst = -9.9; wq = ''
        for q in range(4):
            k = qv == q
            if k.sum() < 20: continue
            m0 = np.abs(pred['V0_incumbent'][k] - yv[k]).mean()
            m1 = np.abs(pred[arm][k] - yv[k]).mean()
            rel = (m1 - m0) / max(m0, 1e-9)
            if rel > worst: worst, wq = rel, f'Q{q}'
        s3 = worst <= 0.05
        rec[arm].update({'mae_delta': d, 'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4,
                         'worst_quartile_rel': float(worst), 'worst_quartile': wq})
        print(f'      {arm:14s} dMAE={d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]  '
              f'S1={"P" if s1 else "f"} S2={"P" if s2 else "f"} '
              f'S3={"P" if s3 else "f"}({worst*100:+.1f}% {wq}) S4={"P" if s4 else "f"}')
    results[m] = rec

cr = results.get(CONTROL)
print()
if cr is None:
    g('M4', False, f'{CONTROL} not evaluable; control unverifiable')
else:
    dead = [a for a in ['V1_linear', 'V2_tiered', 'V3_isotonic'] if cr[a]['mae_delta'][1] > 0]
    g('M4', not dead, f'{CONTROL}: arms damaging it = {dead or "none"}')

m4dead = set()
if cr:
    m4dead = {a for a in ['V1_linear', 'V2_tiered', 'V3_isotonic'] if cr[a]['mae_delta'][1] > 0}

print('\n=== PROMOTION (preregistered: S1 and S2 and S3 and S4, M4 absolute) ===')
print(f'{"market":16s} {"arm":14s} {"S1":>3s} {"S2":>3s} {"S3":>3s} {"S4":>3s}  verdict')
any_p = False
for m, rec in results.items():
    for a in ['V1_linear', 'V2_tiered', 'V3_isotonic']:
        r = rec[a]
        ok = r['S1'] and r['S2'] and r['S3'] and r['S4'] and a not in m4dead
        any_p = any_p or ok
        print(f'{m:16s} {a:14s} {"P" if r["S1"] else "f":>3s} {"P" if r["S2"] else "f":>3s} '
              f'{"P" if r["S3"] else "f":>3s} {"P" if r["S4"] else "f":>3s}  '
              f'{"PROMOTE" if ok else "no"}')
if m4dead:
    print(f'\n  K2: {sorted(m4dead)} damaged the positive control and are dead everywhere.')
print(f'\n=== ANY PROMOTION: {"YES" if any_p else "NO"} ===')
json.dump({'gates': gates, 'markets': results, 'm4_dead': sorted(m4dead), 'any': any_p},
          open('data/research/volume_conditional_v1_results.json', 'w'), indent=2, default=float)
print('wrote data/research/volume_conditional_v1_results.json')
