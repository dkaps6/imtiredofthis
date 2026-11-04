"""PRISM — ensemble method: should the sum-to-one constraint be relaxed?
Preregistered: docs/research/PRISM_ENSEMBLE_METHOD_PREREGISTRATION_V1.md

Fit 2024 / blind 2025, per market. Clustered over players. pass_yards is an
absolute positive control: an arm that damages it is dead for every market.
Writes nothing to production.
"""
from __future__ import annotations
import json, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, '.')
from scripts.modeling.ensemble_v2 import _normalise_nonnegative

RNG = np.random.default_rng(20261104)
HIST = 'docs/research/overnight/clean_v1_full_stack_vegas_benchmark_detail.csv'
COMP = ['mc_proj', 'ml_proj', 'state_proj']
CONTROL = 'pass_yards'
FORBIDDEN = ['line', 'over_odds', 'under_odds', 'p_over', 'p_under', 'ev_over', 'ev_under',
             'edge_pct', 'edge_abs', 'over_novig', 'under_novig', 'bet_result', 'unit_result',
             'side', 'signal', 'vegas_error', 'over_implied', 'under_implied', 'best_ev']
gates = {}


def g(tag, ok, msg):
    gates[tag] = {'pass': bool(ok), 'msg': msg}
    print(f'  [{tag}] {"PASS" if ok else "FAIL"} — {msg}')
    return ok


h = pd.read_csv(HIST, low_memory=False)
for c in COMP + ['actual']:
    h[c] = pd.to_numeric(h[c], errors='coerce')
opp = pd.to_numeric(h.get('actual_opportunities'), errors='coerce')
h = h[~((opp.fillna(0) <= 0) & (h['actual'].fillna(0) <= 0))]
h = h.dropna(subset=COMP + ['actual'])
h['player'] = h['player_clean_key'].astype(str)
MARKETS = sorted(h['market'].unique())

g('M1', not [c for c in FORBIDDEN if c in COMP + ['actual']],
  f'target=actual, features={COMP}; {len(FORBIDDEN)} forbidden columns absent')

inc = pd.read_csv('data/model_ensemble_weights.csv')
incw = {str(r['market']).lower(): np.array([r['mc_weight'], r['ml_weight'], r['state_weight']], float)
        for _, r in inc.iterrows()}
print(f'  incumbent rows: {sorted(incw)}')
print(f'  markets in this dataset: {MARKETS}')
missing = [m for m in incw if m not in MARKETS]
print(f'  NOT evaluable here (left untouched per risk 4): {missing}')

fits, m2ok = {}, True
for m in MARKETS:
    f = h[(h['market'] == m) & (h['season'] == 2024)]
    if set(f['season']) != {2024}:
        m2ok = False
    X, y = f[COMP].to_numpy(float), f['actual'].to_numpy(float)
    r1 = LinearRegression(fit_intercept=False, positive=True).fit(X, y)
    r3 = LinearRegression(fit_intercept=True, positive=True).fit(X, y)
    fits[m] = {'C1': _normalise_nonnegative(r1.coef_), 'C2': r1.coef_.copy(),
               'C3': (r3.coef_.copy(), float(r3.intercept_)), 'n': len(f),
               'players': int(f['player'].nunique())}
g('M2', m2ok, 'every fit frame is 2024 only; 2025 untouched by all fits')

# M3: forcing C2 to sum to 1 with no intercept must reproduce C1 exactly
m3 = max(float(np.max(np.abs(_normalise_nonnegative(fits[m]['C2']) - fits[m]['C1']))) for m in MARKETS)
g('M3', m3 <= 1e-12, f'max |normalise(C2) - C1| = {m3:.2e} across {len(MARKETS)} markets')


def arms_for(m, X):
    f = fits[m]
    out = {'C1_refit_norm': X @ f['C1'], 'C2_refit_unnorm': X @ f['C2'],
           'C3_unnorm_intercept': X @ f['C3'][0] + f['C3'][1]}
    w0 = incw.get(m)
    out['C0_incumbent'] = X @ w0 if w0 is not None else X[:, 0]
    return out


def boot(pa, pb, idx, up, y, reps=3000):
    da = {u: np.abs(pa[idx[u]] - y[idx[u]]).mean() for u in up}
    db = {u: np.abs(pb[idx[u]] - y[idx[u]]).mean() for u in up}
    dv = np.array([da[u] - db[u] for u in up])
    bs = np.array([dv[RNG.integers(0, len(up), len(up))].mean() for _ in range(reps)])
    return float(dv.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


results = {}
print('\n' + '=' * 108)
for m in MARKETS:
    v = h[(h['market'] == m) & (h['season'] == 2025)]
    if len(v) < 60:
        print(f'{m}: only {len(v)} blind rows — skipped'); continue
    X, y = v[COMP].to_numpy(float), v['actual'].to_numpy(float)
    pl = v['player'].to_numpy(); up = np.unique(pl)
    idx = {u: np.where(pl == u)[0] for u in up}
    pr = arms_for(m, X)
    f = fits[m]
    print(f'\n--- {m}   fit n={f["n"]} ({f["players"]} players) | blind n={len(v)} ({len(up)} players)')
    print(f'    C1 norm   w={np.round(f["C1"],4)} sum={f["C1"].sum():.4f}')
    print(f'    C2 unnorm w={np.round(f["C2"],4)} sum={f["C2"].sum():.4f}   <-- the discarded scale')
    print(f'    C3 +icpt  w={np.round(f["C3"][0],4)} sum={f["C3"][0].sum():.4f} b={f["C3"][1]:+.3f}')
    if m in incw:
        print(f'    C0 incumb w={np.round(incw[m],4)} sum={incw[m].sum():.4f}')

    def ppm(p):
        return float(np.mean([np.abs(p[idx[u]] - y[idx[u]]).mean() for u in up]))
    print(f'    {"arm":22s}{"per-player MAE":>16s}{"pooled MAE":>12s}{"level":>9s}')
    for k in ['C0_incumbent', 'C1_refit_norm', 'C2_refit_unnorm', 'C3_unnorm_intercept']:
        print(f'    {k:22s}{ppm(pr[k]):16.4f}{np.abs(pr[k]-y).mean():12.4f}{pr[k].mean()/y.mean():9.4f}')

    d_mae = boot(pr['C2_refit_unnorm'], pr['C0_incumbent'], idx, up, y)
    lv0 = abs(pr['C0_incumbent'].mean() / y.mean() - 1)
    lv1 = abs(pr['C2_refit_unnorm'].mean() / y.mean() - 1)
    lvc1 = abs(pr['C1_refit_norm'].mean() / y.mean() - 1)
    s1 = d_mae[2] < 0
    s1n = not (d_mae[1] > 0)
    s2 = (lv0 - lv1) >= 0.05
    s3 = (lvc1 - lv1) >= 0.05
    k6 = 0.7 <= f['C2'].sum() <= 1.5
    print(f'    S1  superiority  d={d_mae[0]:+.4f} [{d_mae[1]:+.4f},{d_mae[2]:+.4f}] -> {"PASS" if s1 else "ns/fail"} (reported only)')
    print(f'    S1N non-inferior                                     -> {"PASS" if s1n else "FAIL"}')
    print(f'    S2  level  |lv-1| {lv0:.4f} -> {lv1:.4f}  gain {(lv0-lv1)*100:+.1f}pp -> {"PASS" if s2 else "FAIL"}')
    print(f'    S3  vs C1  |lv-1| {lvc1:.4f} -> {lv1:.4f}  gain {(lvc1-lv1)*100:+.1f}pp -> {"PASS" if s3 else "FAIL"}')
    print(f'    K6  weight sum {f["C2"].sum():.4f} in [0.7,1.5] -> {"PASS" if k6 else "FAIL"}')

    q = np.quantile(pd.to_numeric(h[(h["market"]==m)&(h["season"]==2024)]['line'], errors='coerce').dropna(), [.25,.5,.75])
    vq = np.digitize(pd.to_numeric(v['line'], errors='coerce').fillna(-1).to_numpy(), q)
    worst, wc = -9.9, ''
    for qq in sorted(set(vq)):
        k = vq == qq
        if k.sum() < 20:
            continue
        m0 = np.abs(pr['C0_incumbent'][k] - y[k]).mean()
        m1 = np.abs(pr['C2_refit_unnorm'][k] - y[k]).mean()
        rel = (m1 - m0) / max(m0, 1e-9)
        print(f'      Q{qq} n={int(k.sum()):4d} C0={m0:7.3f} -> C2={m1:7.3f} {rel*100:+6.1f}%')
        if rel > worst:
            worst, wc = rel, f'Q{qq}'
    s4 = worst <= 0.05
    print(f'    S4  worst quartile {worst*100:+.1f}% at {wc} vs +5% bar -> {"PASS" if s4 else "FAIL"}')

    results[m] = {'fit_n': f['n'], 'blind_n': len(v), 'blind_players': int(len(up)),
                  'w_C1': f['C1'].tolist(), 'w_C2': f['C2'].tolist(),
                  'w_C2_sum': float(f['C2'].sum()),
                  'w_C3': f['C3'][0].tolist(), 'icpt_C3': f['C3'][1],
                  'w_C0': incw.get(m, np.array([1.0, 0, 0])).tolist(),
                  'ppm': {k: ppm(p) for k, p in pr.items()},
                  'pooled': {k: float(np.abs(p - y).mean()) for k, p in pr.items()},
                  'level': {k: float(p.mean() / y.mean()) for k, p in pr.items()},
                  'S1_delta_ci': list(d_mae),
                  'S1': s1, 'S1N': s1n, 'S2': s2, 'S3': s3, 'S4': s4, 'K6': k6,
                  'worst_quartile_rel': float(worst),
                  'promote': bool(s1n and s2 and s3 and s4 and k6)}

# ---------------------------------------------------------------- M4 absolute control
print('\n' + '=' * 108)
cr = results.get(CONTROL)
if cr is None:
    g('M4', False, f'{CONTROL} not evaluable — control cannot be verified, arm disqualified')
else:
    dmg = cr['S1_delta_ci'][1] > 0
    g('M4', not dmg, f'{CONTROL} C2 vs C0 per-player MAE d={cr["S1_delta_ci"][0]:+.4f} '
                     f'[{cr["S1_delta_ci"][1]:+.4f},{cr["S1_delta_ci"][2]:+.4f}] — '
                     f'{"DAMAGED, C2 DEAD FOR EVERY MARKET (K2)" if dmg else "not damaged"}')

m4 = gates['M4']['pass']
print('\n=== PROMOTION DECISION (preregistered rule, §4) ===')
print(f'{"market":18s} {"S1N":>5s} {"S2":>4s} {"S3":>4s} {"S4":>4s} {"K6":>4s} {"sum":>7s} {"level C0->C2":>18s}  verdict')
any_p = False
for m, r in results.items():
    ok = r['promote'] and m4
    any_p = any_p or ok
    print(f'{m:18s} {str(r["S1N"]):>5s} {str(r["S2"]):>4s} {str(r["S3"]):>4s} {str(r["S4"]):>4s} '
          f'{str(r["K6"]):>4s} {r["w_C2_sum"]:7.4f} '
          f'{r["level"]["C0_incumbent"]:7.4f}->{r["level"]["C2_refit_unnorm"]:7.4f}  '
          f'{"PROMOTE" if ok else "no"}')
if not m4:
    print('\n  K2 FIRED: the positive control is damaged, so NOTHING is promoted regardless of the above.')
print(f'\n=== ANY PROMOTION: {"YES" if any_p else "NO"} ===')

out = {'gates': gates, 'markets': results, 'control_ok': m4, 'any_promotion': any_p,
       'not_evaluable': missing}
with open('data/research/ensemble_method_v1_results.json', 'w') as fh:
    json.dump(out, fh, indent=2, default=float)
print('wrote data/research/ensemble_method_v1_results.json')
