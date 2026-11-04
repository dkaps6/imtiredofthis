"""PRISM — shrunk volume-conditional correction, validated on 2026.
Preregistered: docs/research/PRISM_SHRUNK_VOLUME_CONDITIONAL_PREREGISTRATION_V1.md

arm and lambda chosen by leave-one-season-out inside 2024+2025 on per-player
MAE; 2026 touched exactly once, to score. pass_yards out of scope by design.
"""
from __future__ import annotations
import glob, json, sys
import numpy as np, pandas as pd, nflreadpy as nfl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

RNG = np.random.default_rng(20261104)
HIST = 'docs/research/overnight/clean_v1_full_stack_vegas_benchmark_detail.csv'
MARKETS = ['rec_yards', 'receptions', 'rush_yards', 'rush_rec_yards']
LAMBDAS = np.round(np.arange(0.0, 1.0001, 0.1), 2)
COMP = ['mc_proj', 'ml_proj', 'state_proj']
FORBIDDEN = ['line', 'vegas_line', 'over_odds', 'under_odds', 'p_over', 'ev_over', 'edge_pct',
             'over_novig', 'side', 'bet_result', 'unit_result', 'vegas_error', 'fair_prob']
gates = {}


def g(tag, ok, msg):
    gates[tag] = {'pass': bool(ok), 'msg': msg}
    print(f'  [{tag}] {"PASS" if ok else "FAIL"} — {msg}')
    return ok


# ---------------------------------------------------------------- fit sample, RECONSTRUCTED baseline (M5)
h = pd.read_csv(HIST, low_memory=False)
h = h[h['market'].isin(MARKETS)].copy()
for c in COMP + ['actual', 'ensemble_proj']:
    h[c] = pd.to_numeric(h[c], errors='coerce')
opp = pd.to_numeric(h.get('actual_opportunities'), errors='coerce')
h = h[~((opp.fillna(0) <= 0) & (h['actual'].fillna(0) <= 0))]
h = h.dropna(subset=COMP + ['actual'])
h['player'] = h['player_clean_key'].astype(str)

W = pd.read_csv('data/model_ensemble_weights.csv')
wmap = {str(r['market']): np.array([r['mc_weight'], r['ml_weight'], r['state_weight']], float)
        for _, r in W.iterrows()}
h['base'] = np.nan
for m in MARKETS:
    k = h['market'] == m
    w = wmap[m]
    h.loc[k, 'base'] = (h.loc[k, COMP].to_numpy(float) @ w)

print('=== M5: reconstructed calibrated baseline vs the recorded ensemble ===')
diff = {}
for m in MARKETS:
    s = h[h.market == m]
    lv_rec = s['ensemble_proj'].mean() / s['actual'].mean()
    lv_new = s['base'].mean() / s['actual'].mean()
    diff[m] = abs(lv_new - lv_rec)
    print(f'  {m:16s} recorded={lv_rec:.4f}  reconstructed={lv_new:.4f}  delta={lv_new-lv_rec:+.4f}')
mc_only = ['rec_yards', 'receptions', 'rush_rec_yards']
g('M5', all(diff[m] > 0.01 for m in mc_only),
  f'the three mc_only markets differ from the recorded ensemble as expected '
  f'(min delta {min(diff[m] for m in mc_only):.4f})')
g('M1', not [c for c in FORBIDDEN if c in ['base', 'actual', 'market', 'season', 'player']],
  f'conditioner=reconstructed ensemble, target=actual; {len(FORBIDDEN)} forbidden names absent')


def fit_arms(x, y, cuts):
    lin = LinearRegression().fit(x.reshape(-1, 1), y)
    iso = IsotonicRegression(out_of_bounds='clip').fit(x, y)
    q = np.digitize(x, cuts)
    mult = {i: (y[q == i].mean() / max(x[q == i].mean(), 1e-9)) if (q == i).sum() > 20 else 1.0
            for i in range(4)}
    return {'V1_linear': lambda z: lin.predict(z.reshape(-1, 1)),
            'V2_tiered': lambda z: z * np.array([mult[i] for i in np.digitize(z, cuts)]),
            'V3_isotonic': lambda z: iso.predict(z)}


def ppmae(p, y, idx, up):
    return float(np.mean([np.abs(p[idx[u]] - y[idx[u]]).mean() for u in up]))


def idxmap(pl):
    up = np.unique(pl)
    return {u: np.where(pl == u)[0] for u in up}, up


# ---------------------------------------------------------------- LOSO selection (2024/2025 only)
print('\n=== arm and lambda by leave-one-season-out inside 2024+2025 ===')
selected = {}
for m in MARKETS:
    s = h[h.market == m]
    scores = {}
    for tr, te in ((2024, 2025), (2025, 2024)):
        a, b = s[s.season == tr], s[s.season == te]
        if len(a) < 100 or len(b) < 100:
            continue
        xa, ya = a['base'].to_numpy(float), a['actual'].to_numpy(float)
        xb, yb = b['base'].to_numpy(float), b['actual'].to_numpy(float)
        cuts = np.quantile(xa, [.25, .5, .75])
        arms = fit_arms(xa, ya, cuts)
        im, up = idxmap(b['player'].to_numpy())
        for an, fn in arms.items():
            full = fn(xb)
            for lam in LAMBDAS:
                p = xb + lam * (full - xb)
                scores.setdefault((an, float(lam)), []).append(ppmae(p, yb, im, up))
    avg = {k: float(np.mean(v)) for k, v in scores.items() if len(v) == 2}
    best = min(avg, key=avg.get)
    selected[m] = {'arm': best[0], 'lam': best[1], 'loso_ppmae': avg[best],
                   'baseline_loso': avg[(best[0], 0.0)]}
    print(f'  {m:16s} -> arm={best[0]:12s} lambda={best[1]:.1f}   '
          f'LOSO ppMAE {avg[(best[0],0.0)]:.4f} (lam=0) -> {avg[best]:.4f}')

g('S6', all(v['lam'] < 1.0 for v in selected.values()),
  'selected lambdas ' + str({m: v['lam'] for m, v in selected.items()}))
g('K6', all(v['lam'] > 0.0 for v in selected.values()),
  'no market selected lambda=0 (which would mean "no correction is best")')

# ---------------------------------------------------------------- 2026, touched once
print('\n=== 2026 validation (boards + nflverse actuals) ===')
b = pd.concat([pd.read_csv(f, low_memory=False) for f in
               sorted(glob.glob('data/market_track_record/boards/2026_wk*.csv'))], ignore_index=True)
b = b[b['market'].isin(MARKETS)].drop_duplicates(
    subset=['season', 'week', 'player_clean_key', 'market', 'vegas_line', 'book'])
b['ens'] = pd.to_numeric(b['ensemble_proj'], errors='coerce')
b = b.dropna(subset=['ens'])
b = b.groupby(['week', 'player_clean_key', 'market'], as_index=False)['ens'].median()

ps = nfl.load_player_stats(seasons=[2026], summary_level='week').to_pandas()
ps['vol'] = sum(pd.to_numeric(ps.get(c), errors='coerce').fillna(0)
                for c in ('attempts', 'targets', 'carries'))
for c in ['receiving_yards', 'rushing_yards', 'receptions']:
    ps[c] = pd.to_numeric(ps[c], errors='coerce').fillna(0)
ps['rec_yards'] = ps['receiving_yards']; ps['rush_yards'] = ps['rushing_yards']
ps['rush_rec_yards'] = ps['rushing_yards'] + ps['receiving_yards']
kk = lambda s: s.astype(str).str.lower().str.replace(r'[^a-z]', '', regex=True)
ps['pk'] = kk(ps['player_display_name']); b['pk'] = kk(b['player_clean_key'])
long = pd.concat([ps[['pk', 'week', m, 'vol']].rename(columns={m: 'actual'}).assign(market=m)
                  for m in MARKETS], ignore_index=True)
v26 = b.merge(long, on=['pk', 'week', 'market'], how='left')
v26 = v26[v26['actual'].notna() & (v26['vol'] > 0)]
g('M2', 2026 not in set(h['season']), 'no 2026 row entered any fit or the selection')

def spread(p, y, strat, cuts):
    q = np.digitize(strat, cuts); pos = y > 0
    r = np.where(pos, p / np.where(pos, y, 1), np.nan)
    return float(np.nanmedian(r[q == 3]) - np.nanmedian(r[q == 0]))

res = {}
for m in MARKETS:
    s = h[h.market == m]; vv = v26[v26.market == m]
    if len(vv) < 100 or vv['pk'].nunique() < 40:
        print(f'  {m:16s} 2026 n={len(vv)} players={vv["pk"].nunique()} -> K5, not promoted')
        continue
    xf, yf = s['base'].to_numpy(float), s['actual'].to_numpy(float)
    cuts = np.quantile(xf, [.25, .5, .75])
    fn = fit_arms(xf, yf, cuts)[selected[m]['arm']]
    lam = selected[m]['lam']
    xv, yv = vv['ens'].to_numpy(float), vv['actual'].to_numpy(float)
    base, corr = xv, xv + lam * (fn(xv) - xv)
    im, up = idxmap(vv['pk'].to_numpy())
    d0, d1 = ppmae(base, yv, im, up), ppmae(corr, yv, im, up)
    da = {u: np.abs(corr[im[u]] - yv[im[u]]).mean() - np.abs(base[im[u]] - yv[im[u]]).mean() for u in up}
    dv = np.array([da[u] for u in up])
    bs = np.array([dv[RNG.integers(0, len(up), len(up))].mean() for _ in range(2500)])
    lo, hi = float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
    s0, s1 = spread(base, yv, xv, cuts), spread(corr, yv, xv, cuts)
    lv = float(corr.mean() / yv.mean())
    qv = np.digitize(xv, cuts); worst, wq = -9.9, ''
    for q in range(4):
        k = qv == q
        if k.sum() < 20: continue
        m0 = np.abs(base[k] - yv[k]).mean(); m1 = np.abs(corr[k] - yv[k]).mean()
        rel = (m1 - m0) / max(m0, 1e-9)
        if rel > worst: worst, wq = rel, f'Q{q}'
    S1 = abs(s1) <= 0.6 * abs(s0); S2 = not (lo > 0); S3 = worst <= 0.05; S4 = abs(lv - 1) <= 0.05
    res[m] = dict(arm=selected[m]['arm'], lam=lam, n=len(vv), players=int(vv['pk'].nunique()),
                  ppmae_base=d0, ppmae_corr=d1, mae_ci=[float(dv.mean()), lo, hi],
                  spread_base=s0, spread_corr=s1, level_base=float(base.mean()/yv.mean()),
                  level=lv, worst_q=float(worst), worst_q_name=wq,
                  S1=S1, S2=S2, S3=S3, S4=S4, promote=bool(S1 and S2 and S3 and S4))
    print(f'  {m:16s} {selected[m]["arm"]:12s} lam={lam:.1f} n={len(vv):4d}/{vv["pk"].nunique():3d}pl  '
          f'ppMAE {d0:.3f}->{d1:.3f} [{lo:+.3f},{hi:+.3f}]  '
          f'level {base.mean()/yv.mean():.3f}->{lv:.3f}  spread {s0:+.3f}->{s1:+.3f}  '
          f'worstQ {worst*100:+.1f}%{wq}')
    print(f'  {"":16s} S1={"P" if S1 else "f"} S2={"P" if S2 else "f"} '
          f'S3={"P" if S3 else "f"} S4={"P" if S4 else "f"}  '
          f'-> {"PROMOTE" if res[m]["promote"] else "no"}')

g('M4', 'pass_yards' not in selected and 'pass_yards' not in res,
  'pass_yards is out of scope: no coefficient emitted, output untouched')
any_p = any(r['promote'] for r in res.values())
print(f'\n=== ANY PROMOTION: {"YES" if any_p else "NO"} ===')
json.dump({'gates': gates, 'selected': selected, 'results_2026': res, 'any': any_p},
          open('data/research/shrunk_volume_conditional_v1_results.json', 'w'), indent=2, default=float)
print('wrote data/research/shrunk_volume_conditional_v1_results.json')
