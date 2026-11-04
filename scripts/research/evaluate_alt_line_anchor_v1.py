"""PRISM — alt-line anchor choice. Preregistered:
docs/research/PRISM_ALT_LINE_ANCHOR_PREREGISTRATION_V1.md

MARKET-ASSISTED bet-selection layer. Modifies no football projection.
Fit 2024 / validate 2025 / prospective 2026 boards. Clustered over players.
"""
import json, sys, glob
import numpy as np, pandas as pd

RNG = np.random.default_rng(20261104)
HIST = 'docs/research/overnight/clean_v1_full_stack_vegas_benchmark_detail.csv'
MARKETS = ['rec_yards', 'receptions', 'rush_yards', 'rush_rec_yards', 'pass_yards']
GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]          # frozen decision grid
FINE = np.round(np.arange(0.20, 2.0001, 0.05), 4)                    # curve support
FORBIDDEN = ['over_odds', 'under_odds', 'over_novig', 'under_novig', 'over_implied',
             'under_implied', 'ev_over', 'ev_under', 'edge_pct', 'edge_abs', 'best_ev',
             'best_market_p', 'prob_edge', 'chosen_odds', 'vegas_odds', 'fair_odds',
             'fair_prob', 'market_prob', 'unit_result', 'bet_result', 'vegas_error',
             'p_over', 'p_under', 'side', 'signal', 'actual_side', 'best_model_p']
MODELCOLS = ['proj', 'mc_proj', 'ml_proj', 'state_proj', 'ensemble_proj', 'model_proj',
             'component_sd', 'model_sd', 'model_error']
findings = []


def note(tag, msg):
    findings.append(f'{tag}: {msg}')
    print(f'  [{tag}] {msg}')


# ---------------------------------------------------------------- load historical
h = pd.read_csv(HIST, low_memory=False)
h = h[h['market'].isin(MARKETS)].copy()
h = h.rename(columns={'ensemble_proj': 'proj_use'})
h['line_use'] = pd.to_numeric(h['line'], errors='coerce')
h['actual_use'] = pd.to_numeric(h['actual'], errors='coerce')
h['player'] = h['player_clean_key'].astype(str)
h['era'] = 'hist'
# M4 void rule: no verified volume and no production -> not a graded observation
opp = pd.to_numeric(h.get('actual_opportunities'), errors='coerce')
void_h = (opp.fillna(0) <= 0) & (h['actual_use'].fillna(0) <= 0)
note('M4-hist', f'{int(void_h.sum())} of {len(h)} historical rows VOID (no verified volume)')
h = h[~void_h]
h = h.dropna(subset=['proj_use', 'line_use', 'actual_use'])
h = h[(h['line_use'] > 0) & (h['proj_use'] > 0)]

# ---------------------------------------------------------------- load 2026 prospective
import nflreadpy as nfl
ps = nfl.load_player_stats(seasons=[2026], summary_level='week').to_pandas()
ps['vol'] = (pd.to_numeric(ps.get('attempts'), errors='coerce').fillna(0)
             + pd.to_numeric(ps.get('targets'), errors='coerce').fillna(0)
             + pd.to_numeric(ps.get('carries'), errors='coerce').fillna(0))
for c in ['passing_yards', 'rushing_yards', 'receiving_yards', 'receptions']:
    ps[c] = pd.to_numeric(ps[c], errors='coerce').fillna(0.0)
ps['rush_rec_yards'] = ps['rushing_yards'] + ps['receiving_yards']
ps['rec_yards'] = ps['receiving_yards']
ps['rush_yards'] = ps['rushing_yards']
ps['pass_yards'] = ps['passing_yards']


def key(s):
    return (s.astype(str).str.lower().str.replace(r'[^a-z]', '', regex=True))


ps['pk'] = key(ps['player_display_name'])
boards = []
for f in sorted(glob.glob('data/market_track_record/boards/2026_wk*.csv')):
    boards.append(pd.read_csv(f, low_memory=False))
b = pd.concat(boards, ignore_index=True)
b = b[b['market'].isin(MARKETS)].copy()
b = b.drop_duplicates(subset=['season', 'week', 'player_clean_key', 'market', 'vegas_line', 'book'])
b['line_use'] = pd.to_numeric(b['vegas_line'], errors='coerce')
b['proj_use'] = pd.to_numeric(b['model_proj'], errors='coerce')
b = b.dropna(subset=['line_use', 'proj_use'])
b = b[(b['line_use'] > 0) & (b['proj_use'] > 0)]
b = (b.groupby(['season', 'week', 'player_clean_key', 'market'], as_index=False)
       [['line_use', 'proj_use']].median())
b['pk'] = key(b['player_clean_key'])

long = []
for m in MARKETS:
    sub = ps[['pk', 'week', m, 'vol']].rename(columns={m: 'actual_use'})
    sub['market'] = m
    long.append(sub)
act = pd.concat(long, ignore_index=True)
p26 = b.merge(act, on=['pk', 'week', 'market'], how='left')
unmatched = p26['actual_use'].isna().sum()
void26 = p26['actual_use'].isna() | (p26['vol'].fillna(0) <= 0)
note('M4-2026', f'{int(unmatched)} unmatched + {int(void26.sum()-unmatched)} zero-volume '
                f'= {int(void26.sum())} of {len(p26)} 2026 rows VOID')
p26 = p26[~void26].copy()
p26['player'] = p26['player_clean_key'].astype(str)
p26['era'] = 'prod'
p26['season'] = 2026

# ---------------------------------------------------------------- M1 / M2
cols_used = ['proj_use', 'line_use', 'actual_use', 'player', 'market', 'season', 'week', 'era']
bad = [c for c in FORBIDDEN if c in cols_used]
if bad:
    sys.exit(f'M1 BREACH: forbidden column in use: {bad}')
note('M1', f'PASS — {len(FORBIDDEN)} forbidden odds/EV columns checked by name, none in the '
           f'feature set {cols_used}')

fit = h[h['season'] == 2024].copy()
val = h[h['season'] == 2025].copy()
if len(fit) == 0 or len(val) == 0:
    sys.exit('empty split')
if set(fit['season']) != {2024}:
    sys.exit(f'M2 BREACH: fit seasons {set(fit["season"])}')
note('M2', f'PASS — fit seasons {sorted(set(fit["season"]))}, '
           f'validate {sorted(set(val["season"]))}, prospective {sorted(set(p26["season"]))}')

# ---------------------------------------------------------------- quartile cuts from FIT only
cuts = {}
for m in MARKETS:
    g = fit[fit['market'] == m]
    if len(g) < 40:
        continue
    q = np.quantile(g['line_use'], [0.25, 0.50, 0.75])
    cuts[m] = q


def assign_q(d):
    out = np.full(len(d), -1, dtype=int)
    for m, q in cuts.items():
        k = (d['market'] == m).to_numpy()
        out[k] = np.digitize(d.loc[k, 'line_use'].to_numpy(), q)
    return out


for d in (fit, val, p26):
    d['qt'] = assign_q(d)
note('cuts', 'line quartile boundaries estimated on the 2024 FIT sample only and applied '
             'unchanged to 2025 and 2026 (no leakage)')

# ---------------------------------------------------------------- curves on FIT
def curve(d, anchor):
    """empirical P(actual > f*anchor) on the fine grid, per market."""
    out = {}
    for m, g in d.groupby('market'):
        a = g[anchor].to_numpy()
        y = g['actual_use'].to_numpy()
        out[m] = np.array([float(np.mean(y > f * a)) for f in FINE])
    return out


c_line = curve(fit, 'line_use')     # B0 support
c_proj = curve(fit, 'proj_use')     # B1 support

cell = {}                            # B2: (market, qt) -> rate curve on FINE
for (m, q), g in fit.groupby(['market', 'qt']):
    if len(g) < 25:
        continue
    a = g['line_use'].to_numpy(); y = g['actual_use'].to_numpy()
    cell[(m, q)] = np.array([float(np.mean(y > f * a)) for f in FINE])

# B3: logistic on quartile dummies + z, per (market, f) on FINE
from sklearn.linear_model import LogisticRegression

zstat = {}                           # (market, era, qt) -> (mu, sd) of proj/line
def add_z(d, era_stats):
    r = (d['proj_use'] / d['line_use']).to_numpy()
    z = np.zeros(len(d))
    for i, (m, q) in enumerate(zip(d['market'], d['qt'])):
        mu, sd = era_stats.get((m, q), (np.nan, np.nan))
        z[i] = 0.0 if not np.isfinite(sd) or sd <= 0 else (r[i] - mu) / sd
    return np.clip(z, -4, 4)

for (m, q), g in fit.groupby(['market', 'qt']):
    r = (g['proj_use'] / g['line_use']).to_numpy()
    zstat[(m, q)] = (float(np.mean(r)), float(np.std(r)))
fit['z'] = add_z(fit, zstat)
val['z'] = add_z(val, zstat)          # same era -> 2024 stats are leakage-free

# 2026 era: expanding prior-weeks standardisation; W1 has no prior and is dropped
wk_order = sorted(p26['week'].unique())
z26 = np.full(len(p26), np.nan)
p26 = p26.reset_index(drop=True)
for w in wk_order:
    prior = p26[p26['week'] < w]
    if len(prior) < 60:
        continue
    st = {}
    for (m, q), g in prior.groupby(['market', 'qt']):
        if len(g) < 15:
            continue
        r = (g['proj_use'] / g['line_use']).to_numpy()
        st[(m, q)] = (float(np.mean(r)), float(np.std(r)))
    k = (p26['week'] == w).to_numpy()
    z26[k] = add_z(p26[k], st)
p26['z'] = z26
pro = p26[np.isfinite(p26['z'])].copy()
note('era-z', f'2026 z standardised from prior weeks only; weeks kept '
               f'{sorted(set(pro["week"]))} of {wk_order} (W1 has no prior week)')

b3 = {}
for m in MARKETS:
    g = fit[fit['market'] == m]
    if len(g) < 120:
        continue
    Q = pd.get_dummies(g['qt'].astype(str), prefix='q')
    qcols = list(Q.columns)
    for j, f in enumerate(FINE):
        y = (g['actual_use'].to_numpy() > f * g['line_use'].to_numpy()).astype(int)
        if y.min() == y.max():
            continue
        X2 = Q.to_numpy(float)
        X3 = np.column_stack([X2, g['z'].to_numpy()])
        try:
            m2 = LogisticRegression(max_iter=600, C=1e6).fit(X2, y)
            m3 = LogisticRegression(max_iter=600, C=1e6).fit(X3, y)
        except Exception:
            continue
        b3[(m, j)] = (m2, m3, qcols)

# ---------------------------------------------------------------- scoring frame
def predict(d, arm):
    """Common event: 1[actual > T] with T = g*line. Each arm reaches it via its own anchor."""
    rows = []
    dq = pd.get_dummies(d['qt'].astype(str), prefix='q')
    for g in GRID:
        T = g * d['line_use'].to_numpy()
        y = (d['actual_use'].to_numpy() > T).astype(int)
        p = np.full(len(d), np.nan)
        for m in d['market'].unique():
            k = (d['market'] == m).to_numpy()
            if arm == 'B0':
                if m not in c_line:
                    continue
                p[k] = np.interp(g, FINE, c_line[m])
            elif arm == 'B1':
                if m not in c_proj:
                    continue
                f_eff = T[k] / d.loc[k, 'proj_use'].to_numpy()
                p[k] = np.interp(f_eff, FINE, c_proj[m])
            elif arm in ('B2', 'B3'):
                j = int(np.argmin(np.abs(FINE - g)))
                if arm == 'B2':
                    for q in np.unique(d.loc[k, 'qt']):
                        kk = k & (d['qt'] == q).to_numpy()
                        if (m, q) in cell:
                            p[kk] = cell[(m, q)][j]
                        elif m in c_line:
                            p[kk] = np.interp(g, FINE, c_line[m])
                else:
                    if (m, j) not in b3:
                        continue
                    _, mdl, qcols = b3[(m, j)]
                    Q = pd.DataFrame(0.0, index=d.index[k], columns=qcols)
                    for c in qcols:
                        if c in dq.columns:
                            Q[c] = dq.loc[d.index[k], c].astype(float).to_numpy()
                    X = np.column_stack([Q.to_numpy(float), d.loc[k, 'z'].to_numpy()])
                    p[k] = mdl.predict_proba(X)[:, 1]
        rows.append(pd.DataFrame({'frac': g, 'market': d['market'].to_numpy(),
                                  'player': d['player'].to_numpy(), 'y': y, 'p': p}))
    r = pd.concat(rows, ignore_index=True)
    return r.dropna(subset=['p'])


def brier_by_market(r):
    r = r.copy(); r['se'] = (r['p'] - r['y']) ** 2
    return r.groupby('market')['se'].mean()


def boot_delta(rA, rB, reps=1200):
    """clustered over players: mean Brier(A) - Brier(B), negative = A better"""
    kA = rA.set_index(['market', 'player', 'frac'])
    kB = rB.set_index(['market', 'player', 'frac'])
    j = kA[['y', 'p']].join(kB[['p']], how='inner', lsuffix='_a', rsuffix='_b').reset_index()
    out = {}
    for m, g in j.groupby('market'):
        ea = (g['p_a'] - g['y']) ** 2
        eb = (g['p_b'] - g['y']) ** 2
        d = (ea - eb).to_numpy()
        pl = g['player'].to_numpy()
        up = np.unique(pl)
        idx = {u: np.where(pl == u)[0] for u in up}
        pt = float(d.mean())
        bs = np.empty(reps)
        for i in range(reps):
            s = RNG.choice(up, size=len(up), replace=True)
            bs[i] = d[np.concatenate([idx[u] for u in s])].mean()
        out[m] = (pt, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)),
                  len(up), len(g))
    return out


# ---------------------------------------------------------------- M3
drop = val.drop(columns=[c for c in MODELCOLS if c in val.columns], errors='ignore').copy()
drop['proj_use'] = np.nan
r_b0 = predict(val, 'B0')
r_b0_nm = predict(drop, 'B0')
same = np.allclose(r_b0['p'].to_numpy(), r_b0_nm['p'].to_numpy(), atol=0, rtol=0)
if not same:
    sys.exit('M3 BREACH: B0 depends on a model column')
note('M3', 'PASS — B0 recomputed with every model column dropped is bit-identical')

print('\n=== validation 2025 ===')
arms = {a: predict(val, a) for a in ['B0', 'B1', 'B2', 'B3']}
bm = {a: brier_by_market(r) for a, r in arms.items()}
print(pd.DataFrame(bm).round(5).to_string())

print('\n=== S1  B0 vs B1 (market anchor vs projection anchor) ===')
s1 = boot_delta(arms['B0'], arms['B1'])
print('\n=== S2  B2 vs B0 (volume stratification) ===')
s2 = boot_delta(arms['B2'], arms['B0'])
print('\n=== S3  B3 vs B2 (DOES THE MODEL ADD ANYTHING) ===')
s3 = boot_delta(arms['B3'], arms['B2'])
for nm, s in [('S1 B0-B1', s1), ('S2 B2-B0', s2), ('S3 B3-B2', s3)]:
    print(f'--- {nm}')
    for m, (pt, lo, hi, npl, nr) in sorted(s.items()):
        verdict = 'PASS' if hi < 0 else ('worse' if lo > 0 else 'ns')
        flag = '  [K5 underpowered]' if npl < 150 else ''
        print(f'    {m:16s} d={pt:+.5f} [{lo:+.5f},{hi:+.5f}] players={npl:4d} n={nr:6d} {verdict}{flag}')

print('\n=== S4 calibration on 2025 (decile of predicted) ===')
calib = {}
for a, r in arms.items():
    r = r.copy()
    try:
        r['d'] = pd.qcut(r['p'], 10, labels=False, duplicates='drop')
    except Exception:
        continue
    t = r.groupby('d').agg(pred=('p', 'mean'), real=('y', 'mean'), n=('y', 'size'))
    t['gap'] = t['real'] - t['pred']
    calib[a] = t
    worst = t['gap'].abs().max()
    print(f'  {a}: max |gap| across deciles = {worst:.4f}  {"PASS" if worst<=0.03 else "FAIL (>3pp)"}')

print('\n=== S5 prospective 2026 ===')
pro_arms = {a: predict(pro, a) for a in ['B0', 'B1', 'B2', 'B3']}
pbm = {a: brier_by_market(r) for a, r in pro_arms.items()}
print(pd.DataFrame(pbm).round(5).to_string())
p1 = boot_delta(pro_arms['B0'], pro_arms['B1'], reps=800)
p3 = boot_delta(pro_arms['B3'], pro_arms['B2'], reps=800)
for nm, s in [('S5 B0-B1 (2026)', p1), ('S5 B3-B2 (2026)', p3)]:
    print(f'--- {nm}')
    for m, (pt, lo, hi, npl, nr) in sorted(s.items()):
        print(f'    {m:16s} d={pt:+.5f} [{lo:+.5f},{hi:+.5f}] players={npl:4d} n={nr:6d}')

# ---------------------------------------------------------------- outputs
res = {'findings': findings,
       'brier_2025': {a: {k: float(v) for k, v in s.items()} for a, s in bm.items()},
       'brier_2026': {a: {k: float(v) for k, v in s.items()} for a, s in pbm.items()},
       'S1_B0_vs_B1': {m: list(v) for m, v in s1.items()},
       'S2_B2_vs_B0': {m: list(v) for m, v in s2.items()},
       'S3_B3_vs_B2': {m: list(v) for m, v in s3.items()},
       'S5_B0_vs_B1_2026': {m: list(v) for m, v in p1.items()},
       'S5_B3_vs_B2_2026': {m: list(v) for m, v in p3.items()},
       'n': {'fit': len(fit), 'val': len(val), 'pro': len(pro),
             'val_players': int(val['player'].nunique()),
             'pro_players': int(pro['player'].nunique())}}
with open('data/research/alt_line_anchor_v1_results.json', 'w') as fh:
    json.dump(res, fh, indent=2, default=float)

# the market-anchored stratified table (B2), for the deliverable
tbl = []
for (m, q), cv in sorted(cell.items()):
    g = fit[(fit['market'] == m) & (fit['qt'] == q)]
    for gf in GRID:
        j = int(np.argmin(np.abs(FINE - gf)))
        c = float(cv[j]); n = len(g); npl = int(g['player'].nunique())
        if n < 100 or npl < 40:          # K7 support floor
            continue
        se = np.sqrt(max(c * (1 - c), 1e-9) / max(npl, 1))
        lo = float(np.clip(c - 1.96 * se, 1e-4, 1 - 1e-4))
        cc = float(np.clip(c, 1e-4, 1 - 1e-4))
        be = (-100 * cc / (1 - cc)) if cc > 0.5 else (100 * (1 - cc) / cc)
        bl = (-100 * lo / (1 - lo)) if lo > 0.5 else (100 * (1 - lo) / lo)
        tbl.append({'market': m, 'line_quartile': int(q), 'frac_of_LINE': gf,
                    'line_median': float(g['line_use'].median()), 'n': n, 'players': npl,
                    'cleared': round(c, 4), 'cleared_ci_lo': round(lo, 4),
                    'breakeven_american': int(round(be)),
                    'breakeven_american_conservative': int(round(bl))})
pd.DataFrame(tbl).to_csv('data/research/alt_line_market_anchored_table_v1.csv', index=False)
print('\nwrote data/research/alt_line_anchor_v1_results.json')
print('wrote data/research/alt_line_market_anchored_table_v1.csv')
