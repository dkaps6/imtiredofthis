"""PRISM — rush_rec_yards ensemble calibration. Preregistered:
docs/research/PRISM_RUSH_REC_ENSEMBLE_PREREGISTRATION_V1.md

Fit 2024 only / blind holdout 2025. Uses scripts.modeling.ensemble_v2
.fit_market_weights UNMODIFIED, the same promoted method as the five existing
rows. Clustered over players. Writes nothing unless every gate passes.
"""
from __future__ import annotations
import json, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, '.')
from scripts.modeling.ensemble_v2 import fit_market_weights, apply_ensemble, MIN_CALIBRATION_ROWS

RNG = np.random.default_rng(20261104)
HIST = 'docs/research/overnight/clean_v1_full_stack_vegas_benchmark_detail.csv'
MARKET = 'rush_rec_yards'
COMPONENTS = ['mc_proj', 'ml_proj', 'state_proj']
FORBIDDEN = ['line', 'over_odds', 'under_odds', 'over_novig', 'under_novig', 'p_over', 'p_under',
             'ev_over', 'ev_under', 'edge_pct', 'edge_abs', 'vegas_error', 'bet_result',
             'unit_result', 'side', 'signal', 'over_implied', 'under_implied', 'best_ev']
gates = {}


def g(tag, ok, msg):
    gates[tag] = {'pass': bool(ok), 'msg': msg}
    print(f'  [{tag}] {"PASS" if ok else "FAIL"} — {msg}')
    return ok


# ------------------------------------------------------------------ data
h = pd.read_csv(HIST, low_memory=False)
d = h[h['market'] == MARKET].copy()
for c in COMPONENTS + ['actual']:
    d[c] = pd.to_numeric(d[c], errors='coerce')
opp = pd.to_numeric(d.get('actual_opportunities'), errors='coerce')
void = (opp.fillna(0) <= 0) & (d['actual'].fillna(0) <= 0)
print(f'  voided {int(void.sum())} rows with no verified volume')
d = d[~void].dropna(subset=COMPONENTS + ['actual'])
d['player'] = d['player_clean_key'].astype(str)

fit = d[d['season'] == 2024].copy()
val = d[d['season'] == 2025].copy()
print(f'  fit {len(fit)} rows / {fit["player"].nunique()} players  |  '
      f'blind {len(val)} rows / {val["player"].nunique()} players')

# ------------------------------------------------------------------ M1 / M2 / K6
feat = COMPONENTS + ['actual', 'market']
g('M1', not [c for c in FORBIDDEN if c in feat],
  f'target=actual, features={COMPONENTS}; {len(FORBIDDEN)} forbidden sportsbook columns absent')
g('M2', set(fit['season']) == {2024}, f'fit seasons {sorted(set(fit["season"]))}, blind {sorted(set(val["season"]))}')
g('K6', len(fit) >= MIN_CALIBRATION_ROWS, f'{len(fit)} fit rows vs MIN_CALIBRATION_ROWS={MIN_CALIBRATION_ROWS}')

# ------------------------------------------------------------------ A1: the promoted method, unmodified
w = fit_market_weights(fit[['market', 'actual'] + COMPONENTS])
row = w[w['market'] == MARKET]
if row.empty:
    sys.exit('fit_market_weights returned no row for the market')
r = row.iloc[0]
W = np.array([r['mc_weight'], r['ml_weight'], r['state_weight']], float)
print(f'\n  A1 weights  mc={W[0]:.6f} ml={W[1]:.6f} state={W[2]:.6f}  (sum {W.sum():.6f})')
print(f'  calibration_rows={int(r["calibration_rows"])} mae={r["calibration_mae"]:.6f} rmse={r["calibration_rmse"]:.6f}')

X = val[COMPONENTS].to_numpy(float)
y = val['actual'].to_numpy(float)
pred = {
    'A0_mc_only': X[:, 0],
    'A1_blend': X @ W,
    'A2_ml_only_REF': X[:, 1],
}
nn = LinearRegression(fit_intercept=False, positive=True).fit(fit[COMPONENTS].to_numpy(float),
                                                              fit['actual'].to_numpy(float))
pred['A3_unnormalised_REF'] = X @ nn.coef_
print(f'  A3 raw coefs mc={nn.coef_[0]:.4f} ml={nn.coef_[1]:.4f} state={nn.coef_[2]:.4f} '
      f'(sum {nn.coef_.sum():.4f}) — normalisation discards this scale')

# ------------------------------------------------------------------ S1 per-player, clustered
pl = val['player'].to_numpy()
up = np.unique(pl)
idx = {u: np.where(pl == u)[0] for u in up}


def per_player_mae(p):
    return float(np.mean([np.abs(p[idx[u]] - y[idx[u]]).mean() for u in up]))


print('\n=== blind 2025 ===')
print(f"{'arm':22s} {'per-player MAE':>15s} {'pooled MAE':>11s} {'mean/actual':>12s}")
for k, p in pred.items():
    print(f'{k:22s} {per_player_mae(p):15.4f} {np.abs(p-y).mean():11.4f} {p.mean()/y.mean():12.4f}')


def boot(pa, pb, reps=3000):
    """mean per-player MAE(a) - (b); negative = a better. clustered over players."""
    da = {u: np.abs(pa[idx[u]] - y[idx[u]]).mean() for u in up}
    db = {u: np.abs(pb[idx[u]] - y[idx[u]]).mean() for u in up}
    dv = np.array([da[u] - db[u] for u in up])
    bs = np.array([dv[RNG.integers(0, len(up), len(up))].mean() for _ in range(reps)])
    return float(dv.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))

print('\n=== S1  A1 vs A0 (per-player MAE, clustered over players) ===')
pt, lo, hi = boot(pred['A1_blend'], pred['A0_mc_only'])
g('S1', hi < 0, f'delta={pt:+.4f} [{lo:+.4f},{hi:+.4f}] over {len(up)} players')
for nm in ['A2_ml_only_REF', 'A3_unnormalised_REF']:
    p2, l2, h2 = boot(pred[nm], pred['A1_blend'])
    print(f'  reference  {nm} vs A1: {p2:+.4f} [{l2:+.4f},{h2:+.4f}]'
          f'{"  (better than the candidate)" if h2 < 0 else ""}')

# ------------------------------------------------------------------ S2 level
lv0 = abs(pred['A0_mc_only'].mean() / y.mean() - 1)
lv1 = abs(pred['A1_blend'].mean() / y.mean() - 1)
g('S2', lv1 < lv0, f'|level-1| {lv0:.4f} -> {lv1:.4f} '
                   f'(ratio {pred["A0_mc_only"].mean()/y.mean():.4f} -> {pred["A1_blend"].mean()/y.mean():.4f}; '
                   f'A2 ml-only {pred["A2_ml_only_REF"].mean()/y.mean():.4f})')

# ------------------------------------------------------------------ S3 quartile harm
q = np.quantile(pd.to_numeric(fit['line'], errors='coerce').dropna(), [.25, .5, .75])
vq = np.digitize(pd.to_numeric(val['line'], errors='coerce').fillna(-1).to_numpy(), q)
print('\n=== S3 per line-quartile MAE (quartiles from the 2024 fit sample) ===')
worst = 0.0; worst_cell = ''
for qq in sorted(set(vq)):
    k = vq == qq
    if k.sum() < 20:
        continue
    m0 = np.abs(pred['A0_mc_only'][k] - y[k]).mean()
    m1 = np.abs(pred['A1_blend'][k] - y[k]).mean()
    rel = (m1 - m0) / max(m0, 1e-9)
    print(f'  Q{qq} n={int(k.sum()):4d} line~{pd.to_numeric(val["line"],errors="coerce")[k].median():6.1f}  '
          f'A0={m0:7.3f} -> A1={m1:7.3f}  {rel*100:+6.1f}%')
    if rel > worst:
        worst, worst_cell = rel, f'Q{qq}'
g('S3', worst <= 0.05, f'worst quartile degradation {worst*100:+.1f}% at {worst_cell or "none"} vs a +5% bar')

# ------------------------------------------------------------------ M3 / M4 consumer check
BASELINE_BYTES = open('data/model_ensemble_weights.csv', 'rb').read()
BASELINE_BYTES = b'\n'.join(l for l in BASELINE_BYTES.split(b'\n')
                            if l and not l.startswith(MARKET.encode() + b','))
cur = pd.read_csv('data/model_ensemble_weights.csv')
cur = cur[cur['market'] != MARKET].reset_index(drop=True)
new = pd.concat([cur, pd.DataFrame([{
    'market': MARKET, 'mc_weight': W[0], 'ml_weight': W[1], 'state_weight': W[2],
    'calibration_rows': int(r['calibration_rows']), 'calibration_mae': float(r['calibration_mae']),
    'calibration_rmse': float(r['calibration_rmse']), 'method': 'nonnegative_oos_linear_blend_v2',
    'fit_scope': 'fit_2024_only_blind_holdout_2025', 'validated_season': '2025',
    'promotion_lineage': 'RUSH_REC_ENSEMBLE_V23_FIT2024_BLIND2025'}])], ignore_index=True)

probe = d[['market'] + COMPONENTS].head(300).copy()
others = h[h['market'].isin(['pass_yards', 'rec_yards', 'receptions', 'rush_yards'])][['market'] + COMPONENTS].head(400).copy()
probe_all = pd.concat([probe, others], ignore_index=True)
before = apply_ensemble(probe_all, weights=cur)
after = apply_ensemble(probe_all, weights=new)
km = after['market'] == MARKET
g('M3', (after.loc[km, 'ensemble_status'] == 'calibrated').all() and
        (before.loc[km, 'ensemble_status'] == 'uncalibrated_mc_only').all(),
  f'{MARKET} goes uncalibrated_mc_only -> calibrated with no code change')
ko = ~km
same = np.allclose(before.loc[ko, 'ensemble_proj'].to_numpy(float),
                   after.loc[ko, 'ensemble_proj'].to_numpy(float), rtol=0, atol=0)
head_bytes = open('data/model_ensemble_weights.csv', 'rb').read()
pre_rows = [l for l in head_bytes.decode().splitlines() if l and not l.startswith(MARKET + ',')]
baseline = [l for l in BASELINE_BYTES.decode().splitlines() if l]
rows_same = pre_rows == baseline
g('M4', same and rows_same,
  f'{int(ko.sum())} rows of the other 4 markets bit-identical; '
  f'{len(baseline)} pre-existing weight lines byte-for-byte unchanged')

allp = all(v['pass'] for v in gates.values())
print(f'\n=== ALL GATES: {"PASS" if allp else "FAIL"} ===')
out = {'gates': gates, 'weights': {'mc': float(W[0]), 'ml': float(W[1]), 'state': float(W[2])},
       'calibration_rows': int(r['calibration_rows']),
       'calibration_mae': float(r['calibration_mae']), 'calibration_rmse': float(r['calibration_rmse']),
       'blind': {k: {'per_player_mae': per_player_mae(v), 'pooled_mae': float(np.abs(v-y).mean()),
                     'level': float(v.mean()/y.mean())} for k, v in pred.items()},
       'n': {'fit': len(fit), 'blind': len(val), 'blind_players': int(len(up))},
       'a3_raw_coefs': [float(x) for x in nn.coef_], 'all_gates_pass': allp}
with open('data/research/rush_rec_ensemble_v1_results.json', 'w') as fh:
    json.dump(out, fh, indent=2)
print('wrote data/research/rush_rec_ensemble_v1_results.json')
if allp:
    # Text-append, never a pandas round-trip: M4 requires every pre-existing byte
    # of this production artifact to survive unchanged, and a to_csv round-trip
    # silently reprecisions other rows (it turned receptions 1.351260 into 1.35126).
    line = (f"{MARKET},{W[0]!r},{W[1]!r},{W[2]!r},{int(r['calibration_rows'])},"
            f"{float(r['calibration_mae'])!r},{float(r['calibration_rmse'])!r},"
            f"nonnegative_oos_linear_blend_v2,fit_2024_only_blind_holdout_2025,2025,"
            f"RUSH_REC_ENSEMBLE_V23_FIT2024_BLIND2025\n")
    path = 'data/model_ensemble_weights.csv'
    body = open(path).read()
    if MARKET + ',' in body:
        print(f'{MARKET} row already present -> production artifact NOT modified')
    else:
        if not body.endswith('\n'):
            body += '\n'
        open(path, 'w').write(body + line)
        print('APPENDED the new row to data/model_ensemble_weights.csv')
else:
    print('gates failed -> production artifact NOT modified')
