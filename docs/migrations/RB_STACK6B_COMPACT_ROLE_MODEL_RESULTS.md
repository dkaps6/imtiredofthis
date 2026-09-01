# RB STACK6B — Compact Role Model Results

## Authoritative execution

### Frozen compact-role model

- Branch: `research-rb-stack6b-compact-role-model`
- Canonical run: `33570434397`
- Tested SHA: `9e3e8fde006a93e6ea6c8357254432c229303b2b`
- Artifact: `9824722713`
- Artifact SHA256: `edb52d2687a68cbe3bec5171bb745678abc3a2191e50d129579418fd3bc4f789`
- Frozen plan: `docs/migrations/RB_STACK6B_COMPACT_ROLE_MODEL_PLAN.md`
- Broad STACK6 parent artifact run: `33549529203`
- Market benchmark run, downstream only: `33499129109`

### No-fit directional postmortem

- Run: `33570615124`
- Job: `100063540799`
- Tested SHA: `f6cd9e69c162c6cb49d183c2c3f1603b48dec902`
- Artifact: `9824778090`
- Artifact SHA256: `7df984e5ce329258e9102cee9c3a249fb5194d95951a1e50f6de754576ca3378`
- Frozen plan: `docs/migrations/RB_STACK6B_DIRECTIONAL_POSTMORTEM_PLAN.md`
- Model fit: `0`
- Sportsbook used: `0`
- Threshold / feature / hyperparameter / weight search: `0`
- Counterfactuals eligible for retention: `0`

## Protocol integrity

The model consumed the frozen broad-STACK6 casebook so that P3 parent values, eligibility, identity mapping, and strictly-prior situational history were unchanged from the experiment that justified STACK6B.

Coverage / contract checks:

- target RB/FB player-games: `1,393`
- W6-18 rows: `1,001`
- eligible W6-18 rows: `555`
- frozen M95F-risk W6-18 rows: `319`
- depth-rank-1 W6-18 rows: `380`
- as-of leakage pass rate: `1.000`
- `COMPACT_ROLE`: exactly `8` features
- `AGG_PLUS_COMPACT`: exactly `22` features (`14` aggregate + `8` compact)
- M95F-risk change: exactly `0`
- depth-rank-1 change: exactly `0`
- sportsbook upstream: `0`

Training preserved broad-STACK6 clipping semantics: the 5th/95th residual bounds were computed from training rows only, intersected with `[-4,+4]` carries, and applied to the model-predicted correction. Training targets themselves were not winsorized.

## Frozen STACK6B result

### Eligible W6-18 — 555 rows

| Arm | Carry MAE | Carry bias | Yard MAE |
|---|---:|---:|---:|
| P3 parent | 2.959194 | +0.025164 | 15.881551 |
| COMPACT_ROLE | 2.984976 | -0.123478 | 16.012606 |
| AGG_PLUS_COMPACT | 2.983587 | -0.215403 | 15.859464 |

### Eligible W13-18 — 272 rows

| Arm | Carry MAE | Yard MAE |
|---|---:|---:|
| P3 parent | 3.111598 | 16.377570 |
| COMPACT_ROLE | 3.096833 | 16.516773 |
| AGG_PLUS_COMPACT | 3.044818 | 16.051484 |

### Retention gates

`COMPACT_ROLE`:

- eligible carry MAE gain: `-0.025782`
- eligible yard MAE gain: `-0.131056`
- W13-18 eligible yard gain: `-0.139204`
- all-RB W6-18 yard regression: `+0.072663`
- absolute carry-bias worsening: `+0.098314`
- protected populations unchanged: pass
- **retention: FAIL**

`AGG_PLUS_COMPACT`:

- eligible carry MAE gain: `-0.024393`
- eligible yard MAE gain: `+0.022087`
- W13-18 eligible yard gain: `+0.326085`
- all-RB W6-18 yard regression: `-0.012246` (improvement)
- absolute carry-bias worsening: `+0.190239`
- protected populations unchanged: pass
- **retention: FAIL**

Frozen disposition:

`STACK6B_NO_RETAINABLE_COMPACT_ROLE_INCREMENT`

No production change. P3 remains the champion point parent.

## Downstream sportsbook benchmark — descriptive only

Exact 899 market-covered games:

- P3: `24.315798`
- COMPACT_ROLE: `24.361436`
- AGG_PLUS_COMPACT: `24.243147`
- Vegas consensus: `23.701891`

Market-covered eligible subset, 269 rows:

- P3: `19.979784`
- COMPACT_ROLE: `20.132305`
- AGG_PLUS_COMPACT: `19.736982`
- Vegas: `19.048327`

The small downstream market improvement for `AGG_PLUS_COMPACT` does not rescue the failed football-first retention gates.

## No-fit directional postmortem

The compact feature family does contain useful information, but the failed model uses it incorrectly in both directions.

### Direction contribution — all 555 eligible rows

`COMPACT_ROLE`:

- contractions: 280 rows (`50.45%`), mean carry absolute-error recovery `+0.0775`, mean yard recovery `+0.3729`
- expansions: 196 rows (`35.32%`), mean carry recovery `-0.1837`, mean yard recovery `-0.9038`
- unchanged: 79 rows

`AGG_PLUS_COMPACT`:

- contractions: 274 rows (`49.37%`), mean carry recovery `+0.1281`, mean yard recovery `+0.8308`
- expansions: 202 rows (`36.40%`), mean carry recovery `-0.2408`, mean yard recovery `-1.0663`
- unchanged: 79 rows

So the contraction signal is positive while expansion is actively harmful. This independently confirms the directionality suggested by the pre-STACK6B failure atlas.

### Diagnostic contraction-only replay — NOT retainable

These rows are a failure-mechanism counterfactual only. No new model was fit and this result cannot be promoted.

All 555 eligible, using frozen `AGG_PLUS_COMPACT` negative corrections only:

- carry MAE: `2.959194 -> 2.895952`, gain `+0.063242`
- yard MAE: `15.881551 -> 15.471372`, gain `+0.410179`
- carry bias: `+0.025164 -> -0.535101`

Expansion-only:

- carry MAE gain: `-0.087635`
- yard MAE gain: `-0.388092`

The contraction-only replay therefore **does not satisfy the original STACK6B contract** despite the yard gain: carry gain remains far below `+0.20`, and absolute carry-bias worsening is roughly `+0.510`, above the allowed `+0.25`.

### Depth decomposition

`AGG_PLUS_COMPACT` contraction-only, W6-18:

- depth 2: carry gain `+0.036304`, yard gain `+0.407579`
- depth 3+: carry gain `+0.095397`, yard gain `+0.413283`

The contraction signal is not confined to one depth rank.

### Time decomposition

`AGG_PLUS_COMPACT` contraction-only:

- W6-12 all eligible: carry gain `-0.001822`, yard gain `+0.166112`
- W13-18 all eligible: carry gain `+0.130938`, yard gain `+0.664116`

Late-season signal is materially cleaner, but 2025 is exposed development data and this cannot justify choosing a week threshold.

W13-18:

- depth 2 contraction-only: carry gain `+0.098068`, yard gain `+0.620979`
- depth 3+ contraction-only: carry gain `+0.169025`, yard gain `+0.714099`

## Scientific interpretation

STACK6B does **not** show that secondary-back role information is useless.

It shows that a flat, bidirectional continuous residual correction is the wrong architecture:

1. prior situational role can identify cases where P3 is too aggressive on secondary-back workload;
2. the same feature family is poor at identifying justified workload expansion;
3. unrestricted continuous corrections create an excessive negative carry bias;
4. therefore simply zeroing positive deltas is not an acceptable winner;
5. depth rank alone does not distinguish the useful contractions;
6. the next justified work requires either a genuinely different false-high/contraction architecture or new timestamp-safe information that better determines whether a secondary back will actually receive his expected role.

## Retained state

- P3 remains the central point parent.
- M95F-risk and depth-rank-1 protections remain intact.
- Broad STACK6 raw-role regression remains rejected.
- STACK6B compact bidirectional regression remains rejected.
- Compact situational role concepts remain **diagnostic evidence for workload contraction only**, not a retained point model.
- Do not use this feature family to expand P3 carries without genuinely new evidence.

## Next justified direction

Before fitting another point model, audit genuinely new timestamp-safe secondary-back information focused on:

1. exact target-game active/inactive competitor identity where a valid pre-kickoff historical source exists;
2. current usable RB count / competitor return state;
3. prior-game series/drive rotation and substitution stability;
4. opening-series / first-drive role as a lagged tendency only;
5. single-RB vs multi-RB on-field structure and player overlap where reconstructable;
6. team/coach rotation persistence across prior games.

The next source audit must remain no-fit and sportsbook-free. If those sources qualify, freeze the next architecture before fitting it.
