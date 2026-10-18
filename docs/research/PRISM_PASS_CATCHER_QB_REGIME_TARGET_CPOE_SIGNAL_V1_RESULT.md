# PRISM Pass-Catcher × QB-Regime Target-CPOE Signal V1 — Result

Date: 2026-10-17

Status: **COMPLETE / FAILED CLOSED**

Frozen plan:
`docs/research/PRISM_PASS_CATCHER_QB_REGIME_TARGET_CPOE_SIGNAL_V1_PLAN.md`

Source authority:
- run `39087841064` — SUCCESS;
- artifact `12121277666`;
- disposition `PRISM_PASS_CATCHER_QB_REGIME_TARGET_CPOE_STATE_V1_SOURCE_READY_ALL`.

## Canonical scoring authority

- workflow: `Research PRISM Pass Catcher QB Regime Target CPOE Signal V1`
- run: `39088048507` — **SUCCESS**
- exact head: `2f4b0ae6f409a57b645baed2c5cf467f04ae0b32`
- artifact: `12121282907`
- artifact name: `prism-pass-catcher-qb-regime-target-cpoe-signal-v1-39088048507`
- digest: `sha256:1c3d5c6d20bca9afc28f21a51a00164fd063e6460ceb29a181b8eb25043e39ce`
- disposition:
  `PRISM_PASS_CATCHER_QB_REGIME_TARGET_CPOE_SIGNAL_V1_CLOSED`

All workflow steps passed, including frozen safeguards, integrity certification and strict repository audit.

## Frozen candidate

Baseline:
`prior8_mean_cpoe`

Candidate:
`same_qb_mean_cpoe`

No blend, coefficient, threshold, alternate window, subgroup or market translation was fit.

## Results

### WR

2024 development:
- rows: **1,342**
- players: **145**
- MAE: **19.9384 -> 20.0582** (**+0.1198 worse**)
- RMSE: **25.9972 -> 26.0989** (worse)
- p90 AE: **41.4354 -> 41.8680** (worse)
- tie rate: **42.176%**
- candidate closer among non-ties: **48.711%**
- median equal-player MAE delta: **+0.0000**

2025 independent confirmation:
- rows: **1,285**
- players: **147**
- MAE: **19.7923 -> 20.2127** (**+0.4204 worse**)
- RMSE: **25.8358 -> 26.3063** (worse)
- p90 AE: **43.3833 -> 44.6693** (worse)
- tie rate: **42.179%**
- candidate closer among non-ties: **45.222%**
- median equal-player MAE delta: **+0.0583**
- player-cluster 95% CI: **[+0.22533, +0.62421]**
- team-week-cluster 95% CI: **[+0.19369, +0.65189]**

WR qualified: **FALSE**.

The 2025 paired uncertainty is entirely harmful.

### TE

2024 development:
- rows: **574**
- players: **66**
- MAE: **21.4422 -> 22.0396** (**+0.5974 worse**)
- RMSE: **28.1420 -> 29.0589** (worse)
- p90 AE: **41.3661 -> 42.9569** (worse)
- tie rate: **41.986%**
- candidate closer among non-ties: **48.649%**
- median equal-player MAE delta: **+0.0000**

2025 independent confirmation:
- rows: **602**
- players: **70**
- MAE: **21.0106 -> 21.2356** (**+0.2250 worse**)
- RMSE: **27.0096 -> 27.3303** (worse)
- p90 AE: **42.6004 -> 43.9672** (worse)
- tie rate: **40.698%**
- candidate closer among non-ties: **48.459%**
- median equal-player MAE delta: **+0.0149**
- player-cluster 95% CI: **[-0.06588, +0.52007]**
- team-week-cluster 95% CI: **[-0.08469, +0.53435]**

TE qualified: **FALSE**.

## Interpretation

The exact pair-conditioned conversion-quality state is closed.

For WR and TE, replacing the named receiver's ordinary strictly-prior target-CPOE state with his same-QB target-CPOE state does not improve next-game target-CPOE prediction. It worsens every central/tail error metric in both seasons.

Together with:
- same-QB target-share allocation V1 CLOSED; and
- same-QB target-depth V1 CLOSED;

this is now enough negative evidence to stop opening additional simple raw same-QB pair-state replacements from the same historical information family.

A future QB-conditioned mechanism must bring genuinely new pregame football information or representation—not another same-QB mean of target share, depth, CPOE, YPT, catch rate, or a post-hoc blend of them.

## Anti-rescue

Do not:
- blend ordinary and same-QB CPOE;
- fit shrinkage;
- alter support/window gates;
- use >=3-only same-QB history;
- use QB-change-only, WR1-only or TE1-only rescue;
- pool WR+TE;
- substitute completion rate, YPT or target depth;
- translate the failed signal to receiving yards;
- add opponent variables to rescue it;
- use sportsbook inputs or 2026 outcomes.

## Integrity

- candidate variants: 1;
- chronology violations: 0;
- target entitlement loaded: false;
- target entitlement changed: false;
- target depth changed: false;
- receiving yards scored: false;
- receptions scored: false;
- touchdowns scored: false;
- market translation scored: false;
- coefficients fit: 0;
- thresholds searched: 0;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- 2026 outcomes: 0;
- production changed: false.
