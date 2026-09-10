# QB Pass-Rate Down/Distance Decomposition V1 — Result

## Canonical execution

- Branch: `research-qb-pass-rate-down-distance-decomp-v1`
- Frozen plan commit: `5ac5cfca31096dfc53745948846691a27f5fae89`
- Evaluator commit: `70d25d41b20a91d9b0f6a42564cdad8734562632`
- Canonical execution head: `59cb8e953cc89c14892a21081970bcfb8e3d3e24`
- Run: `34543801321`
- Job: `103091966418`
- Artifact: `10178268917`
- Artifact name: `qb-pass-rate-down-distance-decomp-v1`
- Digest: `sha256:8772c9a1203027162b27eee2cf4d9e1fa32acc69af2337163d801e1ba553c8da`
- Canonical disposition: `WITHIN_STATE_PASS_PROPENSITY_PRIMARY_DIAGNOSTIC`
- Production actionable: `false`

## Integrity

All preregistered integrity gates passed.

- exact M89 cohort: 884 QB-games;
- exact one target team-game per row;
- decomposable opportunity-play coverage: 100% pooled, 100% in 2024, 100% in 2025;
- eight down/distance states were exhaustive and mutually exclusive;
- reference and actual occupancy sums reconciled to machine precision;
- target aggregate DBR reconciled direct PBP to `2.22e-16` max absolute error;
- two-factor Shapley identity max error: `3.47e-16`;
- fixed-0.57 three-part identity max error: `3.33e-16`;
- all reference inputs were strictly prior;
- zero sportsbook inputs;
- zero model fitting;
- zero production changes;
- target-game PBP was diagnostic only;
- shared receiver cohorts aligned without duplication.

## Primary finding

The remaining corrected M89 pass-opportunity-rate miss is primarily a **within-down/distance play-selection problem**, not a down/distance occupancy problem and not mainly a league-level centering problem.

Pooled 2024-2025 mean-absolute contribution:

| Component | Mean abs contribution | Absolute mass share |
|---|---:|---:|
| `WITHIN_STATE_RATE` | 0.076715 | 61.47% |
| `LEVEL_VS_057` | 0.030029 | 24.06% |
| `OCCUPANCY` | 0.018063 | 14.47% |

The ranking was stable by season:

- 2024 within-state mean abs: `0.078107`
- 2025 within-state mean abs: `0.075310`
- 2024 level mean abs: `0.029297`
- 2025 level mean abs: `0.030769`
- 2024 occupancy mean abs: `0.018233`
- 2025 occupancy mean abs: `0.017892`

Within-state rate cleared every frozen routing gate:

- largest pooled component: PASS;
- season stability: PASS;
- at least 20% larger than the second-largest pooled component: PASS;
- 2025 WR-target absolute Spearman at least 0.25: PASS.

## Large pass-rate misses

The result strengthens materially in the tails.

For absolute fixed-0.57 rate misses >= 0.08 (n=401):

- within-state absolute mass share: `70.24%`;
- sign agreement: `98.75%`;
- dominant-row rate: `90.27%`.

For absolute fixed-0.57 rate misses >= 0.12 (n=232):

- within-state absolute mass share: `73.97%`;
- sign agreement: `100%`;
- dominant-row rate: `97.41%`.

This is not a small-average-only effect. The largest pass-rate misses are overwhelmingly within-state play-selection shifts.

## Shared receiver attribution

The same within-state play-selection mechanism explains a substantial portion of the shared receiver opportunity miss.

### 2025 WR target-mass residual (n=440)

- actual corrected rate minus 0.57: Spearman `0.508496`
- `WITHIN_STATE_RATE`: Spearman `0.495654`
- `LEVEL_VS_057`: Spearman `0.171894`
- `OCCUPANCY`: Spearman `0.044683`

### 2024-2025 WR reception-mass residual (n=884)

- actual corrected rate minus 0.57: Spearman `0.358991`
- `WITHIN_STATE_RATE`: Spearman `0.379216`
- `LEVEL_VS_057`: Spearman `0.134641`
- `OCCUPANCY`: Spearman `-0.087950`

Within-state rate is therefore not merely a QB bookkeeping effect. It is the strongest identified shared QB/receiver submechanism at this layer.

## State-level context

The largest pooled state-level mean-absolute within-state contributions were:

- first down (`D1`): `0.044837`;
- second-and-long (`D2_LONG`): `0.021366`;
- second-and-medium (`D2_MEDIUM`): `0.018794`;
- third-and-short (`D3_SHORT`): `0.013147`;
- second-and-short (`D2_SHORT`): `0.012792`;
- third-and-long (`D3_LONG`): `0.010588`;
- fourth down (`D4`): `0.006856`;
- third-and-medium (`D3_MEDIUM`): `0.006312`.

First-down play selection is the single largest state-level source of within-state uncertainty. This does **not** authorize a first-down predictive model by itself; the state contributions can offset and remain diagnostic.

## Interpretation

The previously identified shared team-pass-opportunity error has now been localized further:

`TEAM PASS OPPORTUNITY -> PASS-OPPORTUNITY RATE -> WITHIN-DOWN/DISTANCE PASS PROPENSITY`

The model is generally not missing pass volume because it predicts a radically wrong distribution of down/distance states. Instead, teams materially depart from their strict-prior offense/opponent-defense pass tendencies **within the same football situations**, especially on first and second down.

That distinction matters architecturally. The next research target is not generic pace, possessions, score-state occupancy, third-down survival, historical pass-rate weighting, or another 57/43 anchor sweep. It is genuinely new pregame information about **week-specific play-selection intent conditional on game state**.

## Anti-reinvention boundary for next work

Per the frozen stopping rule:

- do not recycle M42 historical pass-rate weighting;
- do not recycle M67/M68 opening-script/playcaller families;
- do not reopen generic score-state, pace, PROE, formation/personnel, no-huddle, or game-script families without genuinely new information;
- do not promote the 0.59 anchor from A1;
- do not fit a generic residual model;
- sportsbook remains downstream only.

The next step may audit genuinely new pregame play-selection information only.
