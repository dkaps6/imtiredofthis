# Rush-Attempt Zero-MC Allocation Lineage V1 — Final Result

Date closed: 2026-09-26

Disposition:

`RUSH_ATT_ZERO_MC_LINEAGE_CLOSED_TOP5_SUPPORT_EXCLUSION_CONFIRMED`

Research / systems-integrity diagnostic only. Production changed: **false**.

## Frozen authority

- parent production main: `1242d5c0b0a9baa884c3b26a3470e052a1ef1540`
- research branch: `research-rush-att-allocation-lineage-v1`
- authoritative workflow head: `208800ba02a883318b8853c5e412532cf9db149c`
- authoritative run: `36285507895` — **SUCCESS**
- artifact: `10920712168`
- artifact digest: `sha256:487e9a59efffaae45494cf830e30650582e130baae2831507fae7dea9da5959a`
- sportsbook inputs upstream: `0`
- candidate variants: `0`
- fitted parameters: `0`
- production mutations: `0`
- Week-3 outcomes used: `0`

The first two launches were cancelled before interpretation after exact-authority provenance mismatches were found. The authoritative run above matches the frozen source build used by historical fair-probability reconstruction run `36275245917`.

## Exact frozen-cohort reproduction

The hard pre-interpretation gate reproduced the original zero-MC / nonzero-ensemble cohort **exactly**:

- total blocked rows: **7,137**

2024:
- QB: **271**
- RB: **74**
- TE: **1,087**
- WR: **2,117**

2025:
- QB: **249**
- RB: **87**
- TE: **1,122**
- WR: **2,130**

Total rush-attempt rows traced across the full pregame simulator universe: **17,265**.

Positive output-share / zero-MC rows: **7,147**. The 10-row difference versus the 7,137 headline is the exact distinction between positive output share and the frozen nonzero-ensemble blocked cohort.

## First-zero-stage result

Every one of the 7,137 frozen blocked rows first becomes zero at the same stage:

| Season | Position | First zero stage | Rows |
|---:|:---|:---|---:|
| 2024 | QB | `TOP5_EXCLUDED` | 271 |
| 2024 | RB | `TOP5_EXCLUDED` | 74 |
| 2024 | TE | `TOP5_EXCLUDED` | 1,087 |
| 2024 | WR | `TOP5_EXCLUDED` | 2,117 |
| 2025 | QB | `TOP5_EXCLUDED` | 249 |
| 2025 | RB | `TOP5_EXCLUDED` | 87 |
| 2025 | TE | `TOP5_EXCLUDED` | 1,122 |
| 2025 | WR | `TOP5_EXCLUDED` | 2,130 |

No blocked row first failed at:
- selected-row lookup;
- selected-row rushing share;
- final multinomial probability after surviving top five;
- finite-MC sampling;
- keyed `SimulationResult` lookup;
- canonical `mc_proj` transmission.

## Exact parity checks

All hard lineage parity checks passed:

- rush-att output-row share vs simulator-selected-row share mismatches: **0**
- maximum output-row vs selected-row share gap: **0.0**
- canonical `mc_proj` vs same-seed keyed lookup mismatches: **0**
- maximum canonical-vs-keyed gap: **0.0**
- keyed lookup vs realized in-simulator carry-array mean mismatches: **0**
- maximum keyed-vs-realized gap: **0.0**
- independently reproduced top-five output vs in-simulator allocation trace mismatches: **0**
- maximum top-five trace gap: **0.0**

Therefore the allocator, keyed lookup, realized carry vector, and canonical MC transmission are internally consistent.

## Kirk Cousins 2024 Week 1 ATL example

The previously confusing case is resolved by ranking against the **full simulator pregame roster**, not the postgame evaluation subset.

Frozen row:

- player: Kirk Cousins
- team: ATL
- season/week: 2024 Week 1
- rush-att output-row `rules_rush_share`: **0.1015381584**
- simulator-selected share: **0.1015381584**
- full-roster rushing-share rank: **6**
- top-five member: **0**
- post-top-five share: **0.0**
- final player probability: **0.0**
- realized MC carry mean: **0.0**
- keyed lookup mean: **0.0**
- canonical `mc_proj`: **0.0**
- ML rush-att projection: **2.4113288409**
- State rush-att projection: **3.3177921758**
- calibrated ensemble projection: **1.6759115431**
- actual rush attempts: **1**
- first zero stage: `TOP5_EXCLUDED`

The earlier apparent contradiction came from ranking a postgame/evaluation subset that omitted other pregame-roster players. The production simulator ranks the full pregame roster.

## Interpretation

The historical zero-MC / nonzero-ensemble state is **not**:

- an RNG defect;
- a multinomial-normalization defect;
- a player-row deduplication defect;
- a keyed-lookup defect;
- an MC-to-pricing transmission defect.

It is the deterministic support behavior of the production rushing architecture:

`positive pre-selector rushing signal -> outside literal team top five -> zero MC support`

At the same time, ML/State can retain nonzero rushing-attempt signal for the same player, so the calibrated ensemble mean can remain nonzero while the empirical MC distribution is all zeros.

That is a real architectural contradiction between:
- a hard-support Monte Carlo component; and
- softer independent component means.

It is not evidence that the top-five selector is mechanically broken.

## Governance / stopping rule

This diagnostic authorizes **no repair**.

Do not use this result to:
- inject ensemble means into zero-MC rows;
- create a QB carveout;
- create an RB carveout;
- change top five to another top-N;
- add depth-chart exceptions;
- add role exceptions;
- add share thresholds;
- revive Rush Pool Evidence Guard variants;
- fit a replacement pool rule from these outcomes.

The Rush Pool Evidence Guard family already received its separate frozen test/integration path and is closed. This lineage result must not be used to rescue that family under another name.

The proper disposition of this lane is:

`CLOSED_AS_HARD_SUPPORT_CONFLICT_NO_REPAIR_AUTHORIZED`

## Next systems-integrity lane

Proceed to the already-frozen **Discrete Count Mean Alignment V1** production-integration carry-forward.

That lane is independent of zero-MC:
- its qualified helper preserves zero/nonfinite-MC rows as exact no-ops;
- its historical integration mechanics already passed;
- no zero-MC repair is bundled with it.

No paid OddsAPI acquisition is required for that work.
