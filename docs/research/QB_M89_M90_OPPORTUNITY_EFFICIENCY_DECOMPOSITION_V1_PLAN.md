# QB M89/M90 Opportunity / Efficiency Decomposition V1 — Frozen Plan

**STATUS: FROZEN BEFORE FIRST RESULT. RESEARCH DIAGNOSTIC ONLY. NO PRODUCTION CHANGE.**

Branch:
`research-qb-m89-opportunity-efficiency-decomposition-v1`

Parent main at freeze:
`37deff5b51b5ec48117556b91d12dba1d4ca1fad`

## Purpose

Diagnose the remaining 2024-2025 error of the promoted football-only QB passing-yards mean authority:

`QB_PASS_SYNTHESIS_V1 / M89-M90`

This is not a new QB feature hunt and not a candidate model.

The question is:

> After M89/M90, how much of the remaining passing-yard miss is attributable to passing opportunity (attempts), passing efficiency (YPA), and the non-factor residual created by the ensemble+synthesis architecture?

No sportsbook line, price, implied probability, or betting result may enter the decomposition.

## Anti-reinvention boundary

The M82 integration ledger remains binding.

Do not reopen or retest:
- generic pass-rate / game-script repackaging;
- generic static matchup;
- range decompression;
- trust-stack / extreme-error classifiers;
- possession/dropback generative state;
- model-zoo combinations;
- opening-script/playcaller tendency;
- generic QB efficiency-volatility;
- aggregate explosive weapon x defense;
- official-inactive identity;
- FTN tactical families;
- any feature family marked FULL_STACK_TESTED_CLOSED or SIGNAL_SCREEN_FAILED.

This audit may only identify where error lives. Any future candidate requires a separately frozen plan and genuinely new information consistent with the ledger reopen conditions.

## Source / cohort

Reuse the already-established clean M89 reconstruction contract:

- 2023: frozen training season;
- 2024 + 2025: evaluation only;
- historical inputs rebuilt leakage-safely using the same scripts and season semantics as the clean M89 authority;
- M89 Ridge alpha = 20;
- residual cap = 45 yards;
- base projection = canonical calibrated ensemble;
- final diagnosed projection = football-only `football_synthesis`;
- target outcome = actual passing yards;
- sportsbook features in football model = false.

The expected evaluation cohort is the same clean M89 2024-2025 population (approximately 894 rows), subject only to rows with finite attempts/YPA decomposition inputs.

## Primitive definitions

For each QB player-game:

- `P = pred_attempts`
- `A = actual_pass_attempts`
- `Yp = pred_ypa`
- `Ya = actual_ypa`
- `M = football_synthesis`
- `Actual = actual_pass_yards`

Define the deterministic primitive projection:

`D = P * Yp`

Define the non-factor residual:

`R = M - D`

Because `Actual = A * Ya` for finite positive-attempt rows, the promoted-model signed error is:

`E = M - Actual`

### Exact symmetric product decomposition

Use the two-factor Shapley decomposition of the product difference:

`Opp = (P - A) * (Yp + Ya) / 2`

`Eff = (Yp - Ya) * (P + A) / 2`

Then:

`E = Opp + Eff + R`

Required rowwise identity:

`max_abs(E - (Opp + Eff + R)) <= 1e-8`

This split avoids assigning the attempts×YPA interaction arbitrarily to one side.

## Oracle counterfactuals

Hold the non-factor residual `R` fixed.

- opportunity oracle:
  `M_opp_oracle = A * Yp + R`
- efficiency oracle:
  `M_eff_oracle = P * Ya + R`
- both primitives oracle:
  `M_both_oracle = A * Ya + R = Actual + R`

These are diagnostic counterfactuals only. They are not deployable models.

Report:
- baseline M89 MAE / RMSE / bias;
- opportunity-oracle MAE / recovery vs baseline;
- efficiency-oracle MAE / recovery vs baseline;
- both-primitives-oracle MAE / recovery vs baseline.

## Contribution diagnostics

For each row report:
- signed total error;
- opportunity contribution;
- efficiency contribution;
- non-factor residual;
- absolute contribution magnitudes;
- dominant absolute component;
- sign alignment / cancellation state.

Aggregate:
- pooled 2024-2025;
- 2024 separately;
- 2025 separately;
- absolute-error buckets: <25, 25-49.999, 50-74.999, 75-99.999, >=100 yards;
- catastrophic rows: abs error >=75 and >=100;
- predicted-attempt quartiles;
- predicted-YPA quartiles.

Report contribution shares descriptively. Do not use them as fitted weights.

## Integrity gates

The diagnostic is invalid if any fail:

1. evaluation seasons are exactly 2024 and 2025;
2. zero sportsbook inputs in football synthesis/decomposition;
3. one unique row per season/week/team/player;
4. finite M89 projection, actual yards, predicted attempts, actual attempts, predicted YPA, actual YPA on evaluated rows;
5. actual attempts > 0;
6. `actual_pass_yards == actual_pass_attempts * actual_ypa` within numeric tolerance;
7. rowwise decomposition identity max absolute gap <= 1e-8;
8. zero production files/models/weights changed.

Integrity failure disposition:
`QB_M89_OPP_EFF_DECOMPOSITION_INTEGRITY_FAILURE`

## Scientific disposition

This run does **not** qualify or reject a production candidate.

It may only emit:
`QB_M89_OPP_EFF_DECOMPOSITION_COMPLETE`

along with quantitative evidence.

Any claim that opportunity, efficiency, residual, or tails deserve a new model change must be supported by:
- the decomposition;
- M82 anti-retest ledger;
- a source/schema audit showing genuinely new pregame football information;
- a separately frozen candidate plan before first candidate result.

## Forbidden after seeing results

Do not:
- fit a correction from these same 2024-2025 outcomes;
- tune a threshold by contribution size;
- create an under/over router;
- use market disagreement;
- reweight MC/ML/State;
- change M89/M90 coefficients or cap;
- change C2 distribution logic;
- revive any M40-M81 closed family;
- cherry-pick a season or error bucket.

