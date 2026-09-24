# RB Rush + Receiving Conservation V2 — Draw-Level Integration Plan

Date: 2026-09-24

Status: FROZEN BEFORE CANDIDATE INTEGRATION RUN

Parent authority:
- mean result: `RB_RUSH_REC_CONSERVATION_V2_MEAN_QUALIFIED`
- mean result commit: `42cec9538431d3fb45e335c57ddbff40d621a4c7`
- historical run: `36005675177`
- historical artifact: `10809812602`

## Purpose

Turn the qualified no-fit mean identity into a coherent pricing distribution for **non-Week-1 RB/FB rush+receiving yards** without changing standalone rush-yard or receiving-yard football means.

Week 1 remains under the existing P3/R22/R26 contracts and is explicitly out of scope for V2.

## Candidate construction

For one non-Week-1 RB/FB player-game:

1. obtain the joint simulator's `rush_yards` draw array;
2. obtain the same joint simulator's `rec_yards` draw array;
3. compute the final standalone generic ensemble mean for `rush_yards`;
4. compute the final standalone generic ensemble mean for `rec_yards`;
5. multiplicatively align each nonnegative component array to its own final standalone mean, exactly as canonical pricing already aligns an MC array to a final football mean;
6. define every rush+receiving draw as:

   `conserved_draw[i] = adjusted_rush_draw[i] + adjusted_rec_draw[i]`

7. use the resulting component-sum mean as the authoritative football `model_proj` for `rush_rec_yards`.

The existing independently calibrated `rush_rec_yards` ensemble remains audit metadata only on candidate-applied rows. It must not pull the conserved mean back away from the component identity.

No coefficient, blend, cap, YPC adjustment, role router, or sportsbook field enters this construction.

## Scope

Candidate applies only when all are true:
- runtime week != 1;
- position family RB or FB;
- source/canonical market = rush+receiving yards;
- same player-game has finite standalone rush and receiving simulation arrays;
- same player-game has finite final standalone generic ensemble means.

Fail closed on missing component authority.

## Research integration mechanism

Implementation is research-gated with an environment flag defaulting OFF.

The exact same branch/code and preserved Week-2 input artifact will be run twice:
1. baseline flag OFF;
2. candidate flag ON.

This avoids attributing unrelated post-Week-2 code drift to the candidate.

Preserved football/sportsbook input artifact:
- Week-2 paid-origin run: `35282021679`
- artifact: `10523345092`

No new OddsAPI pull is permitted.

## Mechanical gates

Candidate run must satisfy all:

1. candidate and baseline price the same offer identity set;
2. all non-`rush_rec_yards` `model_proj` values are bitwise/numerically unchanged within 1e-10;
3. standalone RB/FB `rush_yards` `model_proj` unchanged within 1e-10;
4. standalone RB/FB `rec_yards` `model_proj` unchanged within 1e-10;
5. every applied RB/FB combo `model_proj == standalone rush model_proj + standalone rec model_proj` within 1e-8;
6. every constructed combo draw is finite and nonnegative;
7. pathwise combo identity max gap <=1e-10 at the integration seam;
8. no QB/WR/TE/other-position distribution is changed;
9. candidate does not alter sportsbook inputs, lines, odds, offer eligibility or player universe;
10. no duplicate priced offer rows are introduced.

## Week-2 observational check

After the mechanical gates pass, compare the candidate mean to already-settled Week-2 combo outcomes as a **reported confirmation only**, not a tuning gate.

The candidate formula and all gates were frozen without using Week-2 outcomes.

## Advancement

If mechanical gates pass:
`RB_RUSH_REC_CONSERVATION_V2_INTEGRATION_PASS`

This authorizes production-promotion review for Week 3.

If any gate fails:
`RB_RUSH_REC_CONSERVATION_V2_INTEGRATION_FAIL`

Do not rescue by changing the component formula or searching weights.

## Production protections

- Week-1 P3/R22/R26 untouched.
- M96 untouched.
- QB pass yards untouched.
- no global distribution rescale.
- sportsbook remains downstream.
