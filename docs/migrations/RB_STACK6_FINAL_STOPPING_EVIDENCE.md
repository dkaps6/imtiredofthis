# RB STACK6 Final Stopping Evidence

## Purpose

This document freezes the terminal evidence from the RB STACK6 team-rushing research loop before final RB qualification. It is a permanent stop rule, not a production promotion.

The football-first objective remains accurate real RB projections. Sportsbook/market information remains downstream benchmark only and was not used as an upstream input in any of the experiments summarized here.

## Retained parent

The retained RB point parent remains **P3 / STACK3**:

- Week 1: full-stack rushing-yard projection.
- Weeks 2–18: STACK2 enriched within-RB opportunity/allocation × full-stack implied efficiency/context.
- Authoritative STACK3 run: `33539468967`.
- Authoritative STACK3 SHA: `9d7ea5d0173569ac9e4633685da7e91eed5fcd3d`.
- Authoritative STACK3 artifact: `9812993290` (`rb-stack3-frozen-state-composition`).
- 2025 all-RB rushing-yard MAE: approximately `19.949524`.
- Exact 899-row downstream market benchmark: P3 `24.315798` MAE versus Vegas consensus `23.701891` MAE.

M95F workload/tail state and M95I vacancy/transition state remain diagnostics; they are not substituted for the P3 central point estimate.

## STACK6Q — pregame designed-run-call model did not qualify

Authoritative execution:

- Run: `33653547460`
- Job: `100326322852`
- SHA: `db0cc80abc245c0af349e52d2aeabf5a6ea53ce5`
- Artifact: `9855777371`
- Artifact digest: `60bff542e8495b208958a21b891f02e3d5b9806c6b03ffb465ff271e27a5ad76`

Frozen architecture: state-specific Ridge model using 20 predeclared football-only features, fixed alpha 10, trained on the leakage-safe 2024 W13–18 M94C holdout environment and evaluated on 2025. No sportsbook inputs, no feature/model/hyperparameter/threshold search.

W6–18:

- M94C team-rush MAE: `6.2034547805`
- STACK6Q MAE: `6.1354735988`
- MAE gain: `+0.0679811817`
- RMSE gain: `-0.0211978398`
- correlation gain: `-0.0027823031`
- false-high pool >=5 MAE gain: `+0.6417023657`
- false-low pool >=5 MAE gain: `-0.1433000628`
- W13–18 MAE gain: `+0.0594100952`

Only 3 of 7 frozen scientific gates passed. Disposition: **`STACK6Q_DESIGNED_RUN_MODEL_NOT_QUALIFIED`**. P3 recomposition was not authorized. There is no STACK6Q retuning exception.

## STACK6R — down/distance occupancy was not the missing architecture

Authoritative execution:

- Run: `33654134942`
- Job: `100328316397`
- SHA: `3018def749533b52264d254821394ac368427b5d`
- Artifact: `9856011628`

No-fit oracle separated context occupancy from the conditional designed-run decision. Under both strict-prior team-history schemes, only roughly 4–5% of the designed-run error was attributable to down/distance occupancy; approximately 95–96% remained in the conditional run/pass decision once the situation was known.

Disposition: **`CONDITIONAL_CALL_DOMINANT`**.

Durable implication: do not build a drive/down generator merely to repair the current RB team-rush error.

## STACK6S — contextual run/pass efficiency advantage did not qualify

Authoritative execution:

- Run: `33654839590`
- Job: `100330692497`
- SHA: `fb0208b44588e8ad5dfc8ce748d7f560bebd1fe2`
- Artifact: `9856350334`
- Artifact digest: `5fdd8b469edcb0aae6f1f321cc8c5a27c0b960b09c26d7dc0aef8ee775fa3b54`

The no-fit, play-level audit tested two predeclared contextual signals using strict-prior offense/defense histories for the exact score-state × down/distance context:

1. EPA run-vs-pass advantage.
2. Success-rate run-vs-pass advantage.

There were `23,820` W6–18 decision plays. Neither signal qualified under both frozen TEAM5 and TEAM8 schemes. The strongest case, TEAM5 EPA advantage, had correlation `0.021890` and top-minus-bottom residual spread `0.029480`, below the frozen qualification strength.

Disposition: **`CONDITIONAL_ADVANTAGE_SIGNAL_NOT_QUALIFIED`**.

Durable implication: do not create another fitted RB team-rush model whose novelty is merely contextual run-vs-pass EPA/success advantage.

## STACK6T — fine-state context was not primary

Authoritative execution:

- Run: `33655387002`
- Job: `100332525752`
- SHA: `09ee7904199765f6aceb939e55ca649f5f681337`
- Artifact: `9856767713`
- Artifact digest: `9196b9903575ac9ec23e3bcab6e1174109b756c5d1a01dc869fd168eb568acb7`

The no-fit oracle expanded the conditional context to score state × down/distance × field position × game phase. It exactly reproduced STACK6R parents before correction.

TEAM5:

- all W6–18 parent MAE `3.963615` -> fine-context `3.894427`
- recovery fraction `1.7456%`
- false-high pool >=5 recovery `9.5254%`
- false-low pool >=5 recovery `5.1385%`
- W13–18 recovery `2.7410%`

TEAM8:

- all W6–18 parent MAE `4.015161` -> fine-context `3.912663`
- recovery fraction `2.5528%`
- false-high pool >=5 recovery `11.1141%`
- false-low pool >=5 recovery `4.6911%`
- W13–18 recovery `2.4186%`

Disposition: **`FINE_STATE_CONTEXT_NOT_PRIMARY`**.

## Permanent STACK6 stop rule

The team-rushing loop has now established the following chain:

1. Remaining P3 RB-pool misses are predominantly upstream total-team-rushing misses rather than RB-room share misses (STACK6H).
2. Within total-team-rushing error, effective rushing rate dominates offensive play-count error (STACK6I).
3. Within rushing-rate error, conditional designed-run tendency is the largest theoretical component (STACK6K/P).
4. A compact pregame designed-run model does not rank both sides of the distribution reliably enough (STACK6Q).
5. Down/distance occupancy does not explain the missing variation (STACK6R).
6. Contextual run/pass EPA and success-rate advantages do not qualify as stable predictors (STACK6S).
7. Field position and game phase add only a small oracle recovery overall (STACK6T).

Therefore:

- **No STACK6U feature-fishing continuation is authorized.**
- **No additional score/clock/down-distance/field-position slicing is authorized** absent a genuinely new, timestamp-safe pregame information source or materially different architecture justified independently of exposed 2025 outcomes.
- Do not retune STACK6Q on 2025.
- Do not waive failed STACK6 gates.
- Remaining week-to-week designed-run-call variation should be treated as approaching a pregame uncertainty/noise boundary under the football-only inputs tested so far.
- P3 remains the retained point parent pending the final RB qualification package.

## Next action

Proceed to **RB Final Qualification** using the exact original STACK3 1,393-row football casebook and exact STACK5 899-row downstream market universe. This next phase is evaluation/decision only: no model fitting, no feature search, no threshold optimization, and no sportsbook inputs upstream.
