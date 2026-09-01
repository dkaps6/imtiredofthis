# RB-STACK3 — Frozen Pregame State Composition

Status: PRECOMMITTED DEVELOPMENT PLAN
Parent evidence: RB-STACK2 run 33538770934, job 99959516813, SHA c07150158c5368c02d618f7504d95efed661ed66, artifact 9812754276.

## Purpose

STACK2 established that the strongest current central rushing-yard architecture is:

`enriched M94C opportunity/allocation × full-stack implied efficiency/context`

This migration asks whether already-frozen workload/transition state modules can route the strengths of the existing parents without using target-game outcomes.

2025 has already been inspected. Therefore STACK3 is **development evidence only** and may define a frozen candidate for prospective 2026; it is not independent confirmation.

Sportsbook remains downstream benchmark only.

## Frozen parent projections from STACK2

- `P`: `arch_enriched_opp_stack_eff_yards`
- `STACK`: `stack_yards`
- `ENRICHED_M94C`: `enriched_yards`
- `STACK_EFF`: `stack_implied_ypc`

No STACK2 allocation refit is allowed in this migration.

## Frozen state signals

### Week-1 initialization state

`week == 1` is a deterministic pregame calendar state. STACK2 showed STACK1 parent materially outperformed M94C/enriched opportunity in Week 1. This migration may test an exact Week-1 full-stack override, but the rule is fixed before STACK3 scoring.

### M95F workload-risk state

Use the exact frozen workload-risk guard already carried into M96E:

`M95F calibrated P(20+ carries) >= 0.25 OR M95F p90 workload >= 20 carries`

Source: frozen M95F run `33389924330`, artifact `migration-95f-rb-workload-regime-calibration`, row trace `m95f_2025_rb_trace.csv`.

No probability recalibration and no threshold search.

### M95I vacancy/transition state

Use frozen M95I fields from run `33402566592`, artifact `migration-95i-rb-deep-concentration-tail`, row trace `m95i_2025_trace.csv`:

- `prior_top1_unavailable`
- `m95i_rush_att`
- `m95i_tail_eligible`

No M95I refit, threshold change, or meta-model rerun.

## Precommitted arms

1. `STACK2_PARENT`: P unchanged.
2. `WEEK1_STACK_OVERRIDE`: STACK in Week 1; P otherwise.
3. `M95F_RISK_ENRICHED_OVERRIDE`: ENRICHED_M94C where frozen M95F workload-risk state is true; P otherwise.
4. `WEEK1_PLUS_M95F_RISK`: STACK in Week 1; ENRICHED_M94C in Weeks 2-18 where M95F risk is true; P otherwise.
5. `M95I_CARRY_STACK_EFF`: frozen M95I selective-tail carries × STACK_EFF where M95I tail-eligible; P otherwise.
6. `VACANCY_ENRICHED_OVERRIDE`: ENRICHED_M94C where `prior_top1_unavailable==1`; P otherwise.
7. `WEEK1_RISK_VACANCY_COMPOSITE`: STACK in Week 1; in Weeks 2-18 use ENRICHED_M94C if M95F risk OR M95I vacancy; P otherwise.
8. `WEEK1_RISK_M95I_TAIL_COMPOSITE`: STACK in Week 1; in Weeks 2-18 use M95I carries × STACK_EFF if M95I tail-eligible, else ENRICHED_M94C if M95F risk, else P.

These arms are fixed. Do not invent new thresholds after viewing the outputs.

## Evaluation

For every arm report:

- all-RB 2025 rush-yard MAE/RMSE/bias/correlation;
- Week 1 and Weeks 2-18;
- actual carry bands 0-5, 6-10, 11-14, 15-19, 20+, 25+ for postgame diagnosis only;
- pregame M95F risk vs non-risk;
- pregame M95I vacancy vs incumbent;
- pregame M95I tail-eligible vs not;
- exact same 899 archived-market rows downstream;
- market MAE/RMSE/bias/correlation;
- 0-2.5, 2.5-5, 5-10, 10+ model-vs-market disagreement buckets.

## Scientific interpretation rules

- Actual carries cannot route a projection.
- Vegas cannot route or train a projection.
- A state composition may be retained if it improves a specific regime without unacceptable damage elsewhere, even if it does not replace the whole parent.
- Do not promote a retrospective 2025 winner as independently validated; freeze for prospective 2026.
- If M95F or M95I is redundant after STACK2, record that rather than forcing inclusion.
- If the remaining Vegas gap is concentrated in a different mechanism, reverse-engineer that mechanism and move to the next evidence-backed football module.
