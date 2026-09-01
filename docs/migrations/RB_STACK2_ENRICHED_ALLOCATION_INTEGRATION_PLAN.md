# RB-STACK2 — Enriched Allocation + Full-Stack Integration

Status: PRECOMMITTED PLAN
Parent: RB-STACK1 authoritative run 33535308110, SHA b3cd9d34d35e93f9dfc32a61a1a5c222a1df81f8
Purpose: test whether the production-equivalent RB stack and M94C's validated opportunity signal are complementary after feeding missing timestamp-safe football information into backfield allocation.

## Permanent distinction

A stronger full-stack baseline does **not** imply that it already contains every useful football capability discovered in M91-M96E. This migration must test missing football information, M94C opportunity, explicit backfield allocation, and compatible retained modules as distinct jobs.

Sportsbook data is downstream only. No sportsbook value may enter football projections, allocation, feature construction, fitting, routing, or blend selection.

## Parent references

- Full production-equivalent football parent: RB-STACK1 `ENSEMBLE_2024_FROZEN`.
- M94C central opportunity reference: `candidate_rush_att` and `candidate_rush_yards` from frozen M94C artifact/run 33353485070.
- Corrected listed-market benchmark: 899 matched 2025 rushing-yard rows from RB-STACK1 / market run 33499129109.

## Workstream A — Feed missing football information

Build a strictly pregame, timestamp-safe RB allocation state. Use only fields that can be reconstructed as known before kickoff. The initial fixed information families are:

1. week-tagged depth-chart hierarchy / RB1-RB2-RB3 order;
2. lagged snap/participation share and recent snap trend;
3. lagged carry share and touch/opportunity share;
4. competing-RB strength and recent usage;
5. injury / availability / vacancy state;
6. roster/team change state and prior-team continuity;
7. rookie / draft prior when available pregame;
8. backfield concentration / number of materially used RBs;
9. QB and non-RB rushing competition;
10. offensive-line availability only where a timestamp-safe historical source is actually available.

Missing source coverage must be reported explicitly. Do not synthesize unavailable historical information from postgame outcomes.

## Workstream B — First real depth/snap-driven allocation experiment

Estimate team rushing opportunity separately from player allocation.

- Team opportunity anchor: preserve M94C's validated team/game-environment carry signal where applicable.
- Player allocation: distribute RB opportunity using the pregame allocation state above.
- Preserve team-level accounting: allocated player RB carries may not silently create impossible team rushing volume.
- QB/non-RB rushing competition is handled before RB allocation, not after.
- Actual carries/snaps are evaluation-only.

Primary allocation comparison:

- `M94C_RAW`
- `M94C_ENRICHED_ALLOCATION`

This directly answers whether feeding the missing football state improves M94C's opportunity capability.

## Workstream C — Full-stack + M94C integration, including blend tests

Do **not** force an either/or contest between full stack and M94C.

Required point-space arms:

1. `STACK1_PARENT` — frozen production-equivalent RB-STACK1 projection.
2. `M94C_RAW` — frozen M94C projection.
3. `M94C_ENRICHED_ALLOCATION` — M94C with the fixed pregame allocation state.
4. `BLEND_75_STACK_25_ENRICHED` — 0.75 * STACK1 + 0.25 * enriched M94C.
5. `BLEND_50_STACK_50_ENRICHED` — 0.50 * STACK1 + 0.50 * enriched M94C.
6. `BLEND_25_STACK_75_ENRICHED` — 0.25 * STACK1 + 0.75 * enriched M94C.

The three blend weights are a **precommitted sensitivity grid**, not a weight search. They answer whether the two systems carry complementary information. Do not tune intermediate weights against 2025 outcomes after seeing results.

Required architecture arm:

7. `STACK1_PLUS_ENRICHED_ALLOCATION` — use the enriched opportunity/allocation module as a football capability inside the full-stack RB architecture rather than merely averaging final yard outputs. This is the preferred scientific integration if feasible because it preserves the separate jobs of opportunity/allocation and efficiency/context.

Report both point-space blends and architecture integration. A simple blend can win; it is not disallowed. But a blend must earn its role through the same non-degradation and temporal checks as any other module.

## Workstream D — Where retained M95/M96 capabilities fit

These are not dumped into one soup. After the best safe parent from Workstream C is identified, run precommitted ablations by job:

- `M95F`: workload-distribution / stable-workhorse tail layer. Distribution/tail job; not a universal mean boost.
- `M95I`: vacancy / role-transition state. Conditional opportunity/transition job; not a universal point adjustment.
- `M95C` environment signal: efficiency/environment context only where incremental to the chosen parent.
- `M96C D`: conditional opponent/run-resistance efficiency signal only; no unrestricted global point correction.
- `M96C E/P`: conditional blocking/player-created clues only if isolated without double counting.
- `M95D X`: isolated explosive-tail increment remains rejected unless materially new information changes the question.

For each module record: exact capability, regime helped, regime harmed, whether it is redundant with the parent, and compatibility with other retained modules.

## Workstream E — Corrected 899-game Vegas regrade

For every frozen football arm above, regrade the exact same corrected 899-row listed-market subset downstream only.

Required outputs:

- football projection vs actual: MAE, RMSE, bias, correlation;
- Vegas consensus vs actual on the same rows;
- model minus Vegas line distribution;
- directional over/under accuracy where a nonzero model-line edge exists;
- edge buckets fixed before scoring (0-2.5, 2.5-5, 5-10, 10+ absolute yards) for descriptive calibration;
- no odds/line value may alter the football projection.

The Vegas regrade is a benchmark and decision-layer diagnostic, not a football-model training target.

## Evaluation slices

At minimum:

- all RB rows;
- Week 1 vs Weeks 2-18;
- actual carry bands 0-5, 6-10, 11-14, 15-19, 20+, 25+ (evaluation only);
- projected role/depth strata known pregame;
- stable-workhorse vs committee vs vacancy/transition;
- high vs low backfield concentration;
- listed-market 899-row subset.

## Scientific controls

- No postgame features in pregame projections.
- No sportsbook input upstream.
- No weakening prior truth/integrity gates.
- No arbitrary weight search after results.
- Blend grid is fixed at 25/50/75 enriched-M94C contribution.
- If a blend is promising, freeze it as a candidate and require untouched/prospective 2026 confirmation before calling it independently validated.
- Preserve separate opportunity, efficiency, transition, and tail jobs.
- A module may be retained for a narrow job even if it does not replace the whole model.

## Decision logic

This migration is intended to answer four different questions, not one:

1. Does missing pregame football information improve player-level RB allocation?
2. Does enriched M94C improve the production-equivalent full stack as an architectural module?
3. If architectural integration is imperfect, does a fixed simple blend reveal complementary signal worth freezing?
4. Which previously tested M95/M96 modules remain incremental after the new parent exists?

Do not conclude `STACK1 wins` or `M94C wins` without answering all four.
