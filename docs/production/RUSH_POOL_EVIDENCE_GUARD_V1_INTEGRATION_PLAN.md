# Rush Pool Evidence Guard V1 — Frozen Production-Integration Plan

Status: **FROZEN BEFORE PRODUCTION CODE / FULL-STACK OUTCOME SCORING**

Scientific authority:
- result: `RUSH_POOL_EVIDENCE_GUARD_V1_QUALIFIED`
- frozen run: `36045359696`
- result artifact: `10828600510`
- digest: `sha256:d338f9ecca49dddd02e3e501f70b7b4b2efc25d084064299aea4b5e4b3f46e3b`
- canonical result doc: `docs/research/RUSH_POOL_EVIDENCE_GUARD_V1_RESULT.md`

This plan does **not** change the scientific selector. It defines how to test exact production integration safely.

## 1. Exact selector remains frozen

From Week 2 onward:

1. keep every raw `rules_rush_share` unchanged;
2. keep the finite pool size at five;
3. rank positive-share evidenced players first by the existing raw share;
4. use `position_prior_only` players only to fill unused slots;
5. keep the exact existing 95% player-mass cap and residual semantics.

Week 1 remains an exact no-op.

No new:
- pool size;
- share threshold;
- position weight;
- depth weight;
- exception;
- fitted coefficient;
- sportsbook input;
- 2026 outcome condition.

## 2. Production integration architecture

The current simulator consumes one shared RNG stream for many markets. A rushing-probability change must not accidentally move receiving/pass distributions merely because NumPy consumes a different random sequence inside `multinomial`.

Therefore the integration candidate MUST preserve the canonical base RNG consumption.

For each team:

1. compute the exact current baseline top-five rushing shares;
2. call the exact current baseline carry allocation using the canonical RNG, even when V1 is active;
3. this preserves the canonical RNG state seen by every later non-rushing draw;
4. when V1 is active and Week >= 2, independently allocate candidate carries with a dedicated deterministic rushing RNG;
5. derive that RNG seed only from:
   - canonical simulation seed;
   - canonical game identity;
   - canonical team identity;
   - fixed version string `RUSH_POOL_EVIDENCE_GUARD_V1`;
6. discard the baseline carry values for rushing outputs, but retain the RNG consumption they caused;
7. use the candidate carries only for:
   - `rush_att`;
   - `rush_yards`;
   - base `rush_rec_yards` rushing component.

The dedicated seed must use a stable cryptographic hash, not Python's process-randomized `hash()`.

This architecture is required so:
- receptions remain unchanged;
- receiving yards remain unchanged;
- QB passing remains unchanged;
- target allocation remains unchanged;
- only rushing opportunity changes.

## 3. Fail-closed production prerequisites

V1 may activate only when:

- Week is known and >= 2;
- `bayes_evidence_state` exists for every positive-share player in the team pool;
- game identity is present;
- team identity is present.

If these are unavailable or malformed:
- do not infer;
- do not substitute depth chart;
- do not use sportsbook fields;
- fail closed for explicit production certification.

Week 1 always uses the exact baseline path.

## 4. Historical full-stack A/B

Use current canonical timestamp-safe historical construction independently for:

- 2024 Weeks 2-18 with 2023 prior;
- 2025 Weeks 2-18 with 2024 prior.

For each week:

- build one canonical metric frame;
- run baseline and V1 from that exact same frame and seed policy;
- preserve current specialist/ensemble ordering when measuring final outputs;
- score the same actual rows.

No refit.

No new variants.

### Required metrics

Report baseline -> candidate for:

**rush_att**
- ALL;
- RB/FB/HB;
- QB;
- OTHER.

**rush_yards**
- ALL;
- RB/FB/HB;
- QB;
- OTHER.

**RB rush_rec_yards**
- RB/FB/HB only.

Also report:
- MAE;
- RMSE;
- signed bias;
- p90 absolute error;
- 10+ attempt misses;
- 30+ rush-yard misses;
- candidate-closer rate on changed rows.

## 5. Frozen production-integration qualification gates

All of the following are required.

### Scope / invariance

1. Week-1 baseline and candidate simulated outputs are bit-identical for every market.
2. For Week >= 2, `receptions` arrays are bit-identical baseline vs candidate.
3. For Week >= 2, `rec_yards` arrays are bit-identical baseline vs candidate.
4. For Week >= 2, `pass_yards` arrays are bit-identical baseline vs candidate.
5. target-allocation trace is unchanged.
6. team rush-attempt arrays are unchanged.
7. raw `rules_rush_share` values are unchanged.
8. no sportsbook inputs are used.
9. no target-game outcomes enter the selector.
10. RB Rush+Receiving Conservation V2 remains algebraically valid downstream.

### Rush opportunity

11. ALL rush-att MAE improves in both 2024 and 2025.
12. RB/FB/HB rush-att MAE improves in both 2024 and 2025.
13. QB rush-att MAE is non-worse in both seasons.
14. ALL rush-att p90 is non-worse in both seasons.
15. RB/FB/HB rush-att p90 is non-worse in both seasons.
16. OTHER rush-att p90 is non-worse in both seasons.
17. OTHER 10+ attempt-miss count is non-worse in both seasons.

The small allocator-only OTHER MAE regression already observed is not hidden. Production integration does not require OTHER mean MAE improvement, but it may not worsen OTHER tail behavior.

### Rush yards

18. ALL rush-yard MAE is non-worse in both seasons.
19. RB/FB/HB rush-yard MAE strictly improves in both seasons.
20. RB/FB/HB rush-yard p90 is non-worse in both seasons.
21. RB/FB/HB 30+ yard miss count is non-worse in both seasons.
22. QB rush-yard MAE is non-worse in both seasons.
23. OTHER rush-yard p90 is non-worse in both seasons.
24. OTHER 30+ yard miss count is non-worse in both seasons.

### Combined RB output

25. RB/FB/HB rush+receiving-yard MAE is non-worse in both seasons.
26. RB/FB/HB rush+receiving-yard p90 is non-worse in both seasons.
27. RB V2 exact final identity `rush_rec = rush + rec` is preserved rowwise/drawwise where V2 is in scope.

If any gate fails:
`RUSH_POOL_EVIDENCE_GUARD_V1_PRODUCTION_INTEGRATION_FAILED_CLOSED`

No rescue.

## 6. Preserved 2026 confirmation

Only after the historical full-stack gates are applied:

- replay the preserved Week-2 Full Slate artifact;
- compare exact same pregame rows baseline vs candidate;
- verify expected structural changes and non-rushing invariance;
- grade Week-2 outcomes only as a disclosed confirmation;
- do not alter the selector or gates from that result.

No paid OddsAPI pull is authorized.

## 7. Final production certification

If and only if historical integration qualifies:

- wire V1 into the current stable Full Slate production entrypoint;
- add an explicit lineage marker;
- add fail-closed audit output;
- prove Week-1 no-op;
- prove Week-3+ activation;
- prove non-rushing invariance;
- prove RB V2 ordering/identity;
- run Repo CI;
- run preserved paid-artifact replay only;
- do not perform a new paid odds acquisition.

Only then may a promotion PR be merged.

## 8. Stopping rule

Do not rescue a failure with:

- another selector;
- a different evidence-state definition;
- a different pool size;
- a share floor;
- RB-only routing;
- QB-only routing;
- OTHER-position carveouts;
- depth rank;
- injury narrative;
- rookie exceptions;
- new Bayesian strengths;
- 2026 result tuning.

A failure means the exact V1 integration closes.
