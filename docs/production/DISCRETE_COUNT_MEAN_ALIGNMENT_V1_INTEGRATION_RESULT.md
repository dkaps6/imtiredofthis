# Discrete Count Mean Alignment V1 — Current-Main Integration Result

Date: 2026-09-26

Disposition:

`DISCRETE_COUNT_MEAN_ALIGNMENT_V1_INTEGRATION_PASS_READY_FOR_PROMOTION`

Production promotion is gated on merge of PR #647. This result itself does not spend OddsAPI credits and uses no Week-3 outcomes.

## Current production authority

- integration branch: `integrate-discrete-count-mean-alignment-v1-current`
- parent main at implementation: `1242d5c0b0a9baa884c3b26a3470e052a1ef1540`
- current main during final review: `177049cc6234f8d9f9b7e9b0401bd544d4310ff5`
  - the only intervening main commit is the research-only rush-att zero-MC lineage closure;
  - no intervening production code changed.
- exact code head that passed integration: `7ca41dff47bda37122b1f02e174b67da1c918220`
- draft promotion PR: #647

The branch preserves the merged Week-3 production repair from PR #645, including the complete Bayesian football authority passed into injury redistribution.

## Frozen research authority

- research disposition: `DISCRETE_COUNT_MEAN_ALIGNMENT_V1_QUALIFIED_FOR_INTEGRATION_TEST`
- authoritative research run: `36276366140`
- result artifact: `10916528395`
- digest: `sha256:0563d974f8c25de290aabec9c92665c8e509fce99c2b83c0ec10e9c5a5c69102`

The candidate is unchanged:
- receptions + rush_att only;
- largest-remainder integer-preserving projection after the existing continuous mean-alignment guard;
- stable draw index tie-break;
- zero/nonfinite-MC rows remain exact current-production no-ops;
- every non-count market retains exact continuous legacy semantics.

No coefficients, thresholds, position carveouts, width adjustments, betting gates, or sportsbook-upstream features were added.

## Authoritative current-main integration

Workflow:
- run `36286418528` = **SUCCESS**
- code head `7ca41dff47bda37122b1f02e174b67da1c918220`
- artifact `10920344026`
- digest `sha256:ac0834f33755eb273f91f4d1f25f12d2a0de088995b14bb95c7d3f09ac161b4c`

Focused production-helper tests: **PASS**

Exact immutable parity:
- rows checked: **19,845**
- count rows with V1 applied: **12,698**
- production no-op rows: **7,147**
- integer failures: **0**
- max array gap vs frozen research: **0.0**
- max mean gap vs frozen research: **0.0**
- max CRPS gap vs frozen research: `3.552713678800501e-15`

Non-count legacy invariance:
- pass_yards: **true**
- rush_yards: **true**
- rec_yards: **true**
- rush_rec_yards: **true**
- anytime_td: **true**

Governance:
- parameters fit: **0**
- sportsbook inputs to football: **0**
- Week-3 outcomes used: **false**

The production static routing audit also passed.

## Shadow-safety / zero-MC compatibility

The implementation preserves the existing `adjusted_outcomes` binding and its legacy if/else guard before invoking the V1 representation adapter.

That preserves the RB-PD2 shadow-safety AST contract and means:
- the shadow hook receives the final priced count array exactly as production will use it;
- the discrete helper does not construct a football mean;
- the zero-MC ensemble-transmission lane remains untouched;
- all 7,147 zero/nonfinite-MC rows in the immutable integration authority are exact no-ops.

Repo CI on the current-main integration head:
- run `36286439940` = **SUCCESS**

## Preserved paid-artifact check

PR workflow `36286439933` failed before any replay/model step because the historical artifact named `run_35282021679` is no longer available:

`Artifact not found for name: run_35282021679`

This is not an integration/model failure.

The frozen integration plan explicitly states:
- if an old paid artifact has expired, do not purchase/refetch it merely to satisfy integration testing;
- immutable historical arrays are the primary exact count-path authority;
- no new paid OddsAPI acquisition is authorized.

Therefore this expired-artifact check does not invalidate the integration pass.

## Current Full Slate health

The automatic no-live-odds Full Slate on current main after the zero-MC research closure:
- run `36286302945` = **SUCCESS**
- head `177049cc6234f8d9f9b7e9b0401bd544d4310ff5`

No paid `fetch_live_odds=true` run has been launched.

## Promotion decision

Every frozen promotion requirement that can be satisfied without a new paid odds acquisition has passed:
- exact research parity;
- count mechanics;
- deterministic integer support;
- zero-MC no-op;
- non-count invariance;
- production routing;
- shadow-safety compatibility;
- current-main Repo CI;
- current-main no-live-odds production health.

Disposition:

`DISCRETE_COUNT_MEAN_ALIGNMENT_V1_INTEGRATION_PASS_READY_FOR_PROMOTION`

After promotion, the live Week-3 betting-board gate remains separately controlled and still requires explicit user authorization before any OddsAPI spend.
