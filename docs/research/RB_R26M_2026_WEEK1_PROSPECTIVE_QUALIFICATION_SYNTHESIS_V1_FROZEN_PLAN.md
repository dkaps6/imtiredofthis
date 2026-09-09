# RB R26M 2026 Week-1 Prospective Qualification Synthesis V1 — Frozen Plan

Status: FROZEN BEFORE SYNTHESIS EXECUTION
Date: 2026-09-09
Production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`

## Purpose

R26M is a governance/evidence synthesis only. It asks whether the already-tested, **unmodified** R26 Week-1 vacancy/R9 receiving-entitlement mechanism has enough predeclared evidence to justify design of a separately frozen 2026 Week-1 prospective shadow-candidate build.

R26M does not generate a 2026 prediction and does not evaluate 2026 outcomes.

The synthesis exists because the evidence chain is now structurally complete:

1. R26 established a useful vacancy/R9 mechanism but failed a frozen full-candidate temporal safety gate.
2. R26E isolated the Week-1 component and passed 19/20 frozen gates; the sole failure was 2020 Week-1 season safety.
3. R26J established with source-only pregame evidence that 2020 Week-1 was structurally distinct from 2021-2025.
4. R26K established that no football-coherent 2020 harmful state replicated as a defensible historical router; the relevant states were generally beneficial in 2021-2025.
5. R26L established prospectively, with no 2026 outcomes, that the actual 2026 Week-1 source regime is modern-like rather than 2020-like under a frozen transportability rule.

R26M may synthesize only those already-frozen results. It may not introduce a new retrospective performance threshold, erase 2020, weaken R26/R26E gates, invent a 2020 exemption, refit R9, or change the R26 mechanism.

## Immutable parents

### R26 — original vacancy-gated R9 mechanism

- run `34356222339`
- artifact `10106271075`
- artifact name `rb-r26-vacancy-gated-r9-retrospective-v1`
- digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- required disposition: `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`
- required file: `data/backtests/r26_vacancy_gated_r9_final/r26_final_disposition.json`

R26's failed full-candidate disposition must remain preserved. R26M does not reclassify R26 as historically qualified.

### R26E — latest Week-1 component qualification

- run `34368268224`
- artifact `10110785184`
- artifact name `rb-r26e-week1-component-qualification-v1`
- digest `sha256:2b64fe25a1024136f2bb2cdde42bc74de290b6b9df63b4bb6cd675f6095b8bb7`
- required disposition: `WEEK1_COMPONENT_NOT_QUALIFIED_NO_SHADOW`
- required file: `data/backtests/r26e_week1_component_qualification_v1/r26e_week1_qualification_disposition.json`

This later R26E execution is authoritative over the older 17/18 R26E result file elsewhere in the research tree.

Required frozen gate pattern:
- exactly 20 gates present;
- exactly one false gate;
- the sole false gate must be `14_no_w1_season_worsens_more_than_5pct`;
- all other 19 gates must remain true;
- 5 of 6 Week-1 seasons must have improved;
- 2020 must remain the harmful season with positive relative MAE change;
- no 2020 result may be deleted or reweighted.

This is not a new threshold. It is an exact categorical inheritance of the already-frozen R26E result.

### R26J — 2020 source comparability audit

- run `34374987828`
- artifact `10113466373`
- artifact name `rb-r26j-2020-week1-comparability-source-audit-v1`
- digest `sha256:7d886f228487e052ee1617cc5f42b974cd9da6a9d448f3a86b1de82cb4fc4f46`
- required disposition: `2020_SOURCE_REGIME_DISTINCT_FOR_MECHANISM_FOLLOWUP`
- required file: `data/backtests/r26j_2020_week1_comparability_source_audit_v1/r26j_source_disposition.json`

Required inherited facts:
- all integrity gates passed;
- source-only/pregame-only evidence;
- 10 structurally distinct A-D dimensions across all four A-D sections;
- `exclude_2020_authorized == false`;
- no prediction regeneration or R9 refit.

### R26K — replicated mechanism atlas

- run `34376961740`
- artifact `10114261724`
- artifact name `rb-r26k-week1-allocation-mechanism-atlas-v1`
- digest `sha256:74a3d9ac58fca360f6d2d23e19b28e1254f38e49e3be23878ee1b37cebe22c88`
- required disposition: `2020_SPECIFIC_MECHANISM_NO_REPLICATED_ROUTER`
- required file: `data/backtests/r26k_week1_allocation_mechanism_atlas_v1/r26k_disposition.json`

Required inherited facts:
- all integrity gates passed;
- `qualified_replicated_state_count == 0`;
- no child historical router authorized;
- no prediction regeneration, R9 refit, R22 change, receiving-yard-mean change, or production change.

R26M may not respond to R26K by inventing another router.

### R26L — actual 2026 source-regime transportability

- run `34389455694`
- artifact `10119058769`
- artifact name `rb-r26l-2026-week1-regime-transportability-v1`
- digest `sha256:3351dfb5bbf6b571174a94ddf0a03179d70786edbb91d316e4fdf2c3cf005c46`
- required disposition: `2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION`
- required file: `data/backtests/r26l_2026_week1_regime_transportability_v1/r26l_disposition.json`

Required inherited facts:
- all R26L integrity gates passed;
- all primary 2026 features finite;
- `modern_closer_features >= 5` under the already-frozen R26L rule;
- R26L's already-frozen distance and anomalous-direction rules passed;
- `prospective_qualification_design_authorized == true`;
- 2026 outcomes/participation, sportsbook football inputs, and same-week depth were not used;
- no production/R22/receiving-yard-mean changes.

R26M does not recompute or retune R26L transportability.

## Frozen synthesis question

The R26M qualification question is categorical:

> Given that the unmodified R26 Week-1 component has strong inherited Week-1 evidence but one preserved 2020 failure, that 2020 is source-distinct, that no replicated historical guard is justified, and that the actual 2026 pregame source regime is modern-like, is it scientifically defensible to design a separately frozen 2026 Week-1 shadow candidate using the unmodified R26 vacancy/R9 mechanism?

No new performance fit is performed.

## Frozen qualification contract

R26M returns `2026_WEEK1_UNMODIFIED_R26_SHADOW_CANDIDATE_DESIGN_QUALIFIED` only if **all** of the following are true:

1. all five immutable artifact digests are verified by the workflow;
2. protected production runtime/model paths are clean against `main@f8417f55b04ce0e19baf260e9d532765034c47f1`;
3. R26 has the exact inherited disposition `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW` and preserves its failed temporal gate rather than being rewritten as a pass;
4. R26 structural safety evidence remains intact: sportsbook inputs zero, future-outcome features zero, strict-prior fit true, receiving-yard means unchanged, R22 unchanged;
5. latest R26E has exact disposition `WEEK1_COMPONENT_NOT_QUALIFIED_NO_SHADOW`;
6. R26E has exactly one failed frozen gate, and it is exactly `14_no_w1_season_worsens_more_than_5pct`; all other 19 frozen R26E gates are true;
7. R26E reports exactly 5 improved Week-1 seasons out of the six 2020-2025 seasons, with 2020 preserved as harmful;
8. R26J has exact disposition `2020_SOURCE_REGIME_DISTINCT_FOR_MECHANISM_FOLLOWUP`, all integrity gates pass, and exclusion of 2020 remains unauthorized;
9. R26K has exact disposition `2020_SPECIFIC_MECHANISM_NO_REPLICATED_ROUTER`, all integrity gates pass, and `qualified_replicated_state_count == 0`;
10. R26L has exact disposition `2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION`, all integrity gates pass, and `prospective_qualification_design_authorized == true`;
11. no parent authorizes production promotion or retrospective deletion of 2020;
12. R26M itself regenerates no predictions, fits no model, uses no 2026 outcomes, uses no sportsbook football input, uses no same-week depth, and changes no production parameter, R22 state, or receiving-yard mean.

These are inheritance/consistency conditions, not newly tuned performance gates.

## Alternate disposition

If any scientific inheritance condition above is false while artifact integrity remains valid:

`2026_WEEK1_UNMODIFIED_R26_SHADOW_CANDIDATE_DESIGN_NOT_QUALIFIED`

No candidate build is authorized.

If an immutable parent digest cannot be verified or a protected production boundary is dirty, the workflow must fail closed before scientific synthesis rather than emit a scientific qualification.

## What a qualifying R26M result means

A qualifying result authorizes only **design and frozen construction of a separate R26N-style prospective 2026 Week-1 shadow candidate**.

The R26N candidate, if later frozen, must use the unmodified R26 Week-1 mechanism:
- non-vacancy RB room -> current production baseline exact;
- Week-1 vacancy RB room -> preserved R9 receiving-identity redistribution of the fixed RB receiving opportunity pool;
- RB-room/team receiving mass conserved;
- non-RB outputs exact;
- receiving-yard means exact;
- R22 distribution authority exact;
- no historical 2020 guard or season exemption;
- no R9 refit;
- no sportsbook football inputs.

R26M does **not** itself authorize running that candidate in shadow, publishing 2026 candidate predictions, or changing production.

## Authority ceiling

R26M can never directly authorize:
- production promotion;
- live shadow activation;
- removal/exclusion/reweighting of 2020;
- a new historical router or guard;
- R9 refit;
- receiving-yard-mean changes;
- R22 changes;
- sportsbook-derived football features;
- 2026 outcome use.

Its maximum authority is:

**authorize design/freeze of a separate unmodified-R26 2026 Week-1 prospective shadow-candidate build and structural audit.**

## Required outputs

The R26M evaluator must produce only synthesis evidence, not predictions:
- `r26m_parent_evidence_matrix.csv`
- `r26m_disposition.json`

The disposition JSON must explicitly report:
- each parent run/artifact/digest identity;
- exact inherited parent disposition;
- R26E failed-gate identity/count;
- R26J integrity/distinctness state;
- R26K replicated-router count;
- R26L prospective transportability state;
- production/shadow/exclude-2020/R9-refit/R22/receiving-yard-mean authority flags;
- whether R26N design is authorized.
