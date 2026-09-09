# RB R26 Receptions Production Integration V2 — Frozen Plan

**This V2 contract supersedes V1 before any qualification run/results were observed.**

## Why V2 exists

During implementation inspection, before any qualification execution, the existing production pricing contract was confirmed to use `MC + ML + state -> calibrated ensemble -> model_proj` for ordinary receptions. V1 Gate 27 incorrectly required final `model_proj` to equal the raw R26-adapted MC mean, which would have silently removed the existing ensemble layer for RB receptions.

That would contradict the user's explicit instruction to **add R26 to the existing logic rather than replace the old logic**.

No qualification result existed when this correction was frozen. V1 remains preserved as lineage and is not retroactively edited.

## Production decision

There will be one authoritative RB receptions output.

For R26-eligible Week-1 RB/FB rows:

`existing football baseline -> R26 entitlement refinement -> R26-adapted receptions MC distribution -> existing unchanged calibrated MC/ML/state ensemble -> one final model_proj -> sportsbook comparison`

The user does not choose between baseline and R26. The baseline MC distribution remains audit evidence only on R26-applied rows.

For the sole non-vacancy control team, CIN, the existing baseline path remains unchanged and still emits one final `model_proj`.

## Protected parents and scientific lineage

All parent authorities, historical evidence, R26 failures, R26L/M/N/O/Q lineage, protected production head `f8417f55b04ce0e19baf260e9d532765034c47f1`, and current-main code-parity facts from V1 remain binding.

R26 is promoted because the user has accepted the historical evidence profile: useful general signal with preserved known historical exceptions, followed by failure forensics, 2026 transportability qualification, prospective construction, MC compatibility, sealing, and pregame operational-readiness validation. No 2026 Week-1 outcome is required before this pregame production decision.

## Frozen football mechanism

The V1 mechanism sections remain binding with these precise production semantics:

- current Full Slate/Ourlads roster is live player authority;
- R26L's exact 31-team Week-1 offseason vacancy set is pinned; CIN is the sole non-vacancy control;
- exact R19/R9 strict-prior player identity scorer through 2025 only;
- no R9 refit;
- preserve the existing RB+FB target-entitlement pool per team;
- softmax `log(baseline_within_share + EPS) + calibrated_R9_residual` inside vacancy rooms;
- non-RB entitlement unchanged;
- candidate simulation may be generated internally, but only RB/FB `receptions` arrays are copied into the protected final simulation result after R22;
- R22 receiving yards, rush+receiving yards, P3 rushing, QB, WR, TE and all other markets remain exact;
- sportsbook data stays downstream only.

## Frozen pricing semantics

For every priced R26-applied RB/FB receptions offer:

1. `mc_proj` must equal the mean of the R26-adapted final receptions array;
2. the existing `ml_proj` and `state_proj` inputs are not rewritten by R26;
3. the existing ensemble code and weights are not changed by R26;
4. `ensemble_proj` is recomputed normally using the new R26 `mc_proj` plus the existing ML/state inputs;
5. `model_proj` must equal that unchanged ensemble output for receptions;
6. fair probability and edge are then calculated from the final adjusted distribution versus the sportsbook line;
7. no second baseline-vs-R26 betting choice is exposed.

This is the same architectural principle used by other promoted football refinements: improve the relevant football component, then allow the existing downstream calibrated stack to consume it.

## Frozen qualification gates

PASS requires all 35 gates. Gates 1-26 and 29-35 retain the intent of V1. Gates 27-28 are corrected as follows before any run:

1. V2 plan predates qualification execution/results and documents why V1 was superseded;
2. protected parent production files byte-identical to `f8417f...` before R26 additions;
3. exact R19 model hash and feature contract;
4. exact R26L 31-team vacancy set and CIN control;
5. live/current Full Slate has all 32 teams and 16 Week-1 games;
6. current RB/FB population covers all 32 teams with unique identities;
7. strict-prior identity maximum time < 202601;
8. R9 refit false;
9. every RB+FB room entitlement pool conserved <= 1e-12;
10. every team total entitlement conserved <= 1e-12;
11. non-RB entitlement delta <= 1e-12;
12. CIN RB/FB entitlement exact baseline;
13. R26 candidate reception arrays finite/nonnegative/integer;
14. final result key universe exact V4 key universe;
15. only vacancy-active RB/FB `receptions` arrays may change;
16. forbidden changed arrays = 0;
17. CIN RB/FB final reception arrays exact V4 baseline;
18. RB/FB `rec_yards` exact V4/R22;
19. RB/FB `rush_rec_yards` exact V4/R22 before existing P3 conservation stage and remains governed by the existing conservation layer afterward;
20. RB/FB rush arrays exact V4/P3 simulation baseline;
21. QB arrays exact V4;
22. WR arrays exact V4;
23. TE arrays exact V4;
24. all non-RB arrays exact V4;
25. R22 integration/mean-preservation audit still passes;
26. P3 final rush pricing remains exact P3 synthesis mean;
27. **every priced R26-applied RB/FB receptions `mc_proj` equals the R26-adapted final MC mean, while R26 does not alter `ml_proj`, `state_proj`, ensemble weights, or ensemble method**;
28. **every priced R26-applied RB/FB receptions `model_proj` equals the existing unchanged ensemble output after consuming the R26-updated `mc_proj`; exactly one final `model_proj` is emitted per offer**;
29. R26 pricing lineage identifies applied rows and stores old baseline MC mean only as audit evidence;
30. Week-1 outcomes used = 0;
31. sportsbook football inputs used = 0;
32. no outcome-derived tuning/router/blend;
33. no same-week outcome data;
34. clean-checkout execution using repo-pinned assets;
35. disposition exactly `RB_R26_WEEK1_RECEPTIONS_PRODUCTION_INTEGRATION_PASS_READY_FOR_PROMOTION`.

## Failure disposition

Any failed gate => `RB_R26_WEEK1_RECEPTIONS_PRODUCTION_INTEGRATION_FAIL_NO_PROMOTION`.

Mechanical failures may receive only a documented value-neutral repair. Scientific/contract failures remain failures.

## Promotion authority

The user explicitly authorized R26 integration into the single authoritative RB receptions model. If and only if all 35 frozen V2 gates pass, this branch may be promoted to `main` without waiting for Week-1 outcomes. R26S remains the postgame audit of the pregame production decision.
