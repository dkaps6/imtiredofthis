# CURRENT NFL RESEARCH HANDOFF

This is the authoritative continuity checkpoint for the `NFL stuff` project. A fresh ChatGPT conversation should read this file first, then verify the listed authoritative GitHub Actions run/artifact before continuing research.

## Repository

- Repo: `dkaps6/imtiredofthis`
- Current research branch at this checkpoint: `research-rb-m95h-lead-role-entitlement`
- Stable continuity branch/ref: `research-current-state`
- Last known production `main` SHA: `7532a2c29dde78a5c3758eb1427561cfed801d67`
- No RB research from M91 through M95H has been promoted to production.

## Non-negotiable modeling rules

1. Predict real football first.
2. Sportsbook/player-prop lines are downstream benchmarking and decision inputs only; never feed them upstream into football projections.
3. Use 2024 for development/selection and preserve untouched 2025 validation for the final test unless a migration explicitly establishes a new prospective protocol.
4. Do not retune on 2025 after seeing results.
5. Do not waive production gates simply because one tail slice improves.
6. Mechanical/source-contract corrections are allowed, but they must not change the scientific candidate grid or validation criteria after validation exposure.
7. No fake/synthetic sportsbook lines.

## Production/data state

Phase A production/data cleanup Waves 1-4 is complete and accepted.

Sportsbook operational rules:

- ordinary pushes use `FETCH_LIVE_ODDS=false`;
- live odds are fetched only by explicit live-odds workflows;
- sportsbook data remains downstream;
- 2026 Week 1 live player-prop acceptance is not yet considered fully validated because preseason no-credit rehearsals could not exercise normal live prop inventory.

## QB state — frozen

Broad QB mean-model research is frozen after M90 and should not be reopened without a specific new hypothesis.

M90 headline improvement:

- MAE roughly `60.63 -> 56.56`
- RMSE `75.63 -> 69.63`
- correlation `.173 -> .243`
- 100+ yard misses `81 -> 64`

Frozen ensemble from PR #498:

- MC weight `.208753`
- ML weight `.267121`
- State weight `.524126`

## Current research sequence

1. Production/data cleanup — complete.
2. RB refinement — current.
3. WR refinement — next major position family after RB is scientifically closed.
4. Dedicated TE research.

## RB baseline and core findings

### M91 temporal RB baseline

2025 RB-only ML:

- rush attempts MAE `3.494731`
- rush yards MAE `21.018907`
- rush+receiving MAE `25.352140`

Never confuse the legacy all-player rushing scoreboard (~7.76-8 yards) with RB-only rush-yard MAE.

### M92 oracle decomposition

Opportunity architecture is the primary RB failure.

20+ carry games had roughly `-8.4` carry bias. Perfect team volume and perfect player share both produced large oracle gains. Correct carries also unlocked large rush-yard gains.

### M93 / M93B

Universal concentration helps the extreme workload tail but harms middle slices. Role-aware concentration is real but insufficient alone.

### M94 / M94B / M94C

Direct team-rush modeling compresses toward the middle. Explicit football game-state decomposition improves team rushing opportunity.

M94C 2025:

- team rush MAE `5.812091`
- RB carry MAE `3.411003`
- 25+ carry MAE `11.954550`

M94C is currently the central carry/opportunity reference architecture during tail research; it was not promoted because the legacy guard had a tiny failure.

### M94D

Combining M94C game environment with M93B backfield concentration improved 20+/25+ tail errors but worsened ordinary/middle slices and failed the legacy guard.

Critical structural lesson: sharpening lead-RB share of a carry pool that is itself too small cannot generate realistic 25+ absolute carry projections.

### M95G — role availability

Authoritative run:

- workflow `M95G RB Role Availability v5`
- run `33396339232`
- job `99501648190`
- tested SHA `39d163048d94f733596098e479334cbf7613f87f`
- artifact ID `9759476538`
- artifact SHA256 `9ecf458c782686cc265a1f2c763f70c02fd2c77ab3f9e7a59321e13f2d78e08b`
- disposition `RETAIN_M95G_AS_DIAGNOSTIC_DO_NOT_PROMOTE`

M95G proved current-week roster/injury/depth/role information adds real 20+ carry signal:

- 2025 20+ AUC `.84647 -> .85973`
- Brier `.06264 -> .06113`
- log loss `.20879 -> .20103`

But 25+ ranking degraded:

- AUC `.84432 -> .82630`

Critical finding: **a vacancy is not a successor**. A generic vacated-lead-role flag can apply to several surviving RBs even though only one may inherit the workload.

Full results: `docs/migrations/M95G_RB_ROLE_AVAILABILITY_RESULTS.md`.

## Latest completed migration: M95H — lead-role entitlement

### Authoritative run

- Workflow: `M95H RB Lead-Role Entitlement v2`
- Run: `33399681980`
- Job: `99512643135`
- Tested SHA: `61f509ea6ad99fc8827db3b4a88e48508063db29`
- Research branch: `research-rb-m95h-lead-role-entitlement`
- Artifact: `migration-95h-rb-lead-role-entitlement-v2`
- Artifact ID: `9760755551`
- Artifact SHA256: `ec792214c84016d93600dfbef87583fdcd5ac7a3d627fef7f13a89556f95dee5`
- Artifact size: `155,724` bytes
- Execution: success
- Scientific disposition: `RETAIN_M95H_AS_DIAGNOSTIC_DO_NOT_PROMOTE`
- Production change: `0`
- Sportsbook inputs: `0`
- M94C central carry mean preserved: `1`

The first M95H run (`33399534207`) failed mechanically because pandas 1.5 `Int64Index` lacks `.ne()`. The v2 wrapper replaced that operation with `!=` only. It did not change models, features, development selection, or validation gates.

### M95H hypothesis

Predict the specific upcoming backfield owner rather than giving every surviving RB a generic vacancy signal.

Targets:

1. `P(player leads team RB carries)`
2. `P(player gets >=60% of team RB carries)`
3. `P(player gets >=70% of team RB carries)`

Pregame features included current available depth rank, historical usage rank among available RBs, prior RB1/RB2 identity and availability, depth promotion, competitor historical shares/carries/targets, and recipient-specific vacancy interactions.

### 2025 untouched results

Exact lead-RB identity did not improve:

- history top-1 accuracy `82.8947%`
- M95H `82.8947%`
- history AUC `.922362`
- M95H `.917416`
- Brier `.099155 -> .100566`

Vacancy successor identification remained difficult:

- history top-1 `62.22%`
- M95H `60.00%`

>=60% share was mixed:

- AUC `.907038 -> .913600`
- Brier `.101230 -> .102402`
- share60 validation gate failed.

>=70% share is the strong validated signal:

- AUC **`.903118 -> .919599`**
- Brier **`.096200 -> .090868`**
- log loss **`.298759 -> .280397`**

Incumbent-available >=70%:

- AUC `.91386 -> .92756`
- Brier `.09614 -> .08974`

Late-week >=70%:

- AUC `.91797 -> .92873`
- Brier `.08176 -> .07413`

Vacancy >=70%:

- AUC **`.78401 -> .86501`** — strong ranking improvement
- but mean probability became overconfident (`18.33%` predicted vs `7.76%` actual), and Brier worsened.

M95H gates:

- lead_pass `0`
- share60_pass `0`
- share70_pass `1`
- incumbent_guard `1`
- validation_pass `0`

Full results: `docs/migrations/M95H_RB_LEAD_ROLE_ENTITLEMENT_RESULTS.md`.

## Current scientific interpretation

We have progressively isolated the RB workload-tail problem:

- game environment/team opportunity is real (M94C);
- historical backfield concentration is real but blunt (M93B/M94D);
- current-week role/availability adds 20+ signal (M95G);
- exact replacement identity is still hard (M95H lead target);
- **recipient-specific probability of a >=70% RB carry share is now a validated incremental signal** (M95H).

The important distinction is that M95H does not justify a generic entitlement boost. Its validated component is specifically the deep-concentration / >=70% share signal.

## NEXT MIGRATION — M95I

Name: **M95I — Calibrated Deep-Concentration + Workload-Tail Integration**

Scientific question:

> Can the validated M95H probability that a specific RB commands >=70% of team RB carries be calibrated by role-transition regime and combined with M95F's 20+/25+ workload-regime signal and M94C opportunity architecture to selectively expand the carry tail for the right backs without materially damaging ordinary games?

### Required architecture

Use:

- M94C central/team rushing opportunity;
- M95H `P(RB share >=70%)` only as the validated entitlement/concentration component;
- M95F 20+/25+ workload-tail probabilities.

Do **not** treat M95H's failed exact-lead probability as validated.

Calibrate at least two regimes separately:

1. incumbent/role-stable;
2. vacancy/role-transition.

Reason: M95H vacancy >=70% ranking is strong but its raw probability is overconfident.

### Development protocol

- all integration/calibration architecture selected with 2024 only;
- freeze before 2025;
- evaluate untouched 2025 once;
- no sportsbook;
- preserve M94C as the central mean unless the integration experiment explicitly applies a pre-specified tail transformation;
- no production promotion unless all gates pass.

### Required diagnostics

- all RB carry MAE/bias/correlation;
- actual carry slices 0-5, 6-10, 11-14, 15+, 20+, 25+;
- projected carry max and quantiles;
- counts projected >=18 / >=20 / >=22 / >=25;
- 20+/25+ event recall and precision;
- mean projection on actual 20+/25+ games;
- ordinary-game damage;
- stable-workhorse false positives;
- vacancy vs incumbent calibration;
- top tail false-positive examples;
- rush-yard and rush+receiving downstream effects when carries are changed;
- legacy all-player rushing guard.

M95I should be considered a **selective integration experiment**, not permission to increase tail coefficients manually after observing 2025.

## What not to do next

- Do not promote M95G or M95H wholesale.
- Do not manually boost gamma/coefficients because 25+ games remain compressed.
- Do not feed sportsbook lines into the football model.
- Do not reopen broad QB mean research.
- Do not forget ordinary-game performance simply because tail slices improve.

## Fresh-chat startup procedure

A new chat should:

1. Open this file from branch/ref `research-current-state`.
2. Verify the latest authoritative run/job/artifact listed above through GitHub Actions.
3. Read the latest migration result document if more detail is needed.
4. Continue directly from the `NEXT MIGRATION` section.
5. Update this handoff document and advance `research-current-state` after completing the next migration.
