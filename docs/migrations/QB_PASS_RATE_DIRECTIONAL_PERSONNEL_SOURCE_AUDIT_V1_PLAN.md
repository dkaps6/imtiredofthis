# QB Pass-Opportunity Rate Directional Personnel Source Audit V1 — Frozen Plan

## Purpose

Audit whether the project can construct a genuinely new, strictly pregame **directional personnel consequence** information family for the newly isolated `PASS_OPPORTUNITY_RATE` bottleneck.

This is a source/provenance audit only. It does **not** inspect target-game pass-opportunity-rate residuals, QB attempts, passing yards, WR target residuals, or any model-performance outcome. It cannot fit a predictive model and cannot change production.

The exact hypothesis family to be source-audited is:

> week-specific offensive personnel losses may change pass-vs-run intent differently depending on whether the unavailable prior opportunity belonged primarily to the backfield run game or to the receiving game. Prior M77/M79 tests grouped RB/FB/HB/WR/TE together as offensive skill personnel, so opposite directional consequences could cancel inside a generic `off_skill` burden.

No scientific claim is made here that this family is predictive. V1 only decides whether it is cleanly constructible and materially distinct enough to earn a separately frozen predictive test.

## Parent lineage

### Current bottleneck
- Parent branch: `research-qb-team-pass-opportunity-play-rate-decomp-v1`
- Parent result commit: `7dbbc93e42eae68e6032ec6cb2299d357be6cb20`
- Parent tested head: `36f100799c81769d1ade998c7304d932e5914b44`
- Parent Run: `34535405829`
- Parent Job: `103065737323`
- Parent Artifact: `10175200512`
- Parent digest: `sha256:f2a79b330c7cc6f24d6b83479806478e5534afd2bd09961d37304a2c247b6bb4`
- Parent disposition: `PASS_OPPORTUNITY_RATE_PRIMARY_DIAGNOSTIC`

### Corrected official inactive authority
- M78 corrected source Run: `33288381864`
- M78 corrected Artifact: `9725181841`
- M78 source head: `426fde17668aad049e887ba8da0776a28a4dc9ca`
- Artifact digest: `sha256:af0d72a31bc522791bb887dea9ba4f2087e41949be1db15f92973c06140106e6`
- Persisted CSV: `data/backtests/qb_m78_official_inactives_corrected/m78_corrected_candidate_teamweek.csv`
- CSV SHA256: `d39aaf0feea101f3e0d2721ebd4118ef33fb1a4d3c76670e2a4f17734e37b609`
- Rows: `1088` = `544` each for 2024 and 2025.

## Anti-reinvention boundary

This audit must remain materially distinct from M77/M79.

### M77 already tested
- OL turnover / addition / replacement deficit / role delta;
- aggregate offensive skill turnover / addition / replacement deficit / role delta;
- defensive DB and pass-rush turnover / role / quality;
- selected OL × rush and skill × DB interactions.

### M79 already tested
- official inactive identity with prior role weighting;
- `off_ol_count`, `off_ol_role`, `off_ol_high`;
- aggregate `off_skill_count`, `off_skill_role`, `off_skill_high` where `SKILL={RB,FB,HB,WR,TE}`;
- defensive DB/rush inactive burdens and rush quality.

### Prohibited repackaging
Do not recreate generic inactive counts, generic skill-role burden, generic injury burden, generic depth discontinuity, or a different model on the same M77/M79 feature set.

### Materially new distinction being audited
The source family must preserve separate football channels:

1. `BACKFIELD_RUN_CAPACITY_LOST`
   - unavailable RB/FB/HB prior rushing opportunity;
2. `RECEIVING_CAPACITY_LOST`
   - unavailable WR/TE/RB/FB prior target opportunity;
3. optional position-specific receiving partitions may be audited for provenance only:
   - `WR_TE_RECEIVING_CAPACITY_LOST`
   - `RB_FB_RECEIVING_CAPACITY_LOST`.

The intended future mechanism is directional imbalance, not total burden. No imbalance formula may be optimized in this source audit.

## Frozen source universe

### Target team-weeks
- exact corrected M78 2024 and 2025 regular-season team-weeks;
- target identity is `(season, week, team)`;
- inactive identities/positions are fixed pre-kickoff facts from the corrected M78 snapshot.

### Player identity source
Use nflverse weekly rosters for 2024 and 2025 to map the corrected inactive name tokens to stable player/GSIS identity.

The source audit must report:
- total relevant inactive tokens;
- uniquely mapped tokens;
- ambiguous tokens;
- unmapped tokens;
- mapping rate overall and by season/position family.

No fuzzy mapping may be silently accepted. Any non-exact normalization fallback must be separately labeled and reported.

### Strict-prior usage source
Primary usage source: nflverse weekly player statistics for seasons 2023, 2024, and 2025.

Audit the actual schema and select the first semantically exact available fields from these allowed concepts only:
- rushing attempts/carries;
- targets;
- receptions as a target-source diagnostic only, not a substitute for targets if targets are unavailable;
- recent team / team identity;
- player stable identity;
- season/week.

If exact target counts are unavailable or materially incomplete, `RECEIVING_CAPACITY_LOST` is source-ineligible rather than reconstructed from receiving yards.

If exact rushing attempts/carries are unavailable or materially incomplete, `BACKFIELD_RUN_CAPACITY_LOST` is source-ineligible rather than reconstructed from rushing yards.

### Optional role-normalization source
PFR/nflverse snap counts may be audited only as a coverage/provenance diagnostic. They are not required for V1 eligibility and may not replace actual carry/target opportunity.

## Frozen strict-prior construction feasibility

For each unavailable player on target `(season, week)`:

- use only player usage rows strictly before that target week;
- allow previous-season history for Week 1;
- use at most the most recent 8 eligible games for feasibility reporting;
- report prior-game count, prior carries/game and prior targets/game;
- do not use target-game participation, carries, targets, receptions, yards, snaps, or outcomes.

For team-week feasibility reporting only, aggregate unavailable-player prior opportunity into the fixed channels:

- `backfield_run_capacity_lost_raw = sum(prior carries/game)` for unavailable RB/FB/HB;
- `receiving_capacity_lost_raw = sum(prior targets/game)` for unavailable WR/TE/RB/FB;
- `wr_te_receiving_capacity_lost_raw = sum(prior targets/game)` for unavailable WR/TE;
- `rb_fb_receiving_capacity_lost_raw = sum(prior targets/game)` for unavailable RB/FB/HB.

These are **source-feasibility quantities only**. They must not be correlated with or scored against any target outcome in V1.

No post-result scaling, normalization, interaction, sign assumption, ratio, difference, or threshold may be selected in this audit.

## Current-season deployability audit

The family may advance only if the same conceptual inputs can be produced prospectively in the current stack:

- current definitive availability/current-role layer provides pre-kickoff unavailable identity;
- nflverse/current player history can provide only completed-game prior carries/targets;
- Week 1 may fall back to prior-season completed-game history;
- no target-game result or sportsbook input is required.

V1 need not wire this into production. It must only document whether the live-source contract exists.

## Frozen outputs

1. `directional_personnel_source_inventory.csv`
   - source/table, season, rows, required fields, chosen field names, coverage.
2. `directional_personnel_identity_audit.csv`
   - target season/week/team, inactive token, position, resolved player id, resolution status.
3. `directional_personnel_prior_usage_audit.csv`
   - target identity + inactive identity + prior game count + prior carries/game + prior targets/game + strict-prior max source season/week.
4. `directional_personnel_teamweek_feasibility.csv`
   - target identity plus the four fixed raw opportunity-loss channels and coverage counts only.
5. `directional_personnel_source_result.json`
   - all gates and final source dispositions.

No target pass-opportunity rate, actual attempts, passing yards, target-game WR opportunity, parent residual, or sportsbook field may appear in any V1 output.

## Frozen source eligibility gates

### BACKFIELD_RUN_CAPACITY_LOST
Eligible only if all pass:
1. corrected M78 snapshot exact SHA and row contract pass;
2. relevant RB/FB/HB inactive identity resolution rate >= `95%`;
3. exact carries/rushing-attempt field exists for 2023-2025;
4. carry field populated/semantically valid for >= `99%` of eligible player-week stat rows;
5. >= `90%` of mapped backfield inactive target events have at least one strictly-prior usage game;
6. >= `75%` have at least three strictly-prior usage games;
7. Week-1 prior-season construction is mechanically supported;
8. no target-game outcome or sportsbook input is required;
9. materially distinct from M77/M79 aggregate skill burden.

### RECEIVING_CAPACITY_LOST
Eligible only if all pass:
1. corrected M78 snapshot exact SHA and row contract pass;
2. relevant WR/TE/RB/FB inactive identity resolution rate >= `95%`;
3. exact target-count field exists for 2023-2025;
4. target field populated/semantically valid for >= `99%` of eligible player-week stat rows;
5. >= `90%` of mapped receiving-skill inactive target events have at least one strictly-prior usage game;
6. >= `75%` have at least three strictly-prior usage games;
7. Week-1 prior-season construction is mechanically supported;
8. no target-game outcome or sportsbook input is required;
9. materially distinct from M77/M79 aggregate skill burden.

### Family advance gate
`DIRECTIONAL_PERSONNEL_CONSEQUENCE_SOURCE_ELIGIBLE` only if both backfield-run and receiving-capacity channels are eligible.

If only one channel passes, the family does not advance because the scientific distinction is the directional imbalance between run and pass capacity.

## Allowed dispositions

- `DIRECTIONAL_PERSONNEL_CONSEQUENCE_SOURCE_ELIGIBLE`
- `BACKFIELD_ONLY_SOURCE_ELIGIBLE_DIRECTIONAL_FAMILY_BLOCKED`
- `RECEIVING_ONLY_SOURCE_ELIGIBLE_DIRECTIONAL_FAMILY_BLOCKED`
- `SOURCE_INELIGIBLE_IDENTITY_OR_USAGE_COVERAGE`
- `SOURCE_INELIGIBLE_SCHEMA`
- `MECHANICAL_SOURCE_AUDIT_FAIL`

## Stopping rule

- V1 is source/provenance only.
- Do not read or join the parent pass-opportunity-rate residual.
- Do not inspect QB attempt or passing-yard improvement.
- Do not inspect WR residual correlation.
- Do not fit a model.
- Do not choose weights/windows other than the fixed recent-8 feasibility window.
- Do not alter eligibility thresholds after results are visible.
- If and only if the directional family passes, freeze a separate 2024-development / untouched-2025 predictive plan before opening any target relationship.
- If it fails, preserve the result and move to another genuinely independent pass-intent source rather than loosening this contract.
