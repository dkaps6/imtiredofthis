# RB R27D0 — YAC-Quality Novelty and Source Audit Plan

Status: `SOURCE / NOVELTY AUDIT ONLY / NO MODEL FIT / NO CANDIDATE / NO SCIENTIFIC PERFORMANCE RESULT`

## Parent evidence

- R27C2 result commit: `2cfca82d919290ac18ed01559994d2e0239b0798`
- R27C2 valid run: `34431286455`
- R27C2 job: `102727137722`
- R27C2 artifact: `10134581843`
- R27C2 digest: `sha256:6e3aa7ec2149f6f0b65adb59b4cc36c4f88b24a142234bc368131de8e4de60dd`
- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`

## Why this audit exists

R27C2 identified a physical hypothesis: the 2023 vacancy-RB1 receiving-yard failure is concentrated in post-catch value / YPR compression, not catch-rate, target-depth, screen-frequency or explosive-frequency collapse.

R27B V2 already used strict-prior raw YAC averages at player, team and opponent level. Repeating raw historical YAC or generic YPR would therefore reinvent failed science.

The next candidate, if any, needs genuinely different pregame information that can distinguish:

1. expected YAC created by target/play context; and
2. YAC produced above/below that expectation by the receiver/offense/defense.

This audit checks whether those sources exist with enough historical, weekly, RB-specific coverage to justify a separately frozen predictive experiment.

## Existing-repo novelty boundary

Repository search before this audit found:
- `avg_yac_above_expectation` exists in `scripts/backtest/audit_qb_personnel_tracking_matchup.py`, a prior QB personnel/tracking diagnostic.
- No existing RB receiving-yard mean study using `avg_yac_above_expectation` was found.
- No existing repository use of nflverse PBP `xyac_mean_yardage` was found.

The old QB audit is design/source precedent only. Its QB models/results are not RB evidence and must not be imported as an RB candidate.

## Sources to audit

### A. nflverse play-by-play expected-YAC fields

For REG seasons 2020–2025, inspect completed RB/FB/HB/TB targets for:
- `yards_after_catch`
- `xyac_mean_yardage`
- `xyac_median_yardage`
- `xyac_success`
- `xyac_fd`
- any equivalent xYAC fields exposed by the canonical PBP source

If `xyac_mean_yardage` is sufficiently populated, define source-only diagnostic quantity:

`pbp_yac_over_expected = yards_after_catch - xyac_mean_yardage`

This is NOT yet an eligible predictive feature; the audit only checks source coverage and strict-prior reconstructability.

### B. NFL Next Gen Stats weekly receiving

Using `nflreadpy.load_nextgen_stats(..., stat_type="receiving")`, audit RB-specific weekly availability for:
- targets
- receptions
- `avg_yac_above_expectation`
- `avg_expected_yac`
- `avg_separation`
- `avg_cushion`
- `avg_intended_air_yards` or equivalent
- team, player identity, week, season and position/position bridge

The key question is whether historical RB observations are dense enough for strict-prior player/team features rather than only high-volume WR/TE qualifiers.

### C. More granular PBP target-context proxies

Audit whether RB target rows contain usable weekly fields for potential strict-prior target-type/context histories beyond V2's coarse air-yards/screen averages:
- down
- yards to go
- shotgun
- no-huddle
- pass location
- pass length
- score differential or equivalent game-state field
- air-yards bins, derivable from air_yards

These are source possibilities only. No feature set is authorized by this audit.

## Coverage questions

Report by season 2020–2025:
- RB target rows
- RB completed targets
- xYAC mean non-null rate on completed RB catches
- YAC non-null rate
- NGS receiving rows
- RB NGS rows after position resolution
- RB NGS YACOE non-null rate
- unique RB player-weeks with NGS YACOE
- PBP situational-field non-null rates

Using the exact preserved R27B V2 row universe where practical, also report:
- vacancy RB1 targeted-game count
- percentage with at least 1 strict-prior PBP YACOE observation
- percentage with at least 3 strict-prior PBP YACOE observations
- percentage with at least 1 strict-prior NGS RB YACOE observation
- percentage with at least 3 strict-prior NGS RB YACOE observations
- corresponding non-2023 and 2023 vacancy-RB1 coverage

No prediction errors may be scored against these sources during this audit.

## Strict-prior reconstructability test

For a target player-game `(season, week)`, any historical source row is legal only when:
- source season < target season; or
- source season == target season and source week < target week.

Target-game NGS/PBP rows are outcomes and must not be used as candidate inputs.

The audit may count legal prior observations and source coverage. It may not fit a model or calculate a candidate receiving-yard prediction.

## Audit dispositions

Allowed terminal labels:
- `R27D0_YAC_QUALITY_SOURCES_SUPPORT_SEPARATELY_FROZEN_PREDICTIVE_STUDY`
- `R27D0_YAC_QUALITY_SOURCES_PARTIAL_SUPPORT_NEEDS_SCOPE_REDUCTION`
- `R27D0_YAC_QUALITY_SOURCES_INSUFFICIENT_NO_PREDICTIVE_STUDY`
- `R27D0_MECHANICAL_OR_SOURCE_FAILURE_NO_DECISION`

No disposition authorizes production.

## Production boundary

- no production mutation
- no R26 mutation
- no R22 mutation
- no sportsbook input
- no model fit
- no candidate projection
- no performance gates

If source support is sufficient, a future R27D predictive plan must be newly written and frozen before any performance scoring. It must explicitly avoid generic raw-YAC/YPR persistence already represented by R27B V2/R23 and must state what genuinely new information is being tested.
