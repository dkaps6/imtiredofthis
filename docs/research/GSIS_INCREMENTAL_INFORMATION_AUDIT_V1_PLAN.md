# GSIS Incremental Information Audit V1 — Frozen Plan

**Status:** FROZEN BEFORE RESULT  
**Branch:** `research-gsis-point-in-time-archive-v1`  
**Parent gate:** `GSIS_NOVELTY_EXISTING_SOURCE_AUDIT_V1.md`  
**Scope:** research/source-quality only; no predictive outcome testing; no production or Full Slate changes.

## Purpose

Determine whether the two A-class NFLGSIS sources add current pregame role/personnel state that is not already recoverable from the live stack:

1. **Play Time → Lineup Detail**
2. **Play Time → Formation Usage**

This audit must not reopen prior generic pass-tendency, down/distance, field-position, score-state, static snap/depth, receiver-room targets-per-play, NGS-target, team-coverage, or first-down choice-economics research.

## Frozen input boundary

GSIS authority for the first audit is the immutable 2026 REG point-in-time snapshot captured through Week 3 and recorded in Issue #535 / PR #662.

Existing-stack comparators are restricted to information available through the same pregame boundary:

- current 2026 nflverse snap counts;
- Ourlads/current depth and availability;
- PlayerForm / TeamForm;
- existing RB / TE / WR entitlement state;
- existing team pass tendency / PROE and strict-prior PBP-derived state where already materialized.

No Week-3 realized outcomes may be used to score, tune, select, or rescue a feature family.

## Source-quality gates

Fail closed before information analysis if any required gate fails:

- source snapshot identity/hash does not match the recorded authority;
- team/report key is ambiguous;
- player identity mapping is materially incomplete;
- lineup rows cannot be parsed reproducibly;
- Formation Usage cell semantics cannot be reconstructed from the stored rendered-table structure;
- source-empty states are silently imputed;
- a comparator uses information newer than the GSIS snapshot boundary.

Report missingness separately; do not fill unavailable team/report states from another source.

## Lineup Detail audit

### A. Marginal exposure reconciliation

From exact offensive lineups, derive for every identified player:

- total lineup plays containing the player;
- pass plays containing the player;
- rush plays containing the player;
- share of team lineup plays;
- pass/rush exposure share.

Compare these with current offensive snap counts / snap share where the definitions are compatible.

This is a **source reconciliation**, not a predictive test.

### B. Exact-player co-occurrence novelty

For players with meaningful offensive exposure, derive:

- pairwise shared-play count;
- pairwise shared-play share;
- conditional co-occurrence: P(B on field | A on field);
- observed co-occurrence minus the independence expectation implied by the two marginal exposure shares;
- each player's most-common partners;
- number of distinct lineups containing the player;
- lineup concentration / entropy for the team.

The core question is whether exact co-occurrence is recoverable from marginal snap share and depth-chart state. Do not infer routes, coverage assignments, blocking assignments, or target hierarchy from shared presence.

### C. Exact-lineup play-choice state

For lineups meeting a preregistered minimum sample of **10 plays**, retain:

- plays;
- pass plays;
- rush plays;
- pass rate.

Summarize within-team dispersion across qualifying lineups and compare it descriptively with the existing team-level pass tendency / PROE state.

This is not a rerun of generic pass-tendency research. The only permitted question is whether **exact-player combination conditioning** exposes state that team-level tendency does not encode.

### D. Defensive lineup scope

Perform the same co-occurrence / concentration inventory for defensive lineups, but do not authorize a model experiment from this audit. Defensive Lineup Detail cannot be treated as WR-CB assignment, coverage responsibility, rush matchup, or route defense.

## Formation Usage audit

### A. Current personnel deployment

For each team and down/distance cell, preserve:

- #TE;
- #WR;
- play count;
- rush plays;
- pass plays;
- pass rate.

Derive weighted current personnel shares, including 10/11/12/13/20/21/22-style groupings where the displayed #TE/#WR combination permits an unambiguous mapping. Ambiguous groupings remain literal #TE/#WR values.

### B. Increment beyond snaps/depth

Compare the live personnel mix with what the existing stack actually contains.

Do **not** fabricate a personnel mix from individual snap percentages. Instead explicitly mark any field that cannot be reconstructed from current snaps/Ourlads as non-reconstructible.

Measure:

- personnel-group concentration / entropy;
- team share by #TE/#WR grouping;
- down/distance-specific grouping changes;
- within-state pass-rate differences across personnel groups with **>=10 plays**.

### C. Increment beyond team tendency

Where existing TeamForm/PBP state provides a matched generic pass tendency for the same team/state, report the residual difference between personnel-conditioned pass rate and the existing aggregate state.

No coefficient fitting, target-outcome comparison, or threshold search is permitted.

## Lineup Combinations schema-only gate

Inspect schema only.

- If every useful field is directly derivable from Lineup Detail, disposition = `LINEUP_COMBINATIONS_REDUNDANT`.
- If a genuinely new live field exists, document the exact field and freeze a separate acquisition decision before collecting values.

Do not bulk acquire Lineup Combinations during this audit.

## Required outputs

Produce team-level and league-level summaries for:

- identity/source reconciliation;
- co-occurrence non-reconstructibility;
- lineup concentration;
- exact-lineup pass/rush dispersion;
- personnel grouping distribution;
- personnel-conditioned pass/rush dispersion;
- overlap/redundancy versus snaps, Ourlads, PlayerForm/TeamForm and entitlement state;
- missing/empty source states.

## Frozen dispositions

The audit must end in one of these source-level dispositions for each A-class report:

- `SOURCE_QUALITY_FAIL` — source cannot be trusted or reconciled enough for research.
- `MOSTLY_REDUNDANT` — useful fields are effectively recoverable from the existing live stack; archive only.
- `INCREMENTAL_CURRENT_STATE_CONFIRMED` — the source carries current role/personnel structure not recoverable from existing live inputs. This authorizes only a separately preregistered predictive experiment.
- `NEEDS_PROSPECTIVE_HISTORY` — current novelty is structurally real but churn/change information cannot be evaluated from one snapshot; continue immutable weekly archiving before any predictive claim.

These dispositions are **not** model promotions.

## Anti-leakage / anti-reinvention rules

- No Week-3 outcome grading or target-variable use.
- No sportsbook lines or OddsAPI.
- No model fitting or coefficient tuning.
- No production or Full Slate change.
- No use of full-season historical aggregate tables as historical weekly states.
- No generic down/distance, score-state, field-position, quarter, or pass-tendency retest.
- No player box-score rebuild from GSIS.
- No inference of routes, assignments, separation, blocking duties or play calls from co-occurrence.
- No post-result threshold changes.

## Expected next action after result

If `INCREMENTAL_CURRENT_STATE_CONFIRMED`, freeze a separate small predictive experiment that asks whether the new current role/personnel state improves a documented unresolved opportunity/usage problem.

If `NEEDS_PROSPECTIVE_HISTORY`, continue weekly immutable snapshots and test change/churn only after enough distinct point-in-time observations exist.

If `MOSTLY_REDUNDANT`, stop the GSIS modeling lane and retain the source for QA/archive purposes only.
