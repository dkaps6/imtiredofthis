# QB Team Pass Opportunity Source Audit V1 — Frozen Plan

## Purpose

Perform a **source/provenance audit only** for genuinely new pregame information that could later help predict the `TEAM_PASS_OPPORTUNITY` mechanism isolated by `QB_OPPORTUNITY_CHAIN_DECOMPOSITION_V1`.

This migration does not fit a predictive model, does not score target-game QB/team pass-opportunity residuals, and cannot change production.

The exact upstream mechanism established by the parent diagnostic is team pass opportunities/dropbacks. The next step is therefore not another generic QB pass-yards feature hunt. It is to determine whether any untested information family can be constructed safely before kickoff with adequate historical coverage.

## Parent lineage

- Parent branch: `research-qb-opportunity-chain-decomposition-v1`
- Parent result commit: `43bc0a0db245f401ec270bbe48b6d8812315139b`
- Parent canonical run: `34523313743`
- Parent job: `103025958402`
- Parent tested head: `8916d1d4537c6d6f9470ee4d97a94fdde816eb6d`
- Parent artifact: `10170531084`
- Parent digest: `sha256:75cd32198caf7d9cbf193d5769e5762acce53352f2b2626cd92a9b117e75dafd`
- Parent disposition: `TEAM_PASS_OPPORTUNITY_PRIMARY_DIAGNOSTIC`

## Anti-reinvention boundary

The following are already tested/closed and are prohibited from being relabeled as new information:

- M64 possession/dropback generative family: drives, plays per drive, team/opponent offense, team/opponent defense, dropback rate, attempt conversion, neutral/trailing/leading DBR, no-huddle, seconds between plays, scoring-drive rate.
- M65 state-occupancy family: state shares/rates plus the M64 possession family.
- M73 survival family: first-down rate, third-down conversion, early-down success, sack rate, turnover per drive.
- M67-M69 opening tendency / playcaller / game-script families.
- M74 generic attempt/dropback context.
- M75 receiver/personnel tracking.
- M76-M79 personnel discontinuity and official inactives as QB predictive corrections.
- M80-M82 FTN tactical-call, pressure-response, throw-decision, and receiver-error families.
- M83 defensive adaptive gameplan.
- M84/M85 exact receiver responsibility and blocker-rusher families, which remained source-blocked.
- QB-R1 recent player mechanism history plus existing M89 context.
- generic residual-model/model-zoo searches.

Current QB C2 is also not a solution to this mean problem: it preserves the M89/M90 mean and replaces only the selected QB pass-yard distribution in production. Receiver production arrays remain canonical.

## Source families authorized for audit

Only the three families below may be audited in V1.

### Family A — PENALTY_DRIVE_EXTENSION

Candidate source: nflverse/nflfastR play-by-play.

Audit only whether historical play-level fields can support strictly-prior team/offense and opponent-defense summaries such as:

- accepted offensive penalty rate;
- accepted defensive penalty rate;
- defensive first downs by penalty / drive-extension penalty rate where source semantics support it;
- penalty no-play/replay-down indicators where source semantics support them;
- penalty yards, with offense/defense attribution only when the source field is unambiguous.

The audit must distinguish raw penalty flags from accepted/declined/no-play semantics. A field whose semantics cannot be established from the source contract is ineligible rather than guessed.

### Family B — FOURTH_DOWN_AGGRESSION

Candidate source: nflverse/nflfastR play-by-play.

Audit only whether strictly-prior team and opponent-defense summaries can be constructed for:

- fourth-down go-for-it attempt rate where opportunity denominator is source-defensible;
- fourth-down conversion rate;
- fourth-down drive-extension rate;
- fourth-down pass-vs-rush tendency if available without target-game leakage.

This is distinct from old postgame `fourth_down` chaos labels. No target-game fourth-down behavior may become a pregame predictor.

### Family C — SCHEDULE_REST_CONTEXT

Candidate source: nflverse schedules/game metadata.

Audit only deterministic information known before kickoff, including when present:

- team rest days;
- opponent rest days;
- rest differential;
- short-week indicator;
- long-rest/bye indicator;
- day-of-week / Thursday-Monday schedule context;
- home/away status.

Do not use score/result fields, closing lines, moneylines, totals, or any sportsbook-derived game field from schedule tables.

## Quarantined family — OFFICIATING_CREW

Officiating/referee data are **not authorized for predictive use in V1**.

Reason: game-level nflverse officials records represent the official who worked the game and may reflect late substitutions. They are not automatically proven to be archived pre-kickoff assignment records.

V1 may document that an external public pregame assignment source exists, but the family remains `QUARANTINED_NO_HISTORICAL_PREGAME_PROVENANCE` unless a versioned historical archive with publication timestamp before kickoff can be established independently. No actual-game official identity may be treated as a pregame feature by assumption.

## Historical source scope

Audit source availability for regular-season 2023, 2024, and 2025.

Purpose of the span:
- 2023 can supply strict-prior history for 2024 development rows;
- 2024 can supply strict-prior history for 2025 confirmation rows;
- 2025 confirms that the same source schema remains available in the latest historical season.

No 2026 target outcomes are needed or allowed.

## Source-audit outputs

For each family, report:

1. exact source endpoint/package/table;
2. source seasons successfully loaded;
3. row counts by season;
4. required raw fields found/missing by season;
5. field-level non-null/populated coverage by season;
6. explicit semantic notes for each field used;
7. whether source rows can be keyed uniquely to season/week/game/team as appropriate;
8. whether the information can be known or constructed strictly before the target game;
9. whether a Week-1 target can be constructed from prior-season history where required;
10. whether any sportsbook/result/postgame field is required (must be false for eligibility);
11. a no-retest crosswalk against prior QB migrations;
12. family disposition.

For PBP-derived historical-rate families, also construct a **schema-only strict-prior feasibility table** for 2024-2025 target team-games. This table may contain target identifiers and counts of prior eligible source games/rows, but it must not contain target-game pass opportunities, target-game QB attempts, target-game passing yards, or any parent residual/outcome label.

For schedule-rest context, the target-game schedule row itself may be used only for deterministic pregame metadata such as rest/day/home-away. Result/score/market columns are forbidden.

## Frozen source eligibility gates

A family is `SOURCE_ELIGIBLE_FOR_PREDICTIVE_PREREGISTRATION` only if all applicable gates pass:

1. source loads successfully for every required season;
2. required semantic fields exist in every required season;
3. no sportsbook or result field is required;
4. target-game outcome/PBP is not required to construct a target pregame feature;
5. every derived historical-rate feature can be constructed using only rows strictly before the target `(season, week)`;
6. at least 95% of 2024-2025 target team-games have at least one eligible prior source game for the family; Week 1 may use prior-season history;
7. field-level populated coverage is >=95% for core required fields in every required season, except explicitly sparse event indicators whose denominator field is fully covered;
8. team/game keys resolve without ambiguous duplicate team-game records after documented canonicalization;
9. the family is materially distinct from the prohibited no-retest ledger;
10. source semantics are explicit enough to avoid guessing accepted/declined/no-play or opportunity definitions.

If a family fails any gate, it is not eligible for predictive testing in its current form.

## Family dispositions

Each family receives exactly one:

- `SOURCE_ELIGIBLE_FOR_PREDICTIVE_PREREGISTRATION`
- `SOURCE_INELIGIBLE_SCHEMA_OR_COVERAGE`
- `SOURCE_INELIGIBLE_PREGAME_PROVENANCE`
- `DUPLICATE_OR_CLOSED_INFORMATION_FAMILY`
- `MECHANICAL_SOURCE_AUDIT_FAIL`

Officiating remains separately quarantined unless historical pregame provenance is independently proven.

## Stopping rule

- Do not correlate any audited family with target `TEAM_PASS_OPPORTUNITY` residual in this migration.
- Do not inspect QB/pass-yard improvement.
- Do not fit any Ridge, tree, ensemble, or correction model.
- Do not choose transformations/windows based on target outcomes.
- Do not add another information family after source-audit results are visible.
- If one or more families pass source eligibility, freeze a **separate predictive development/confirmation plan** before opening their relationship to the parent residual.
- If all three fail, do not loosen the gates; move to a separate architecture/source-discovery decision.
