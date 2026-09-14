# WR Post-R18 Receiver Process Source Audit V1 — Result

**STATUS: NOVELTY CLEARED + SOURCE CONTRACT PASSED. R19 MAY BE FROZEN FOR ADVERSARIAL PRE-RESULT REVIEW. NO WR OUTCOME RUN. 2024 HOLDOUT SEALED.**

## Claude novelty audit

Claude independently reviewed the post-R18 proposal against M81, R18, R5/R7 and repo history.

Conclusion:
- receiver-specific `is_catchable_ball` history is **genuinely open** at this aggregation level;
- M81 tested `is_catchable_ball` only inside QB/offense/opponent `THROW_DECISION_QUALITY`, not receiver-specific history;
- R18 CPOE and FTN catchability are materially different constructs: CPOE is outcome-adjusted completion-over-expectation, while catchability is a charted process judgment independent of whether the receiver completed the catch;
- whole-repo search found no prior receiver-specific catchable-ball history experiment;
- expectations should remain conservative because this is adjacent to R18 delivery quality and many free-data WR efficiency lanes have already failed.

Claude identified one blocking source question before any R19 freeze: verify 2022 FTN availability and receiver-level join/population empirically rather than assume it.

## Direct nflverse release verification

The nflverse `ftn_charting` release contains `ftn_charting_2022.parquet` as well as 2023 and 2024 assets. The current nflreadr FTN dictionary also states data are available from 2022 onward.

Receiver identity is not taken from FTN names. The source contract is:

`FTN nflverse_game_id + nflverse_play_id -> nflverse PBP game_id + play_id -> receiver_player_id (GSIS)`

This preserves exact play identity and stable receiver identity.

## Canonical source-only CI audit

Branch:
- `research-wr-yard-efficiency-feature-audit-v1`

Workflow head:
- `13c571af78368238419fe37076fdd3995813f30b`

Run:
- `34907348510`

Artifact:
- ID `10373021925`
- name `wr-receiver-catchability-source-audit-v1`
- digest `sha256:eb0c50110070691cc552027619a430112d60d5a564b61b97a6222f9192c5a924`

The audit loaded **no WR outcomes**, **no WR projection authority artifact**, **no sportsbook inputs**, and explicitly asserted no `wr_r19_receiver_catchability_v1` outcome directory existed.

### Exact source results

| Season | FTN reg rows | PBP reg rows | Weeks 1-18 complete | Exact FTN->PBP play join | Receiver target rows | Catchable populated | Catchable coverage | Unique target receiver IDs |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 2022 | 39,742 | 47,157 | yes | 100.0% | 17,325 | 17,325 | 100.0% | 508 |
| 2023 | 46,025 | 47,399 | yes | 100.0% | 17,558 | 17,558 | 100.0% | 486 |
| 2024 | 45,880 | 47,274 | yes | 100.0% | 17,103 | 17,103 | 100.0% | 494 |

Source contract thresholds were prospectively set in the audit at:
- exact FTN/PBP join >= 95%;
- receiver-target catchability coverage >= 80%;
- regular weeks 1-18 complete.

All seasons passed by a wide margin.

## Scientific implication

The source problem Claude identified is resolved:
- 2022 history exists for early-2023 trailing windows;
- exact target receiver GSIS identity is available through the PBP play join;
- `is_catchable_ball` is fully populated on receiver-targeted pass attempts in all audited seasons;
- the source is historical and in-season deployable.

Therefore receiver-specific catchability now clears both prerequisites for a prospectively frozen candidate:
1. **material novelty** versus tested WR/M81 families;
2. **honest historical + live source contract**.

This does **not** mean the football signal is expected to work. Accumulated negative WR evidence remains important, and R19 must be treated as a narrow information-family test rather than a fresh coin flip.

## Design discipline carried into any R19 plan

A defensible R19 should use:
- one primary receiver signal only: prior receiver target catchable rate;
- fixed positive direction;
- exact WR-R15 authority cohort;
- 2023 development first, untouched 2024 holdout;
- GSIS-first authority identity using the already-audited prior-roster bridge;
- FTN->PBP exact game/play joins for feature-event receiver IDs;
- last-8 prior target-bearing games chosen before filtering/aggregation;
- a predeclared minimum target-event support floor;
- team/offense catchable rate, mean target depth, M38/R15 entitlement and WR rank only as robustness/mediation controls;
- no `is_contested_ball`, `is_created_reception`, `is_drop`, `read_thrown`, or other FTN add-ons;
- no window search, threshold search, WR1-only rescue, tail-only rescue, or post-result change of primary statistic;
- adversarial Claude review of the exact frozen plan before any WR outcome exposure.

## Disposition

`POST_R18_CATCHABILITY_NOVELTY_AND_SOURCE_CLEARED__R19_PLAN_ELIGIBLE`

No production change. No paid Full Slate. No RB work. 2024 remains sealed.
