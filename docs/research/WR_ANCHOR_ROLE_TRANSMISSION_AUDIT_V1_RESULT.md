# WR Anchor / Role-Transmission Audit V1 — Result

Date: 2026-09-25  
Branch: `research-wr-anchor-role-transmission-audit-v1`

## Disposition

**`WR_ANCHOR_ROLE_TRANSMISSION_NO_GAP_CLOSED`**

The preregistered diagnostic did **not** support the hypothesis that strict-prior receiver participation identifies a truer current WR leader that is failing to propagate through the existing M38 WR1 / WR-R15 hierarchy.

No hierarchy candidate is authorized from this diagnostic.

## Authoritative execution

- run: `36203881629`
- job: `108296131500`
- tested head: `13748f88e7c9eedf83b220b3af467ed48f351ec6`
- artifact: `10893281689`
- digest: `sha256:33b12fd521196838054f4ee4240d8cea8f66c2fce5b5bfc9eb902bc67d730adb`
- frozen WR-R15 authority run/artifact: `34238301577 / 10061328722`
- frozen authority source commit used only for identity/label bridge: `02c3dd1a681d4ab2953683039e39830554f9ec9f`

Scientific contract remained unchanged:
- candidate variants scored: **0**
- parameters fit: **0**
- sportsbook inputs used: **0**
- production mutations: **0**

## Mechanical authority bridge

All final bridge gates passed:

- frozen WR2+ rows bridged: **5,321**
- frozen team-game anchors bridged: **1,088**
- scored WR1 identity checks: **1,026**
- WR1 identity mismatches: **0**
- secondary identity mismatches: **0**
- secondary event-id mismatches: **0**
- secondary prior-count max gap: **0**
- secondary prior-availability mismatches: **0**
- secondary snap-pct max gap: **0**
- secondary snap-count max gap: **0**
- scored actual-target max gap: **0**
- strict-prior future violations: **0**

Frozen M38 baseline entitlements and WR-R15 candidate entitlements remained sourced only from the immutable authority artifact. Fresh historical replay was used only after parity for stable identity equivalence, anchor participation and actual-target labels.

## Pooled diagnostic

Eligible team-games: **1,087**

M38-anchor / participation-leader mismatch:
- team-games: **530**
- mismatch rate: **48.7580%**

On those mismatch games:
- participation-leader actual-top hit advantage vs M38 anchor: **-25.4717 percentage points**
- participation leader minus M38 anchor mean actual targets: **-1.4981**
- WR-R15 final leader follows participation leader: **1.3208%**
- WR-R15 candidate target AE per scored WR, mismatch cohort: **2.045053**
- WR-R15 candidate target AE per scored WR, match cohort: **2.042382**

Across all eligible team-games:
- M38 anchor actual-top hit rate: **63.2015%**
- participation leader actual-top hit rate: **50.7820%**
- WR-R15 candidate leader actual-top hit rate: **62.9255%**
- baseline target AE per scored WR: **2.128581**
- WR-R15 candidate target AE per scored WR: **2.043687**

## Season stability

### 2023
- eligible team-games: **543**
- mismatch games: **249** (**45.8564%**)
- participation actual-top hit advantage vs anchor: **-30.5221 pp**
- participation minus anchor actual targets: **-1.5783**
- WR-R15 follows participation: **2.0080%**

### 2024
- eligible team-games: **544**
- mismatch games: **281** (**51.6544%**)
- participation actual-top hit advantage vs anchor: **-20.9964 pp**
- participation minus anchor actual targets: **-1.4270**
- WR-R15 follows participation: **0.7117%**

The adverse participation-leader result is directionally consistent in both seasons.

## Frozen criteria

| Criterion | Result |
|---|---|
| mismatch_team_games_ge150 | PASS |
| pooled_participation_top_hit_advantage_ge5pp | **FAIL** |
| participation_top_hit_advantage_nonnegative_both_seasons | **FAIL** |
| participation_actual_target_advantage_positive_pooled_nonnegative_both | **FAIL** |
| wr_r15_follows_participation_lt50pct | PASS |
| mismatch_final_target_error_ge5pct_worse_than_match | **FAIL** |
| strict_prior_future_violations_zero | PASS |
| sportsbook_inputs_zero | PASS |
| candidate_variants_scored_zero | PASS |

## Interpretation

There is a real identity mismatch between the M38 anchor and the most recent strict-prior participation leader in roughly half of team-games, and WR-R15 usually preserves the M38 anchor rather than following the participation leader.

But that is **not a defect supported by outcomes**. When the two leaders disagree, the participation leader is much less likely to become the actual top-target WR and averages about 1.5 fewer targets than the M38 anchor. The mismatch cohort also does not show the preregistered target-error degradation required to justify a hierarchy intervention.

Therefore the current evidence says:
- strict-prior participation alone is **not** a superior WR1 identity signal;
- M38/WR-R15 is not shown to be missing a simple current-role transmission;
- no alternate participation threshold/window/blend rescue is authorized from this exposed cohort;
- no WR1/hierarchy production change is authorized.

## Scientific closure / next frontier

This closes the simple WR anchor/current-participation transmission hypothesis.

Combined with the failed Receiver Room Targets-Per-Play V1 confirmation and failed Offensive Regime Boundary hard-reset test, the project should not keep transforming the same WR-room/history information.

Next work should move to genuinely different pregame information or architecture. The already-frozen prospective **RB Vacancy Opportunity V1** lane is the cleanest sanctioned continuation: teammate definitive-unavailability -> vacated rushing opportunity -> strictly-prior successor entitlement, with YPC frozen and no retrospective M96 reopening.
