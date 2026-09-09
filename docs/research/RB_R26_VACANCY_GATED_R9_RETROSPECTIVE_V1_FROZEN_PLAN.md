# RB R26 — Vacancy-Gated R9 Retrospective V1

Status: **FROZEN BEFORE CANDIDATE EXECUTION**
Date: 2026-09-09
Production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Research branch: `research-rb-r26-role-transition-entitlement-v1`

## Scientific label

This is a **predeclared retrospective mechanism test**, not fresh historical confirmation. All modern historical RB-receiving scoring blocks have already been exposed somewhere in the R8-R25 lineage. A positive R26 result may justify a separately locked 2026 prospective shadow, but may not by itself authorize production promotion.

## Why this candidate exists

The R26 transition-error atlas found a materially sharper state than generic room turnover:

- changed-room baseline receptions MAE was about 6.9% worse than stable-room MAE;
- for same-team incumbents, one or more room exits enriched baseline receptions MAE by about 16.9%;
- incumbent reception bias after exits was about -0.63 receptions/game versus about -0.35 without exits;
- generic entrant presence produced only about 0.9% baseline MAE enrichment;
- both RB1 and RB2+ incumbents showed the exit/vacancy problem;
- Week 1 was materially harder and systematically underprojected;
- broad R23/R25 history redistribution was not a safe Week-1 fix.

The R9 identity family is the existing scientifically supported RB receiving-identity mechanism. Its feature set contains player-wide prior receiving identity plus separate same-team history, so veteran receiving identity can port across teams rather than collapsing to zero when a player changes clubs.

## Frozen hypothesis

The current production RB receiving baseline is generally adequate when the active RB room has not lost a member, but is too slow to redistribute receiving opportunity after a competitor exits.

Therefore:

> Apply R9 receiving-identity redistribution only inside RB rooms with leakage-safe vacated competition; leave all other rooms on the production baseline.

## Frozen state definition

For target season/week, construct the active RB/FB room from the canonical weekly roster contract:

- allowed target-week roster statuses: `ACT`, `INA` only;
- current room comes from target-week roster snapshot;
- prior room comes from the strictly earlier roster snapshot;
- Week 1 prior room = final regular-season snapshot of the prior season;
- Weeks 2+ prior room = latest earlier regular-season snapshot in the same season;
- no target-game participation;
- no target-game box score;
- no sportsbook input;
- no same-week historical depth chart required.

`room_exits_n` = count of RB/FB player identities present in the prior room but absent from the current `ACT/INA` room.

Vacancy gate:

`VACANCY_ACTIVE = room_exits_n >= 1`

No threshold search is allowed.

## Frozen candidate

### Baseline

Current production finite RB receiving target entitlement before R22.

### Stable / no-vacancy rooms

If `VACANCY_ACTIVE == False`:

- candidate RB target entitlement = production baseline exactly.

### Vacancy rooms

If `VACANCY_ACTIVE == True`:

1. Preserve the production RB-room target mass exactly.
2. Attach the existing R9/R8 receiving-identity feature family using only strict-prior data.
3. Fit the R8 Ridge residual identity model on training data only.
4. Fit the R9 reliability multiplier from rolling-origin OOF predictions on training data only, clipped to `[0,1]` exactly as R9.
5. Compute R9 score:
   - `log(production_within_rb_share + EPS)`
   - plus `reliability * raw_R8_residual`.
6. Softmax only within the current RB/FB room.
7. Rescale to the exact production RB-room target mass.
8. Do not alter any non-RB entitlement.

This candidate intentionally does **not** add a new coefficient for the number of exits and does not transfer a hand-chosen percentage of vacated share. The only gate is vacancy yes/no; redistribution strength remains the existing training-derived R9 reliability mechanism.

## New-to-team veterans and rookies

No player is suppressed merely for being new to the team.

R9 player-wide prior identity may carry across clubs. Same-team history remains a separate feature. A player with no prior NFL receiving history naturally has little persistent identity evidence, but is not assigned an arbitrary rookie penalty.

## Current hierarchy / depth-chart source

Timestamped 2025+ nflverse/ESPN depth hierarchy (`dt`, `pos_slot`, `pos_rank`) is **not included in this V1 candidate**.

Reason: hierarchy is a distinct current-state mechanism and should be source-audited/tested separately rather than mixed into the first vacancy test. Ourlads remains separately valuable for LWR/RWR/SWR/alignment and matchup plumbing.

A positive hierarchy audit can support a later R26 hierarchy-state candidate or router.

## Retrospective evaluation block

Evaluate modern seasons **2020-2025** with strict-prior fitting/state construction.

These seasons are not scientifically fresh. They are chosen to test whether the mechanism is consistent enough to justify a prospective 2026 Week-1 shadow lock.

Primary populations:

- ALL RB/FB
- VACANCY_ACTIVE rows
- same-team incumbents in vacancy rooms
- RB1 incumbents in vacancy rooms
- RB2+ incumbents in vacancy rooms
- new-to-team veterans in vacancy rooms
- no-prior-NFL-history backs in vacancy rooms
- Week 1
- Weeks 2+

No cohort may be removed after results are inspected.

## Frozen metrics

Targets and receptions separately:

- MAE
- RMSE
- bias
- Pearson
- Spearman
- median absolute error
- p75 absolute error
- p90 absolute error

Structural:

- RB-room target-mass conservation gap
- team entitlement conservation gap
- non-RB entitlement max delta
- receiving-yard point-mean delta
- R22 authority delta
- sportsbook input count
- future-outcome count

## Frozen retrospective support gates

These gates determine only whether R26 is strong enough to justify a **prospective 2026 shadow lock**. They do not authorize production promotion.

All must pass:

### Integrity

1. sportsbook inputs upstream = `0`;
2. target/future outcomes in features = `0`;
3. strict-prior state/fitting contract holds;
4. max RB-room target-mass gap `< 1e-10`;
5. max non-RB entitlement delta `< 1e-12`;
6. receiving-yard production mean unchanged exactly;
7. R22 assets/logic unchanged;
8. protected production files unchanged.

### Vacancy mechanism support

9. pooled vacancy-room incumbent **receptions MAE improves**;
10. pooled vacancy-room incumbent receptions RMSE is non-worse;
11. absolute vacancy-room incumbent reception bias improves or is non-worse;
12. vacancy-room incumbent receptions p90 may not worsen by more than `2%`;
13. vacancy-room incumbent target MAE improves;
14. at least **4 of 6** seasons improve vacancy-room incumbent receptions MAE;
15. no single season worsens vacancy-room incumbent receptions MAE by more than `2%`;
16. neither RB1 nor RB2+ vacancy-room incumbent receptions MAE may worsen by more than `1%` pooled;
17. at least one of RB1 or RB2+ improves pooled vacancy-room incumbent receptions MAE.

### Global safety

18. ALL-RB pooled receptions MAE may not worsen by more than `0.25%`;
19. ALL-RB pooled receptions RMSE may not worsen by more than `0.25%`;
20. Week-1 pooled receptions MAE may not worsen by more than `0.50%`.

## Disposition

- `RETROSPECTIVE_SUPPORT_FOR_2026_PROSPECTIVE_SHADOW`: all integrity and retrospective support gates pass. This authorizes only a separately locked 2026 Week-1 shadow forecast using the exact mechanism.
- `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`: any gate fails. Preserve the result and do not tune the vacancy threshold, R9 reliability formula, role cohorts, or gates after inspection.

## What this candidate cannot claim

Even on a retrospective PASS it does not solve or certify:

- production RB receptions;
- RB receiving-yard mean;
- R22 tail science;
- current depth hierarchy as a receiving authority;
- WR/TE efficiency;
- shared QB-receiver conservation;
- anytime TD;
- game ML/spread/total.

The only next scientific authority after retrospective support is a **prospective 2026 prediction lock and grade**.
