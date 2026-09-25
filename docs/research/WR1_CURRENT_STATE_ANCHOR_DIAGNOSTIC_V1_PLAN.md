# WR1 Current-State Anchor Diagnostic V1 — Frozen Plan

Date: 2026-09-25

Status: **FROZEN BEFORE DIAGNOSTIC COMPUTE**

This is diagnostic-only. It does not qualify a production change.

## 1. Why this diagnostic exists

Three independent findings now intersect:

1. `CURRENT_SEASON_STATE_PERSISTENCE_V1` showed WR target-share state updates
   quickly and improved next-game MAE in every evaluated season 2022-2025.
2. `WR_R15_PRODUCTION_MODEL_V1` deliberately preserves the highest-entitlement
   M38 WR1 as an immutable anchor and only redistributes WR2+.
3. `CURRENT_STACK_RECEIVER_COMPENSATION_AUDIT_V1` showed the valid
   targetable-dropback team-volume correction fails primarily at the top of the
   WR hierarchy:
   - baseline WR1 target bias already negative;
   - WR1 within-room share bias negative;
   - WR2+ within-room share bias slightly positive;
   - targetable volume materially improves WR2+ target MAE but worsens WR1.

This raises a distinct question:

**Is the immutable M38 WR1 anchor stale relative to strictly prior current-season
WR role state?**

This is not an M38 multiplier retune.

## 2. Anti-retest boundaries

Do not:
- retune the M38 WR1 multiplier;
- reopen generic M36-M38 hierarchy multiplier searches;
- reopen WR-R15 WR2+ coefficients;
- change WR room mass;
- use a player-name carveout;
- use targetable-dropback outcomes to choose a threshold;
- use sportsbook inputs;
- use 2024-2025 to select a rule in this diagnostic.

WR-R15 remains closed/canonical inside WR2+.

## 3. Discovery frame

Discovery seasons only:
- 2022
- 2023

2024 and 2025 are not inspected by this diagnostic and remain reserved for a
separately frozen confirmation if a candidate is later justified.

For target week W >= 2:
- prior-season state uses season S-1 only;
- current-season state uses season S games with week < W only;
- target-week/future rows are forbidden.

Stable player identity is required.

## 4. Current-state authority

Use the already-validated `CURRENT_SEASON_STATE_PERSISTENCE_V1` construction.

For each WR:
- prior target share = prior-season cumulative player targets / cumulative team targets;
- current target share = completed-current-season cumulative player targets /
  cumulative team targets;
- `w_current = current_games / (current_games + 4)`;
- blend-4 target share =
  `(1-w_current)*prior_share + w_current*current_share`.

No new window or shrinkage constant is introduced.

## 5. M38 WR1 authority

Reconstruct the historical pregame current-stack baseline through the existing
historical context and explicit target-entitlement path.

For each team-game:
- identify WR rows;
- identify the M38 WR1 anchor as the highest explicit WR entitlement before any
  WR2+ redistribution;
- record anchor team target share `p_m38`;
- record total WR room mass `P_WR`.

No WR-R15 adjustment is needed to define the anchor because production freezes
that anchor exactly.

## 6. Primary diagnostic quantities

For each M38 WR1 with valid prior + current state:

### Team-share view
- M38 anchor share: `p_m38`
- blend-4 WR1 share: `p_blend`
- actual next-game WR1 team target share: `p_actual`
- state gap: `p_blend - p_m38`
- needed correction: `p_actual - p_m38`

Report:
- M38 MAE/bias/correlation;
- blend-4 MAE/bias/correlation;
- current-only and prior-only descriptive MAE;
- Spearman/Pearson between state gap and needed correction;
- sign agreement where both gaps are nonzero.

### Conserved WR-room view
For all modeled WRs with valid blend-4 state on the team:
- normalize blend-4 shares inside the WR room;
- `state_wr1_room_share = blend_WR1 / sum(blend_WR)`;
- M38 WR1 room share =
  `p_m38 / P_WR`;
- actual WR1 room share =
  `actual_WR1_targets / actual_WR_room_targets`.

Report:
- M38 room-share MAE/bias/correlation;
- state-normalized room-share MAE/bias/correlation;
- state-vs-M38 room-share delta vs actual needed room-share correction;
- sign agreement.

This is descriptive only. The diagnostic does not install the state share.

## 7. Targetable-volume interaction diagnostic

The parent team-volume science is preserved but not re-qualified here.

For each discovery WR1:
- baseline predicted targets = projected dropbacks * `p_m38`;
- targetable predicted targets =
  projected dropbacks * strict-prior `R_T` * `p_m38`;
- actual targets attached only after pregame state freezes.

Classify whether targetable volume makes the WR1 absolute target error better or
worse.

Then report state-gap distributions separately for:
- targetable helps WR1;
- targetable hurts WR1.

Question:
**When uniform team-volume thinning hurts WR1, does strictly prior state usually
say the WR1 anchor should have moved upward before thinning?**

No threshold or routing rule is selected.

## 8. Required outputs

Row-level:
- season/week/team/event/player/identity;
- M38 WR1 entitlement;
- total WR room entitlement;
- prior/current/blend target share;
- prior/current game counts;
- actual player targets/team targets;
- actual WR-room targets;
- M38 and state-normalized room shares;
- state gaps and needed corrections;
- strict-prior targetable rate;
- baseline/targetable predicted WR1 targets;
- baseline/targetable target absolute errors;
- targetable helps/hurts flag.

Summaries:
- 2022
- 2023
- pooled discovery
- early current-games buckets: 1, 2, 3, 4, 5-8, 9+
- targetable-helps vs targetable-hurts cohorts.

## 9. Integrity gates

Stop interpretation unless all pass:

1. target-week/future player state usage = 0;
2. stable player identity only;
3. M38 WR1 anchor reconstructed from pregame entitlement only;
4. WR room mass untouched;
5. current-state formula exactly matches the validated blend-4 construction;
6. targetable rate strict-prior;
7. no sportsbook fields used;
8. parameters fit = 0;
9. candidate variants scored = 0;
10. production changed = false.

## 10. Interpretation rule

This audit can justify a later candidate only if all of the following are true
on pooled 2022-2023 discovery:

1. blend-4 WR1 team-share MAE is lower than M38 anchor MAE;
2. state gap vs needed correction Spearman is positive;
3. state-gap sign agreement exceeds 50%;
4. state-normalized WR1 room-share MAE is lower than M38 WR1 room-share MAE;
5. the targetable-hurts cohort shows the same-direction state signal rather
   than a contradictory one.

If these do not hold, close the WR1-state lane.

If they do hold:
- freeze exactly one WR1 anchor state-update candidate;
- preserve total WR-room mass;
- preserve WR-R15 WR2+ relative allocator;
- choose the candidate rule using discovery evidence only;
- then score 2024-2025 once under a separate frozen confirmation;
- any historical pass still requires prospective 2026 confirmation before
  production.

No production change is authorized by this plan.
