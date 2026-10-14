# Player Opportunity Difficulty -> Target Allocation Source Audit V1

Date: 2026-10-14

Status: **FROZEN SOURCE/LINEAGE AUDIT BEFORE INTEGRATION DESIGN**

## Purpose

The confirmed WR/TE player-specific opportunity uncertainty signal is measured in receiving-yards-equivalent opportunity error:

`opportunity_error = (pred_targets - actual_targets) * pred_ypt`.

Before integrating that signal into the real target-allocation distribution, verify that the exact promoted authority bytes expose enough **pregame** lineage to recover:
- exact promoted player target mean;
- exact promoted player target share or an equivalent share;
- exact pregame team target total implied by that share;
- stable player/team/week identity.

Do not use realized target-game team volume to manufacture a pregame allocation probability.

## Exact authorities

WR:
- WR-R15/M38 authority run `34238301577`
- artifact `10061328722`
- file `wr_r15_confirmation_predictions.csv`
- candidate variant `WR_R15_WR1_ANCHORED_PARTICIPATION`

TE:
- TE-R5P production-contract authority run `34152797603`
- artifact `10029942404`
- file `te_r5p_oos_player_casebook.csv`

## Audit

WR:
1. require promoted `pred_targets`, `entitlement_tgt_share`, season/week/team/player identity;
2. infer `pred_team_targets = pred_targets / entitlement_tgt_share` where share > 0;
3. require one internally consistent implied team target total per team-game within numerical tolerance.

TE:
1. require promoted `candidate_targets_r5p`;
2. enumerate pregame target/share/pool fields in the exact casebook;
3. prefer an explicit candidate share + target mean ratio if present;
4. otherwise test whether baseline target/share lineage in the same frozen row can recover the common pregame team target total without target-game outcomes;
5. if not possible from frozen casebook bytes, disposition TE as needing production-order replay reconstruction. Do **not** substitute realized team targets.

## Dispositions

Overall source audit:
- `PLAYER_OPPORTUNITY_DIFFICULTY_TARGET_ALLOCATION_SOURCE_READY` if WR and TE both expose/recover exact pregame team-total lineage;
- `PLAYER_OPPORTUNITY_DIFFICULTY_TARGET_ALLOCATION_REPLAY_REQUIRED` if at least one position needs existing production-order replay reconstruction;
- `PLAYER_OPPORTUNITY_DIFFICULTY_TARGET_ALLOCATION_SOURCE_FAILURE` for identity/integrity failure.

This audit scores no forecast and changes no model.

No sportsbook. No OddsAPI. No 2026 outcomes. No production mutation.
