# Current Player Availability — Full Slate V1 Implementation Lock

Status: `RELOCKED AFTER VALUE_NEUTRAL_PLAYER_HISTORY_PUBLICATION_REPAIR`

## Frozen authority

- protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- frozen 35-gate integration plan commit: `91ee6aa3ad3813c7d285f6f3163368205937eb09`
- integration-plan blob: `54eb4629c48062fcaef3153918b2069238584d0a`
- locked availability-core implementation commit: `596f084750a8c8e3c36ca09d737dc2edb808dc09`
- canonical T-75 first valid result: run `34437715931`, job `102746163583`, head `092fee088f3402d6313bc02b2f8cc05d1f3f54f9`

## Exact candidate implementation blobs

Existing locked availability core remains unchanged:
- `scripts/providers/ourlads_depth_status_v1.py` = `c115816ea8aa4ba7150a635c3115546d43f94b3c`
- `scripts/build/build_current_player_availability_v1.py` = `9a1b0a672db7854ff52764ad93e5fcd89f9cb0ea`
- `scripts/providers/nfl_official_inactives_v1.py` = `0d67316b6d7b7b9aa9d3637a07da4cd2b171639e`
- `scripts/validate_current_player_availability_timing_v1.py` = `d67ae30ed6e837f62098671c499d05462fe9d837`
- `scripts/build/build_reconciled_active_roles_v1.py` = `b2d05882cada048337f1f5f8b8db8ec7f9eef001`

Explicit current-role / timing-withholding seams:
- `scripts/utils/current_roles_v1.py` = `7540cc40e02546b4feb2cc503aa506aa613c44f3`
- `scripts/run_player_form_current_roles_v1.py` = `0a82649801794c560ccfd0ef368b7a340ec6f38f`
- `scripts/run_rb_week1_current_roles_v1.py` = `5874128aaa31dca9ed0a401f4dfa9197ff4c282e`
- `scripts/build/build_production_eligible_active_roles_v1.py` = `8be1e607f4d35359459aa2d888c4cfee12cd27c7`
- `scripts/run_current_player_availability_candidate_prep_v1.py` = `1b16c8052f079ba94423b21a755ce4107f60a311`

Week-1 TeamForm value-neutral repair remains:
- `scripts/run_team_form_context_week1_prior_v1.py` = `240f62d018fa419567663cae3c96647483305361`

New strict-prior publication regression:
- `tests/test_player_form_strict_prior_publication_v1.py` = `ee3c88060a3d12b10ada01a0eff6e2e2c5b5d1d5`
- candidate workflow `.github/workflows/ops-current-player-availability-full-slate-v1.yml` = `51362ee87a164824cc96dc186aeb9f1f66973f4a`

## Preserved mechanical failures

### Run 1 — Week-1 TeamForm source selection
- run `34439714153`, job `102752015236`
- artifact `10137497418`
- digest `sha256:0f3f5195fce8bc6514804b411e0b312475a379cfeb722eb5f5280237f8ff50f2`
- record: `docs/operations/CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_RUN1_QB_CONTEXT_MECHANICAL_FAILURE.md`
- failure occurred before availability execution; no integration result.

### Run 2 — strict-prior PlayerForm history publication
- run `34443710690`, job `102763847787`
- artifact `10138897760`
- digest `sha256:165c0e431e4771eba05472b17d6680457fedac37d5769bfc5c0b478aad830b1f`
- record: `docs/operations/CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_RUN2_PLAYER_HISTORY_PUBLICATION_MECHANICAL_FAILURE.md`
- availability/T-75/current roles and PlayerForm model-form generation succeeded;
- failure occurred at `run_model_context_bridge.py` because published `player_game_logs.csv` retained same-week 2026 Week-1 NE/SEA rows;
- core PlayerForm blending itself was already strict-prior (`week < target_week`), so this was a publication/provider-contract defect, not model leakage in `player_form.csv` and not a 35-gate result.

## Value-neutral Run-2 repair

The current-role PlayerForm wrapper now republishes `player_game_logs.csv` using exactly the history set the protected PlayerForm model already used:
- all declared prior-season rows; and
- active-season rows only when `week < target_week`.

It recomputes `player_season_totals.csv` from that same strict-prior set and asserts that no active-season same/future-week rows remain. It does not alter PlayerForm formulas or `player_form.csv` values.

The dedicated regression fixture proves same-week, future-week, and unrelated-season rows cannot enter the published target-week history. The candidate workflow also independently asserts the live artifact contains no active-season `week >= target_week` rows before the model bridge executes.

No model formula, scientific parameter, T-75 timing rule, availability hierarchy, role semantics, R22/R26 behavior, sportsbook boundary, or frozen 35 gate has changed.

## Locked candidate contract

Each candidate execution under this repaired lock must:

1. run with `FETCH_LIVE_ODDS=false`;
2. preserve raw `data/roles_ourlads.csv` as provider evidence;
3. build current availability and T-75 certification before PlayerForm/current opportunity;
4. materialize `data/roles_current_production_eligible_v1.csv` and expose it only through explicit `ACTIVE_ROLES_CSV` plumbing;
5. exclude definitive-unavailable players and timing-withheld games before PlayerForm and RB P3;
6. retain QUESTIONABLE/DOUBTFUL unless stronger definitive evidence applies;
7. publish only strict-prior player history at the target-week boundary;
8. keep all protected M89/M90/C2, M38, WR-R15, TE-R5P, P3, R26 and R22 scientific parameters/artifacts unchanged;
9. use zero sportsbook inputs to availability/role/opportunity;
10. preserve all historical research records;
11. upload the complete candidate artifact set whether the run succeeds or fails.

Mechanical failures may receive only value-neutral repair and must be preserved separately. A scientifically valid first 35-gate result may never be retuned, rerouted or partially promoted after inspection.

## Promotion boundary

This lock does **not** authorize production. Promotion remains contingent on all 35 gates in the frozen integration plan and a recorded disposition of exactly:

`CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION`

Anything else leaves protected production unchanged.
