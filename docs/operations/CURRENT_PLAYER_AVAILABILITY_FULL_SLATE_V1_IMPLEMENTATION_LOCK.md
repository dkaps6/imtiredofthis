# Current Player Availability — Full Slate V1 Implementation Lock

Status: `RELOCKED AFTER VALUE_NEUTRAL_WEEK1_TEAMFORM_MECHANICAL_REPAIR`

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
- `scripts/providers/nfl_official_inactives_v1.py` = `0d67316b6d7b9aa9d3637a07da4cd2b171639e`
- `scripts/validate_current_player_availability_timing_v1.py` = `d67ae30ed6e837f62098671c499d05462fe9d837`
- `scripts/build/build_reconciled_active_roles_v1.py` = `b2d05882cada048337f1f5f8b8db8ec7f9eef001`

Explicit current-role / timing-withholding seams remain unchanged:
- `scripts/utils/current_roles_v1.py` = `7540cc40e02546b4feb2cc503aa506aa613c44f3`
- `scripts/run_player_form_current_roles_v1.py` = `6252142859042dac7411ba23a482d8c0c128c601`
- `scripts/run_rb_week1_current_roles_v1.py` = `5874128aaa31dca9ed0a401f4dfa9197ff4c282e`
- `scripts/build/build_production_eligible_active_roles_v1.py` = `8be1e607f4d35359459aa2d888c4cfee12cd27c7`
- `scripts/run_current_player_availability_candidate_prep_v1.py` = `1b16c8052f079ba94423b21a755ce4107f60a311`

Value-neutral mechanical repair after first run:
- preserved failure record: `docs/operations/CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_RUN1_QB_CONTEXT_MECHANICAL_FAILURE.md`
- first locked run `34439714153`, job `102752015236`, artifact `10137497418`, digest `sha256:0f3f5195fce8bc6514804b411e0b312475a379cfeb722eb5f5280237f8ff50f2`
- exact failure: `run_team_form_context.py` rejected absence of legal current-season `week < 1` PBP before any availability/opportunity execution
- repaired Week-1-only wrapper `scripts/run_team_form_context_week1_prior_v1.py` = `240f62d018fa419567663cae3c96647483305361`
- repaired candidate workflow `.github/workflows/ops-current-player-availability-full-slate-v1.yml` = `c432d6985323e1a5f6029464ca401dbd68393b78`
- repair semantics: target Week 1 only, force the already-declared PRIOR_SEASON PBP source; wrapper refuses to run outside Week 1; no TeamForm formulas, model parameters, T-75 logic, availability hierarchy, role semantics, R22/R26 behavior, sportsbook boundary, or frozen 35 gates changed.

Pre-lock/current-role seam evidence remains:
- Run `34439533260`, Job `102751487393`: SUCCESS.
- Current-role resolver fixtures + timing-withheld game fixtures PASS.
- No scientific model files changed from the frozen integration-plan parent.

## Locked candidate contract

Each candidate execution under this repaired lock must:

1. run with `FETCH_LIVE_ODDS=false`;
2. preserve raw `data/roles_ourlads.csv` as provider evidence;
3. build current availability and T-75 certification before PlayerForm/current opportunity;
4. materialize `data/roles_current_production_eligible_v1.csv` and expose it only through explicit `ACTIVE_ROLES_CSV` plumbing;
5. exclude definitive-unavailable players and timing-withheld games before PlayerForm and RB P3;
6. retain QUESTIONABLE/DOUBTFUL unless stronger definitive evidence applies;
7. keep all protected M89/M90/C2, M38, WR-R15, TE-R5P, P3, R26 and R22 scientific parameters/artifacts unchanged;
8. use zero sportsbook inputs to availability/role/opportunity;
9. preserve all historical research records;
10. upload the complete candidate artifact set whether the run succeeds or fails.

Mechanical failures may receive only value-neutral repair and must be preserved separately. A scientifically valid first 35-gate result may never be retuned, rerouted or partially promoted after inspection.

## Promotion boundary

This lock does **not** authorize production. Promotion remains contingent on all 35 gates in the frozen integration plan and a recorded disposition of exactly:

`CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION`

Anything else leaves protected production unchanged.
