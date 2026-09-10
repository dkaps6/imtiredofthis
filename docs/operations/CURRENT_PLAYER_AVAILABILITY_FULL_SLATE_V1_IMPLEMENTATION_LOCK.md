# Current Player Availability — Full Slate V1 Implementation Lock

Status: `LOCKED BEFORE FIRST CANDIDATE FULL_SLATE RESULT`

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

New explicit current-role / timing-withholding seams:
- `scripts/utils/current_roles_v1.py` = `7540cc40e02546b4feb2cc503aa506aa613c44f3`
- `scripts/run_player_form_current_roles_v1.py` = `6252142859042dac7411ba23a482d8c0c128c601`
- `scripts/run_rb_week1_current_roles_v1.py` = `5874128aaa31dca9ed0a401f4dfa9197ff4c282e`
- `scripts/build/build_production_eligible_active_roles_v1.py` = `8be1e607f4d35359459aa2d888c4cfee12cd27c7`
- `scripts/run_current_player_availability_candidate_prep_v1.py` = `1b16c8052f079ba94423b21a755ce4107f60a311`
- `.github/workflows/ops-current-player-availability-full-slate-v1.yml` = `1ea734ab5dd9e5f9846857a449d8c8c948774bf5`

Pre-lock mechanical seam evidence:
- Run `34439533260`, Job `102751487393`: SUCCESS.
- Current-role resolver fixtures + timing-withheld game fixtures PASS.
- No scientific model files changed from the frozen integration-plan parent.
- This smoke is plumbing evidence only and is not the first Full Slate integration result.

## Locked first-run contract

The first workflow execution caused by this lock is the immutable first candidate Full Slate integration run. It must:

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

A mechanical failure may receive only value-neutral repair and must be preserved separately. A scientifically valid first 35-gate result may never be retuned, rerouted or partially promoted after inspection.

## Promotion boundary

This lock does **not** authorize production. Promotion remains contingent on all 35 gates in the frozen integration plan and a recorded disposition of exactly:

`CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION`

Anything else leaves protected production unchanged.
