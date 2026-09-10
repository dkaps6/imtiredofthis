# Current Player Availability — Production Branch Run2 R22 Disposition Mechanical Repair V1

Status: `FROZEN_BEFORE_RUN2_MECHANICAL_REPAIR_IMPLEMENTATION_OR_RUN3`

## Preserved Run2 evidence

Parent production verification:
- run/job: `34477957076` / `102873279264`
- head: `944426bae1a6d9292fded5dcf41f8951e7357467`
- immutable evidence artifact: `10152500791`
- digest: `sha256:0f24bbf30094aaf8e22133eb70669d7f4e29275faa0d9b29ae48c051e66d6c14`
- conclusion: `FAILURE`
- disposition: `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_MECHANICAL_FAILURE_NO_DECISION`.

Canonical child Full Slate:
- run: `34478009290`
- head: `944426bae1a6d9292fded5dcf41f8951e7357467`
- artifact: `10152451377`
- digest: `sha256:dd7a2cace7a5ec014ad020d9959238b7896e372ad3b9c748c14e58ec077fdeb3`
- conclusion: `SUCCESS`.

The Run1 static-audit compatibility repair therefore succeeded: the exact branch Full Slate completed successfully with live odds disabled and emitted the expected current-availability and promoted-football-stack audits.

## Exact Run2 mechanical failure

The dynamic verifier `scripts/operations/run_current_availability_production_verify_v1.py` rejected the R22 audit solely because its literal accepted-disposition set contained only:
- `RB_R22_PRODUCTION_TAIL_INTEGRATION_PASS`
- `R22_PRODUCTION_TAIL_INTEGRATION_PASS`.

The successful child emitted the canonical production R22 adapter disposition:
`RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS`.

That child R22 audit simultaneously reported:
- `integration_valid: true`
- `sportsbook_inputs_added: 0`
- `current_or_future_outcomes_used: 0`
- `max_mean_delta: 3.552713678800501e-15`
- all frozen R22 audit gates `true`.

The canonical R22 production adapter and existing production validators use `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS`. Therefore Run2 is classified as a verifier-contract/name mismatch, not an R22 semantic/scientific failure and not an availability failure.

## Frozen minimum repair

The only authorized verifier behavior change is to recognize the canonical current R22 production-adapter PASS disposition `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS` in the existing R22 accepted-disposition check.

The existing R22 mean-neutrality threshold remains exactly `max_mean_delta <= 1e-10`. No model, tail shape, mean, opportunity, eligibility, role, T-75, sportsbook ordering, Full Slate behavior, or scientific threshold may change.

The production verification launcher must pin this repair document and the repaired verifier blob before Run3. The implementation lock must be relocked before Run3.

## Run3 decision rule

Run2 remains immutable mechanical evidence. Run3 must rerun the exact no-odds branch verification from a clean checkout. Any subsequent mechanical failure is preserved separately. Any semantic/scientific invariant failure means no promotion. Only `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_PASS_READY_FOR_MAIN_PROMOTION` may authorize main promotion.