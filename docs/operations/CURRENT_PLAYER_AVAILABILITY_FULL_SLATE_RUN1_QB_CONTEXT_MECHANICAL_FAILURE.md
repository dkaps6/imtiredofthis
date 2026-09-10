# Current Player Availability Full Slate — Run 1 Mechanical Failure

Status: `MECHANICAL_FAILURE_BEFORE_AVAILABILITY_EXECUTION_NO_INTEGRATION_RESULT`

## Immutable failed lineage

- candidate branch: `ops-current-player-availability-full-slate-v1`
- locked candidate head: `9147bde6d08c05249415ec4cd364db482d812a35`
- first lock-triggered run: `34439714153`
- job: `102752015236`
- artifact: `10137497418`
- artifact digest: `sha256:0f3f5195fce8bc6514804b411e0b312475a379cfeb722eb5f5280237f8ff50f2`
- failed step: `Build team context and promoted QB context`
- exact failing command from decoded job log: `python scripts/run_team_form_context.py --season 2026 --box-backfill-prev`
- exact exception: `RuntimeError: active-season PBP has no completed pre-target rows for season=2026 week=1`

The decoded job log shows Sharp context and the prior bridge completed. The legacy TeamForm build then wrote a partial current-season table before the wrapper's strict pre-target repair correctly rejected the lack of legal `week < 1` current-season PBP. `run_qb_promoted_context.py` never executed. Therefore the failure is a Week-1 TeamForm source-selection/plumbing incompatibility and occurred before any current-availability, PlayerForm, RB P3, C2, static-audit, or 35-gate integration execution.

This run is not a scientific/integration FAIL and must not be scored against the frozen availability design.

## Value-neutral repair

For target Week 1 only, force the already-declared `PRIOR_SEASON` PBP fallback before the canonical TeamForm runtime wrapper executes. This matches the wrapper's pre-existing documented contract: at Week 1 there are no completed current-season games strictly before the target week, so prior-season regular-season PBP is the legal pregame source. The repair refuses to run outside Week 1.

No TeamForm formula, model code, feature definition, source hierarchy beyond this declared Week-1 fallback, T-75 rule, role logic, availability logic, production code, R22/R26 behavior, or frozen integration gate changes are authorized.

The repaired run remains pre-integration plumbing evidence until the no-odds Full Slate reaches substantive completion.
