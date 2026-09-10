# Current Player Availability — Production Verification V1 Frozen Plan

Status: `FROZEN_BEFORE_PRODUCTION_VERIFICATION_IMPLEMENTATION_OR_RUN`

## Authority

- Promotion plan commit: `9b3241996c50f1a05d124acd370fa09f0bc07e35`
- Promotion plan blob: `2d350379377ae65b2fb094504624da0591f53f7e`
- Immutable 35/35 finalized artifact: `10146675272`
- Immutable 35/35 digest: `sha256:dcc0fec6f07c62542e7115a49c1c45e9a185d18fd3baa2e423e0bd3dd08d0286`
- First valid gates 1-34 source run/job: `34461561636` / `102820358570`
- First valid gates 1-34 artifact/digest: `10145975346` / `sha256:dd32b45f6746176911ca68aa1d73a8a75325f92cf7b0951c335ae5484752e0ed`
- Certified no-odds football-stack runner blob: `dc60ab10bc18777d6b13cb5f81f285e8c248f3ab`.

## Why a separate verification runner is required

The immutable certification runner was intentionally written against the historical certification snapshot in which exactly one already-kicked-off game was withheld, yielding 30 eligible teams. Its hard-coded 30-team assertion is part of that immutable certification evidence and must not be edited or reinterpreted.

Production verification must instead validate the **current certified eligible team set**, which can legitimately be 32, 30, or another even count depending on kickoff/T-75 certification state. Therefore production verification requires a separately frozen verification-only adapter rather than altering the immutable certification runner.

## Frozen production verification semantics

The verification-only runner must:

1. read `roles_current_production_eligible_v1.csv`, `current_player_availability.csv`, and `current_player_availability_game_certification.csv` generated in the same clean Full Slate run;
2. derive the expected team set solely from the certified current-role artifact using `scripts.utils.eligible_team_set_v1`;
3. independently confirm that the derived team count equals exactly two times the number of `production_eligible=1` certified games and that no team occurs in more than one weekly game;
4. require `player_form_consensus.csv` to contain exactly the same certified eligible team set;
5. require every definitive-unavailable identity to be absent from PlayerForm and from generated simulation arrays;
6. execute the same promoted football stack as the certified runner using synthetic football-only lookup rows with no line, odds, book, market probability, or sportsbook event identity;
7. require the Full Slate football-universe audit to report the exact dynamically derived eligible team count and eligible game count;
8. require M38/TE-R5P/WR-R15 entitlement conservation, QB C2 mean-neutral/current-output invariants, R22 mean neutrality, R26 football-only usage, and outer P3 conservation exactly as already certified;
9. require sportsbook inputs used by this verification to equal zero;
10. write a production-verification JSON result containing the eligible team/game counts and invariant dispositions.

The runner may generalize only the immutable certification runner's hard-coded historical `30`/`15` snapshot assumptions into dynamically derived certified eligibility counts. It may not alter any model, entitlement, tail, opportunity, role, availability, or timing semantics.

## Full Slate verification boundary

The dedicated promotion-branch verification run must use `FETCH_LIVE_ODDS=false` and a clean checkout. It must execute availability before opportunity, explicit current roles, the complete 32-team QB state-context source build, the availability-aware current-output coverage seams, and then the verification-only promoted football stack.

A PASS requires:
- exact 35/35 authority verified;
- exact certified source/provider/helper blobs verified;
- clean scientific-model/protected-artifact diff boundary;
- valid T-75/current-role artifacts;
- zero definitive unavailable positive opportunity;
- zero withheld-team resurrection;
- dynamic eligible-team/game coverage exact;
- complete 32-team QB state-context source integrity retained;
- promoted football-stack invariants pass;
- sportsbook inputs to football and verification equal zero;
- strict repository and 2026 production-readiness audits pass.

## Disposition

Allowed branch-verification terminal results:
- `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_PASS_READY_FOR_MAIN_PROMOTION`
- `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_MECHANICAL_FAILURE_NO_DECISION`
- `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_SEMANTIC_FAILURE_NO_PROMOTION`

Any mechanical failure must be preserved and minimally repaired under a separately frozen repair. A semantic/scientific invariant failure means no promotion. Only the PASS disposition may authorize a subsequent main promotion and clean-main verification.
