# Current Player Availability V1 — Core Implementation Lock

Status: `CORE_IMPLEMENTATION_LOCKED / FULL_SLATE_PRODUCTION STILL UNWIRED`

This lock freezes the validated source/reconciliation/timing core before any production Full Slate wiring. It does not itself authorize production promotion.

## Frozen design authority

- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- Confirmed gap result commit: `810c344a437d411707185317033acc7004f1c7db`
- Operational fix-plan commit: `b2206e7ad693148623447bcf9a3ad6b594033500`
- Fix-plan blob: `aa2980e61517eeeac650d79be2a09b1014cf49cb`
- Canonical timing-plan commit: `905f1bbe55d51676587d941295d357a1d2c31e9b`
- Timing-plan blob: `e72bf0c5e32aa7b06c559a5836fd883028e9cc1a`
- Timing concurrency resolution commit: `910e707159bfd98f6d62f0c493d0e0fb30ab1881`
- Canonical isolated timing branch: `ops-current-player-availability-t75-v1`

## Exact locked core blobs

- timestamped Ourlads depth/status source: `scripts/providers/ourlads_depth_status_v1.py`
  - blob `c115816ea8aa4ba7150a635c3115546d43f94b3c`
- current-player availability resolver: `scripts/build/build_current_player_availability_v1.py`
  - blob `9a1b0a672db7854ff52764ad93e5fcd89f9cb0ea`
- official NFL inactive adapter: `scripts/providers/nfl_official_inactives_v1.py`
  - blob `0d67316b6d7b7b9aa9d3637a07da4cd2b171639e`
- T-75 game-window certification validator: `scripts/validate_current_player_availability_timing_v1.py`
  - blob `d67ae30ed6e837f62098671c499d05462fe9d837`
- availability semantic fixtures: `tests/test_current_player_availability_v1.py`
  - blob `c51ac79173b651395fb733d92ac8138e7422e822`
- T-75 timing fixtures: `tests/test_current_player_availability_timing_v1.py`
  - blob `249fe23bff606fa977a245097eeb5875908337c1`

## Validated evidence

Semantic availability fixtures:
- preserved workflow-plumbing Run1: `34436894543`, job `102743747393`; all 8 semantic tests passed, shallow-checkout parent-diff failed only because the parent object was unavailable locally.
- first clean semantic run: `34436970099`, job `102743973821`, head `d3fadbd82953a3b2b1ad4168dbb82741ca62d167`; SUCCESS; all 8 fixtures PASS and protected production files unchanged.

Live source smoke:
- run `34437032282`
- job `102744156238`
- artifact `10136545256`
- digest `sha256:decb703afe4769befba790d8b1adceb0accb5a49eec4f5fd8f5d2adc6c7eb75a`
- 32/32 Ourlads teams, 468 rows, timestamp/source provenance present.
- NFL `/inactives/` endpoint reachable but zero complete sections at the pre-publication snapshot; correctly recorded as no valid official-inactive payload rather than active evidence.

T-75 timing fixtures:
- preserved dependency-plumbing Run1: `34437550771`, job `102745673287`, head `86c68b98b4ed6974b5d25c06e6ef99d6ed9aff17`; frozen-plan/production-boundary checks PASS, tests not run because pytest was absent.
- repair record: `docs/operations/CURRENT_PLAYER_AVAILABILITY_TIMING_V1_RUN1_PYTEST_DEPENDENCY_MECHANICAL_REPAIR.md`
- first valid timing run: `34437715931`
- job `102746163583`
- head `092fee088f3402d6313bc02b2f8cc05d1f3f54f9`
- SUCCESS; frozen T-75 contract PASS, protected production boundary PASS, all 8 timing fixtures PASS.

## Locked semantics

- complete validated official NFL inactive section is strongest availability authority;
- definitive weekly OUT/IR/PUP is next;
- Ourlads inactive is next;
- QUESTIONABLE/DOUBTFUL remain uncertain/eligible until definitive evidence;
- definitive unavailable players have zero eligibility and no reconciled active role;
- endpoint reachability never certifies official-inactive evidence;
- official sections become required for production certification at T-75 minutes before scheduled kickoff;
- missing/incomplete required sections fail closed only the affected game;
- at/after kickoff, a game is locked against new pregame pricing;
- sportsbook data is never used to establish availability or role.

## Production boundary

No current production Full Slate workflow, PlayerForm implementation, P3, R26, R22, M89/M90, WR/TE production model, simulation rule, or sportsbook path has been modified by this locked core.

The next step must be a separately frozen Full Slate integration plan that defines exactly how reconciled eligibility/roles feed each promoted position component and how opportunity is conserved when a definitive unavailable player is removed. No production wiring may precede that integration plan.
