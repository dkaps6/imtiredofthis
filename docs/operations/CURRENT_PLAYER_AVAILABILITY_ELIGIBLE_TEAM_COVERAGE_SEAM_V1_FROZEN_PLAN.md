# Current Player Availability — Eligible-Team Coverage Seam V1 Frozen Plan

Status: `FROZEN_BEFORE_DOWNSTREAM_AVAILABILITY_AWARE_COVERAGE_IMPLEMENTATION`

## Parent authority

- protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- frozen 35-gate integration plan: `91ee6aa3ad3813c7d285f6f3163368205937eb09`
- integration-plan blob: `54eb4629c48062fcaef3153918b2069238584d0a`
- first mechanically valid no-odds availability candidate head: `9800254f3ab208ab42501faac83a0d6e5fe3b93d`
- candidate run: `34447900206`
- candidate job: `102776660124`
- candidate artifact: `10140425929`
- candidate digest: `sha256:6b31ae40d648780673b7953b57323d509b302ea7a82fe99801b024ede2603f37`

## Why this seam is required

The frozen T-75 design withholds a non-production-eligible game before opportunity/simulation. In the first mechanically valid live candidate snapshot, NE-SEA was already kicked off, so the certified production-eligible football universe contained 30 teams / 15 games.

Two downstream legacy guards were written for an untouched 32-team Week-1 slate:

1. `scripts/run_pricing_with_full_roster_universe_v1.py` requires exactly 32 teams in the football simulation universe.
2. `scripts/modeling/rb_r26_receptions_production_adapter_v1.py` requires RB/FB coverage across exactly 32 teams.

Those guards would mechanically reject a correctly withheld game before the frozen 35-gate integration test can exercise M38/TE-R5P/WR-R15/R26/R22 on the eligible universe.

## Frozen change

When and only when an explicit current-availability active-role input is configured (`ACTIVE_ROLES_CSV` or an equivalent explicit candidate argument), downstream *coverage validation* may derive the expected current team set from that certified active-role artifact.

The eligible-team set must equal the teams in `data/roles_current_production_eligible_v1.csv`, which itself is downstream of the locked T-75 game certification and current-player availability resolver.

Default behavior with no explicit availability seam remains the original 32-team requirement.

### Full football universe

- Require the PlayerForm/model-context football universe team set to equal the certified eligible-team set exactly.
- Require canonical game count to equal half the eligible-team count.
- Do not add placeholder players or placeholder games for withheld teams.
- Do not use sportsbook offers to determine eligible teams.

### R26

- Keep the frozen R26 model, R9 feature contract, vacancy-team set, CIN control semantics, room-state asset, coefficients, reliability, clipping and redistribution math unchanged.
- Replace only the legacy *current coverage assertion* `RB/FB teams == 32` with exact equality to the certified eligible-team set when the explicit availability seam is active.
- A withheld team contributes no R26 current opportunity because it is absent from the eligible football universe; do not renormalize across teams.
- The frozen 31-team vacancy asset remains an immutable prior/contract; it is not rewritten because a current game is withheld.

### QB C2 state context

The team-level QB C2 context source may still materialize all scheduled teams. At the actual football-universe/application seam, only eligible-team/player keys may receive simulation/pricing output. Do not alter C2 selector parameters or state construction merely to force the upstream context artifact to 30 teams.

### WR/TE/R22

No scientific change is authorized. Their existing per-team/per-player conservation or tail logic must operate on the eligible universe and pass the frozen integration gates without parameter changes.

## Frozen invariants

- certified eligible teams come only from current availability + timing, never sportsbooks;
- withheld teams are absent from active roles, PlayerForm current universe, simulation metrics and priced output;
- no cross-team transfer of target/rush/pass entitlement occurs because a game is withheld;
- model versions/hashes unchanged;
- R26 and R22 strict-prior histories unchanged;
- M38/TE-R5P/WR-R15 room-conservation semantics unchanged;
- no current/future outcomes introduced;
- no modification of the 35 frozen promotion gates.

## Validation before first 35-gate result

A dedicated regression must demonstrate both modes:

1. legacy/no-availability mode still expects 32 teams;
2. explicit availability mode accepts exactly the certified eligible team set and rejects both missing and extra teams.

The first valid 35-gate certification may run only after this seam is implemented and locked. Any implementation/plumbing failure before the 35-gate evaluator produces all gate values is mechanical and cannot be interpreted as an integration failure.
