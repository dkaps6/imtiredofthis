# RB PD2 FORWARD / SHADOW ACTIVATION V1

**STATUS: OPERATIONAL ACTIVATION CONTRACT. SCIENCE UNCHANGED. NOT ACTIVE ON `main` UNTIL THE IMPLEMENTATION PR LANDS.**

## Purpose

This document records the operational boundary that turns the already-frozen
RB-PD2 forward/shadow confirmation from an implemented research pipeline into a
prospective study that can accumulate immutable future observations.

It does **not** change the candidate, the historical qualification, the forward
gates, or production pricing science.

Authority remains:

- `docs/research/RB_PD2_FORWARD_SHADOW_CONFIRMATION_V1_PLAN.md`;
- `docs/research/RB_PD2_YARD_DIFFICULTY_MC_WIDTH_V1_PLAN.md`;
- `docs/research/RB_PD2_YARD_DIFFICULTY_MC_WIDTH_V1_AMENDMENT3.md`.

## Prospective activation boundary

Frozen operational start:

`2026-09-22T12:00:00Z`

Consequences:

- no Week-1 or Week-2 2026 player-game may ever count as a prospective
  confirmation observation;
- the first possible confirmation observation is a future Week-3-or-later
  eligible RB/HB/FB rushing-yards player-game;
- already-played 2026 rows may enter predictor history only through the
  separately certified leakage-safe history route; they are never backfilled
  as prospective locks.

The boundary is intentionally after completion of the 2026 Week-2 slate so an
implementation or replay performed during Week 2 cannot accidentally become
observation #1.

## Frozen history artifact for first live lock

Pinned successful workflow:

- workflow: `RB PD2 Forward History Build V1`
- run ID: `35620321000`
- artifact: `rb-pd2-forward-history-v1`
- artifact ID: `10648116139`
- artifact digest:
  `sha256:52e56ce090576480d9b8cf4ec1999bfaf9e6884e6a0d1266efc075f1afac3af8`
- retention: 90 days
- source branch head for the successful run:
  `da9a6fd86652aee7b334944d335dd13b84e25a96`

Verified contents:

- 1,500 history rows;
- 168 players;
- 107 certified completed 2026 Week-1 rows;
- 808 scoreable history rows under the frozen minimum-prior/reference contract;
- maximum strict-prior reference N = 882;
- Week-1 certified history SHA-256:
  `c9c3ab5f1b1873fb32521358342fa1917fdf9da8fb7b80b8500d437f74ba00ef`;
- Week-1 prospective confirmation observations created = 0.

The Week-1 route mechanically reasserts exact P3/STACK1 parity from the two
underlying projection values. A caller-provided boolean is not sufficient.

## Live Full Slate activation

The authoritative Full Slate remains production-first.

Research capture is controlled by workflow-dispatch input:

`rb_pd2_shadow_capture`

Default is **false**. The environment variable is empty unless the manual input
is explicitly true, preserving the accepted §7 default-OFF import/no-op
contract.

When and only when all of the following are true:

1. live odds were explicitly requested;
2. live odds are available;
3. `rb_pd2_shadow_capture=true`;

the Full Slate job performs the following research-only actions after canonical
pricing has completed:

1. same-process §7 capture of exact production `adjusted_outcomes`;
2. restore the pinned green forward-history artifact above;
3. assemble the frozen §8 candidate and §9 immutable pregame lock;
4. immediately upload a dedicated artifact named
   `rb-pd2-forward-lock-<github_run_id>`.

History restore, lock assembly, and the dedicated research upload are all
non-blocking for canonical production pricing. A research failure therefore
invalidates the research lock, not the production board.

## Observation #1 acceptance rule

A live run does **not** count merely because shadow capture was enabled.

The first prospective observation exists only when the dedicated uploaded
artifact proves all of the following:

- capture session receipt is valid;
- capture provenance run ID / code SHA match the authoritative Full Slate run;
- stable identity-contract fingerprints are non-empty and exactly match the
  history manifest: `manual_name_overrides.csv` plus the exact
  `scripts/utils/canonical_names.py` implementation;
- `roles_ourlads.csv` whole-file SHA is retained as diagnostic provenance,
  not a hard equality gate, because it is a mutable weekly enrichment artifact;
- final football mean / exact baseline empirical array were captured before
  kickoff;
- schedule matchup matches both canonical team and opponent;
- capture timestamp and lock timestamp are both earlier than authoritative
  `kickoff_utc`;
- capture and lock timestamps are on/after
  `2026-09-22T12:00:00Z`;
- the player has >=4 prior eligible games;
- strict-prior difficulty reference N >=100;
- candidate is the exact frozen mean-neutral transform;
- baseline/candidate draw count matches;
- baseline/candidate mean parity is within 1e-8;
- sportsbook inputs used in candidate = false;
- production output mutated = false;
- outcome present at lock = false;
- lock artifact itself was uploaded pregame.

A valid session with zero scoreable locks does not create an observation.

## Production consequence

None.

This activation only permits prospective evidence collection. A future
scientific PASS still enables a separate promotion review; it does not
automatically alter production science.

## Parallel mean-information lane

Per the frozen plan, the separate RB mean-information lane may formally begin
only after the first valid future lock exists.

Before that point, new-data work is limited to source inventory/readiness and
production-correctness audits. No new RB candidate, threshold search, or
retrospective outcome-tuned router is authorized.
