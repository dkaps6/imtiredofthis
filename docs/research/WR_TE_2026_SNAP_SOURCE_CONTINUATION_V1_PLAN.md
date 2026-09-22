# WR/TE 2026 Snap Source Continuation V1 — Frozen Plan

**Status:** FROZEN BEFORE RESULT OUTPUT  
**Branch:** `research-current-season-state-persistence-v1`  
**Parent result:** `CURRENT_SEASON_STATE_PERSISTENCE_V1_COMPLETE`  
**Production change authorized:** NO  
**Model refit authorized:** NO  
**Sportsbook inputs to football model:** PROHIBITED

## Purpose

Test whether the promoted WR-R15 and TE-R5P entitlement models should be allowed
to consume already-available 2026 strict-prior offensive snap participation.

Current production shares one snap loader whose source-season list is hardcoded
to 2020-2025. nflverse already exposes complete 2026 Weeks 1-2 snap counts.

This is a source-continuation test only. The learned WR-R15 and TE-R5P model
coefficients, scalers, clips, room-conservation rules, and M38 WR1 anchor are
immutable.

## Candidate change under test

Baseline source seasons:

`[2020, 2021, 2022, 2023, 2024, 2025]`

Candidate source seasons:

`[2020, 2021, 2022, 2023, 2024, 2025, 2026]`

No other difference is allowed.

## Week-1 invariance proof

Use the real 2026 snap source with a target ordinal of 2026 Week 1.

Required proof:

- baseline and candidate strict-prior features are exactly equal for the same
  player/team fixture;
- candidate may load 2026 rows, but no 2026 row may be eligible because strict
  prior requires source ordinal < target ordinal;
- baseline and candidate TE-R5P / WR-R15 entitlement outputs must therefore be
  numerically identical on a representative frozen fixture.

This is an adapter-level proof. The preserved Week-1 certification artifact does
not retain the full pre-specialist football frame, so no claim of full Week-1
slate replay is permitted.

## Real Week-2 preserved-slate replay

Source authority:

- Full Slate run `35282021679`
- artifact `10523345092`
- artifact name `run_35282021679`
- digest `sha256:6024ed21d7032e6e6440145277d9d4b74e98f319ef286962bac2a786a8d28e3c`
- source was frozen before Week-2 kickoff.

Use the preserved `data/football_simulation_universe.csv`.

The frame already contains the exact pre-TE-R5P entitlement column
`baseline_entitlement_tgt_share`, plus the columns required by the promoted
adapters.

Reconstruct the common pre-specialist input by setting:

`entitlement_tgt_share = baseline_entitlement_tgt_share`

and leaving all other football inputs unchanged.

Run two paths:

1. baseline snap source (2020-2025);
2. candidate snap source (2020-2026).

Apply the exact promoted sequence:

`TE-R5P -> WR-R15`

No refit and no coefficient mutation.

## Strict-prior gate

For the Week-2 candidate:

- source Week 1 2026 snaps may be consumed;
- source Week 2 or later 2026 snaps are forbidden;
- any eligible snap row with ordinal >= 202602 is fatal;
- the test must report how many TE/WR rows actually acquire a 2026 prior-snap
  observation.

## Projection-impact gate

Run the exact explicit-entitlement football simulation on baseline and candidate
frames using identical seed and iteration count.

Report for WR and TE receiving markets:

- player rows changed in entitlement;
- max / mean absolute entitlement delta;
- receiving-yards mean delta;
- receptions mean delta;
- max / mean absolute projection delta;
- largest player-level movers.

The candidate is expected to change Week-2 football projections when Week-1
participation differs from the stale 2025-only snap history. A non-zero change is
not itself a pass or fail.

## Conservation / authority gates

Both arms must preserve:

- one player/game/team row;
- TE room target mass exactly;
- WR2+ room target mass exactly;
- M38 WR1 target entitlement exactly;
- non-target-position entitlement exactly;
- team total player entitlement exactly;
- same football simulation key universe;
- no sportsbook-derived input;
- no target/future 2026 snap use.

## Interpretation

Possible dispositions:

- `SNAP_CONTINUATION_MECHANICALLY_VALID_AND_MATERIAL`
- `SNAP_CONTINUATION_MECHANICALLY_VALID_SMALL_EFFECT`
- `SNAP_CONTINUATION_NO_EFFECT`
- `SNAP_CONTINUATION_INTEGRITY_FAIL`

This experiment does not determine whether Week-2 outcomes improve. The user and
Claude are independently grading Week-2 model performance. Outcome-based
promotion requires a separately justified evaluation or prospective evidence;
do not tune the source continuation against Week-2 results after seeing them.

## Production boundary

No production edit to `SOURCE_SEASONS` is authorized by this plan alone.

A production change may be proposed only after:

1. all integrity/parity gates pass;
2. effect size is documented;
3. the result is reviewed against the independent Week-2 model-performance
   grade and the existing WR/TE authority lineage;
4. no coefficient/threshold retuning is introduced.

## Disposition

`WR_TE_2026_SNAP_SOURCE_CONTINUATION_V1_PLAN_FROZEN`
