# RB Vacancy Opportunity V1 — Cohort / Provenance Audit

Date: 2026-09-24
Branch: `research-rb-vacancy-opportunity-v1`
Status: NO SCIENTIFIC VERDICT — LEGITIMATE EVALUATION COHORT NOT YET AVAILABLE

## Purpose

This checkpoint follows the frozen V1 mechanics and no-outcome constructor. It asks only whether a legitimate pregame cohort exists on which the frozen vacancy transfer can be evaluated once, without reopening M96 or reconstructing availability from postgame outcomes.

## Frozen Week-2 pregame authority checked

The preserved successful Week-2 Full Slate smoke artifact was inspected directly:

- run: `35263557385`
- artifact: `10515308221` (`run_35263557385`)
- artifact digest: `sha256:cf9a155a1bd7eb7c3542cf8718c07bd6b9c857532f5060fab1c84dccf6d7e0da`
- artifact generated 2026-09-17 before the Week-2 slate
- `current_player_availability.csv`: 468 rows
- `roles_current_production_eligible_v1.csv`: 467 rows
- `player_game_logs.csv`: present with strict-prior historical/current evidence
- official inactive section is complete in the availability artifact

A second exact paid Week-2 football-origin artifact (`35282021679` / `10523345092`) was also inspected as a cross-check.

### Vacancy-event result

Both preserved Week-2 artifacts contain **zero RB/FB rows with `definitive_unavailable == 1`**.

Therefore Week 2 contains no qualifying V1 vacancy event under the already-frozen semantics. This is not a model failure and not permission to broaden the cohort to DOUBTFUL/QUESTIONABLE or infer absence from target-game participation.

## Historical reconstruction status

A retrospective 2024-2025 vacancy grade is still not legitimate. The repository's canonical operating contract already records that historical W2-18 RB availability/injury source-timestamp provenance is unresolved. The V1 frozen plan explicitly requires failure closed if pregame availability cannot be reconstructed with timestamp-safe provenance.

Accordingly this audit does **not**:

- infer OUT status from zero target-game snaps/carries;
- use postgame participation to create vacancy labels;
- promote DOUBTFUL/QUESTIONABLE to definitive absence;
- reopen M96 exposed-2025 routing/threshold work;
- search injury-status weights or snap/rush blends;
- attach sportsbook data upstream.

## Mechanical execution harness

A research-only workflow was added at `.github/workflows/rb-vacancy-opportunity-v1.yml` to materialize the frozen no-outcome state from the preserved Week-2 pregame artifact plus nflverse strict-prior snap counts. It retains the focused invariant tests and uploads only no-outcome state/exclusions. Because the qualified Week-2 authority contains zero definitive RB/FB vacancies, this harness is a mechanics/provenance check, not a science grade.

## Disposition

`NO_QUALIFYING_PRESERVED_VACANCY_EVENT_YET_PROSPECTIVE_CAPTURE_REQUIRED`

No scientific V1 verdict exists. The hypothesis remains open.

## Exact next action

Use the canonical current-player-availability pipeline to freeze the next prospective 2026 slate **before kickoff**. If that frozen slate contains at least one definitive `UNAVAILABLE_*` RB/FB with strict-prior rush-share evidence and at least one eligible successor with strict-prior snap evidence, materialize the no-outcome vacancy state immediately and lock it. Attach target-game outcomes only after the game is complete, then grade once against the frozen V1 gates.

If the next slate again contains no qualifying vacancy event, record `NO_EVENT` and continue prospective capture; do not manufacture a retrospective cohort.


## Prospective 2026 Week-3 capture — 2026-09-25

A newer canonical no-live-odds Full Slate artifact was inspected prospectively while Week 3 remained in progress:

- Full Slate run: `36179691351`
- artifact: `10883314750` (`run_36179691351`)
- digest: `sha256:7d5b10af63c9a64d58b002f7a5cfad247d1cb5b80f0ec3ee806f412293321645`
- main head: `6ee39bf86fcf031eb498f10e3e111b71ebcd40c3`
- availability generated: `2026-09-25T19:28:25.523654+00:00`
- season/week from game certification: **2026 / Week 3**
- sportsbook inputs used: **0**

Availability state at this capture:
- availability rows: **463**
- definitive unavailable players: **9**
- definitive-unavailable RB/FB players: **0**
- game certification: **16** games total; **15** still `NOT_YET_REQUIRED`; ATL-GB already `KICKED_OFF_LOCKED`
- production-eligible output excludes the already-locked ATL/GB game

Disposition for this timestamped capture:

**`RB_VACANCY_V1_WEEK3_2026_09_25_NO_EVENT_ASOF_CAPTURE`**

This is a legitimate prospective no-event observation, not a scientific failure. Because 15 games had not yet reached their final pre-kickoff availability-certification window, it is also **not** a declaration that all of Week 3 is permanently event-free. Continue prospective capture as the remaining games approach kickoff.

Do not broaden V1 to DOUBTFUL/QUESTIONABLE, infer vacancy from later target-game participation, or attach outcomes when the frozen definitive-unavailable RB/FB condition is absent.
