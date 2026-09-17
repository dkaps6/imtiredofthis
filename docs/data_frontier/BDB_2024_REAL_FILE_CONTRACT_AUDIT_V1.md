# BDB 2024 Real-File Contract Audit V1

**Scope:** Phase 0 data-engineering only. No predictive experiment, no production-science change, no sportsbook input, no Issue #535 direction.

**Branch:** `data-frontier-phase0-bdb-contact-v1`

**Prior checkpoint:** `88afda5258c7d5c2826e48120a6ee8b1d2943b04`

## Verdict

The current Phase-0 loader contract is compatible with the official NFL Big Data Bowl 2024 data dictionary for its required columns and file naming. A real competition-file run remains blocked in this execution environment because Kaggle requires an authenticated account that has accepted the competition rules before the files can be downloaded. No rules were accepted and no restricted data were acquired in this audit.

## Official file contract checked

Official competition data page: `https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/data`

Official rules page: `https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/rules`

The official package documents:

- `plays.csv`, keyed by `gameId`, `playId`;
- `tackles.csv`, keyed by `gameId`, `playId`, `nflId`;
- `tracking_week_[week].csv`, keyed by `gameId`, `playId`, `nflId`, with `frameId` providing frame grain.

### Required play columns

The V1 implementation requires:

- `gameId`
- `playId`
- `ballCarrierId`

All are documented in the official 2024 play table. The implementation optionally consumes `possessionTeam` and `defensiveTeam`; both are also official play columns and should normally make defender identification explicit.

### Required tackle columns

The V1 implementation requires:

- `gameId`
- `playId`
- `nflId`
- `tackle`
- `assist`
- `forcedFumble`
- `pff_missedTackle`

All are documented in the official 2024 tackle table. `pff_missedTackle` is explicitly identified by the competition as a Pro Football Focus-provided field.

### Required tracking columns

The V1 implementation requires:

- `gameId`
- `playId`
- `frameId`
- `playDirection`
- `x`
- `y`
- `club`

It also consumes when present:

- `nflId`
- `displayName`
- `s`
- `a`
- `dis`
- `o`
- `dir`
- `event`

All are documented in the official 2024 tracking table. The official dictionary says `nflId` is null for football rows, which matches the V1 normalization design: football is separated before player `nflId` non-null enforcement.

## Important source behavior confirmed

The official competition page states that tracking is event-window filtered rather than unrestricted full-play/full-game tracking:

- designed rushes: five frames before the snap event through five frames after the play-end event;
- scrambles: five frames before the quarterback-crossed-LOS event through five frames after play end;
- completions: five frames before the catch event through five frames after play end.

This matters materially for the contact benchmark. Designed rushes can support handoff/snap-to-end contact geometry. Completed passes do **not** provide route development or pre-catch geometry in this 2024 corpus. Scrambles begin around the LOS-crossing event rather than the snap. Therefore Phase 0 must not describe BDB 2024 as a universal snap-to-end tracking source.

## Event labels and V1 behavior

The official dictionary describes `event` as tagged play details including ball snap, pass release, pass catch, tackle, etc., but does not guarantee that every desired semantic event is present on every eligible play.

V1 therefore remains intentionally fail-visible:

1. prefer a source `handoff` event as the geometry start;
2. fall back to `ball_snap` when `handoff` is absent;
3. use recognized play-end events when present;
4. otherwise use the last available tracking frame;
5. abstain with `NO_CONTACT_EVENT_RESOLUTION` when a scoreable contact cannot be resolved.

No event-name expansion is justified from the public dictionary alone. Exact observed event-value frequencies must be recorded on the first lawful real-file run before any V2 event mapping is proposed.

## Licensing / access boundary

The official rules identify competition data access/use as **CC BY-NC 4.0**. Kaggle currently requires sign-in/registration and acceptance of competition rules to access the actual files. This automated audit did not accept terms on the user's behalf and did not download the competition files.

A future real-file benchmark may run only after the files are already lawfully available to the execution environment or the user explicitly handles the required access decision.

Raw competition files must not be committed to git.

## Compatibility disposition

| Contract area | V1 status | Action |
|---|---|---|
| filenames | compatible | none |
| play keys / ball carrier | compatible | none |
| offense / defense team columns | compatible | none |
| tackle/PFF labels | compatible | none |
| tracking keys/coordinates | compatible | none |
| football null `nflId` | compatible | none |
| speed/acceleration/direction fields | compatible | safe to derive richer geometry |
| event coverage | partially specified by public dictionary | record observed values on first real run; do not guess aliases |
| full-play coverage | **not universal** | preserve play-type/window limitations in QA |
| real-file execution | access-blocked | do not accept terms automatically |

## Next engineering tasks

1. Add V1-compatible deterministic geometry derivatives using the already-documented `x`, `y`, `s`, `dir` contract: relative closing speed, pursuit angle and sideline distance.
2. Add source-label reconciliation breakdowns that distinguish primary tackle, assist and missed-tackle overlap instead of one pooled overlap flag.
3. Persist normalized tracking and tackle tables in benchmark output so downstream QA can inspect the exact normalized source rows used.
4. Add an event-frequency/source-window audit to the real-file runner before the first benchmark result is interpreted.

These tasks do not alter the frozen 1-yard / two-consecutive-frame contact-candidate rule.