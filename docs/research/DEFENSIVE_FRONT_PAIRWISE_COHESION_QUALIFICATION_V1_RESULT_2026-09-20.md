# DEFENSIVE FRONT PAIRWISE COHESION QUALIFICATION V1 — RESULT — 2026-09-20

## Final disposition

`READY_FOR_FROZEN_EXPERIMENT`

This is a mechanically corrected qualification result. The original canonical qualification is preserved as an integrity rejection and was not overwritten or reinterpreted.

No predictive outcome was scored during qualification, forensic diagnosis, or identity correction.

## Frozen candidate

`def_front_pairwise_cohesion_prior_share`

- grain: scheduled team-game
- seasons: 2019-2025
- current front eligibility: `DE, DT, NT, DL, EDGE, OLB, ILB, LB`
- backups retained
- up to 20 strictly prior scheduled same-team games, crossing season boundaries
- no target-game participation, snaps, PBP, future roster, sportsbook, or outcome input

The frozen formula, position set, lookback, support floors, stability threshold, and redundancy threshold were not changed.

## Original qualification — preserved integrity rejection

Original implementation commit:

`dc34825ebd0c7a2c37bf530bf369d5c92fd32ba4`

Canonical original run:

- run: `35515764090`
- job: `106091315051`
- artifact: `10606857557`
- artifact digest: `sha256:c3a5d19b130569f30f85068fa4da5562aee5630e8ab2c1728b30eb4f53657c65`
- weekly roster SHA256: `f2b791d47b146fe703a73d3111d609504779c7e9d2dc0ad47b3bd1996776f18a`

The original run had strong scientific support but failed the frozen identity gate:

- eligible scheduled team-games: 3,742
- observed: 3,710
- coverage: 99.1448%
- raw nonblank GSIS coverage: 99.9692%
- adjacent-game stability Spearman: 0.813271
- redundancy holdout R2: 0.397787
- ambiguous same-week GSIS/team groups: **13**
- original disposition: `REJECTED_INTEGRITY`

The frozen identity gate required zero ambiguous same-week GSIS/team conflicts, so the rejection was correct.

## Forensic audit

A dedicated source-only audit was added without changing the candidate or reading outcomes.

Audit implementation/workflow commits:

- `f8c09b63c1b24be61e33ef82ee67c1307e086027`
- `20b3338f2d3030f95f5279625fa0d20aa0aeffda`
- enhanced alternate-ID audit: `c63bbbbf5caf593bcc4d21bbeb5f5d2192b8cc31`

Enhanced canonical forensic run:

- run: `35516818596`
- job: `106094055976`
- artifact: `10606947000`
- artifact digest: `sha256:ef8b0d3f164e1ebaf7052602da4b9acb261cac9f664bac3a684ac4e5ff708414`
- exact weekly roster SHA match: **true**

### Finding

All 13 original same-week team conflicts were the same GSIS value:

`00-0035718`

The source rows represented two different people:

- NYJ: Quinnen Williams
  - ESB ID: `WIL132672`
  - Smart ID: `32005749-4c13-2672-b05d-48fefa792d12`
- NYG: Isaiah Searight
  - ESB ID: `SEA499236`
  - Smart ID: `32005345-4149-9236-20c1-d64e2ac2e062`

The weekly roster source contains no transaction timestamp field that could establish a deterministic “latest pregame team” for an ambiguous row. More importantly, the differing ESB/Smart IDs prove this case is not one player moving teams in-week; it is a GSIS/person identity collision.

Because nflverse joins/coalesces additional player metadata by GSIS, the bad shared GSIS also causes downstream person metadata contamination. Choosing NYJ or NYG by name, status, result, participation, or outcome would have been a post-hoc person/team selection and was not allowed.

## Mechanical correction

Commit:

`2aeacc2004a42cda9a21282d21d2d09f2dfce15c`

A general source-identity rule was added:

> Quarantine a GSIS from stable-ID use when the same GSIS has more than one distinct nonblank upstream ESB ID or more than one distinct nonblank upstream Smart ID anywhere in the frozen weekly-roster source horizon.

Properties of the correction:

- does not select a team or player;
- does not use names as the identity authority;
- does not use target-game outcomes, snaps, participation, PBP, or future weeks;
- does not alter the pairwise-cohesion formula;
- applies globally to the source, not only to the 13 rows that caused the first failure;
- preserves the original same-week ambiguity gate for a single person/signature appearing on multiple teams.

Focused tests explicitly verify:
1. proven multi-person GSIS collisions are quarantined completely; and
2. same-person/multi-team rows remain ambiguous and fail closed.

## Corrected canonical qualification

Run:

- run: `35517035459`
- job: `106094605432`
- artifact: `10607480543`
- artifact digest: `sha256:367a608fbf6f37d83b63befbc405d38579702fda98b399c16147e80606363fa0`
- implementation SHA: `2aeacc2004a42cda9a21282d21d2d09f2dfce15c`
- weekly roster SHA256: `f2b791d47b146fe703a73d3111d609504779c7e9d2dc0ad47b3bd1996776f18a`

The weekly-roster bytes are identical to the original rejected run.

### Identity/integrity

- raw nonblank GSIS coverage: 99.9692%
- semantic collision GSIS IDs detected globally: 14
- front-roster rows quarantined by the general rule: 136
- post-quarantine stable-ID coverage: **99.8141%**
- ambiguous same-week GSIS/team conflicts after quarantine: **0**
- duplicate source identity rows: 0
- published duplicate team-week rows: 0
- chronology violations: 0
- schedule join fanout: 0
- redundancy join fanout: 0

The frozen stable-ID gate was >=99%, so the corrected source remains above the required floor.

### Support

- eligible scheduled team-games: **3,742**
- observed/known: **3,710**
- pregame coverage: **99.1448%**
- unknown: 32, all horizon cold starts
- median current front roster count: 23
- median current pair count: 253
- median prior roster games available: 20
- median mean prior co-rostered games: 10.03643

### Stability

- adjacent-game pairs: **3,678**
- Spearman: **0.813629**
- median absolute adjacent change: **0.036957**
- frozen stability gate: >=500 pairs and Spearman >=0.50
- result: **PASS**

### Redundancy

Train 2019-2023 / holdout 2024-2025:

- train rows: 2,622
- holdout rows: 1,088
- holdout reconstructibility R2: **0.401720**
- disposition: `INCREMENTAL_INFORMATION_SURVIVES_REDUNDANCY_GATE`

This remains well below the frozen 0.75 review threshold.

## Boundary audit

- predictive outcomes scored: false
- sportsbook read: false
- target-game PBP used: false
- target-game snaps/participation used: false
- future-week roster used to resolve identity: false
- production changed: false
- Issue #535 touched: false

## Final scientific interpretation

The first `REJECTED_INTEGRITY` result was a legitimate source-identity failure, not a failed cohesion signal. The forensic audit showed that the apparent team ambiguity came from a GSIS collision between different people. A deterministic, general, outcome-free quarantine rule repairs that identity defect without selecting a favorable row or weakening a frozen gate.

After that mechanical correction, Defensive Front Pairwise Cohesion V1 passes every frozen qualification gate and is authorized for a separately frozen mechanism experiment.

Do not use the original 13-conflict failure as a reason to retune thresholds, and do not remove the quarantine rule for future reruns.
