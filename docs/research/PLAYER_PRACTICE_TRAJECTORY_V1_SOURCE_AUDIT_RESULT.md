# Player Practice Trajectory V1 — Source Audit Result

Date: 2026-09-24

Status: `PRACTICE_TRAJECTORY_SOURCE_NOT_DENSE`

## Canonical execution

- frozen plan: `docs/research/PLAYER_PRACTICE_TRAJECTORY_V1_SOURCE_AUDIT_PLAN.md`
- final source-audit head: `e9136af88d365756b28979f9ac258402b6189dc6`
- valid run: `36004655237`
- job: `107649589475`
- artifact: `10810042082`
- artifact digest: `sha256:907c9d9ac62a9c1b72ee7462a4a2a0cf6660a612f377a12783a011bb18d9fc5f`

Integrity:
- football outcomes read: 0
- sportsbook inputs: 0
- predictive models fit: 0
- production changes: 0

## Findings

The maintained nflreadpy injury feed exposes:
- `practice_status`
- `report_status`
- `date_modified`
- complete position resolution in this audit.

But it does **not** preserve the within-week daily sequence needed for a practice-trajectory signal.

2023-2025 skill-position player-weeks with a practice status: **4,820**.

Density:
- player-weeks with >=2 distinct report dates: **0.0%** under the frozen gate (only one isolated 2024 player-week had two dates in the full density file);
- player-weeks with >=3 distinct report dates: **0.0%**;
- 2026 W1-W3 skill player-weeks in the same weekly feed: **169**, but there is still no within-week multi-day trajectory.

Raw feed shape confirms the problem:
- 2023: 5,599 rows / 5,599 player-weeks;
- 2024: 6,215 rows / 6,213 player-weeks;
- 2025: 6,068 rows / 6,068 player-weeks;
- 2026: 692 rows / 692 player-weeks.

`date_modified` is populated historically for 2023/2024 but does not create Wednesday/Thursday/Friday observations. It is absent in the current 2025/2026 feed in this audit.

## Interpretation

The current maintained injury source can support a **single weekly practice/status snapshot**, which prior RB work has already used as binary DNP/LIMITED/questionable/out-doubtful information.

It cannot support the genuinely new hypothesis:
`DNP -> LIMITED -> FULL`, consecutive limited days, first-practice-back timing, or other within-week practice trajectory.

Therefore do not pretend the weekly status row is a trajectory and do not rerun old single-status injury features under a new name.

A future trajectory study would require a genuinely new daily practice-report source or prospectively captured daily reports.

## Disposition

`PRACTICE_TRAJECTORY_SOURCE_NOT_DENSE`

This closes this exact historical source lane. It is a source/data failure, not a scientific rejection of practice trajectory as a football signal.
