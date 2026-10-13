# WEEK 6 INDIVIDUAL OPPORTUNITY LOCK — READINESS RESULT

Date: 2026-10-13

Status: **NOT READY — EXACT WEEK-6 ROSTER SOURCE NOT YET PUBLISHED. FAIL CLOSED.**

Canonical readiness authority:
- workflow run: `38399980031`
- exact head: `70090477c416541ce2475b218ea502f3d9833091`
- diagnostic artifact: `11812829801`
- result disposition: `PLAYER_STATE_FUTURE_WEEK_NOT_READY`

What is ready:
- immediately prior Week-5 stats: complete for all required teams;
- immediately prior Week-5 snaps: complete for all required teams;
- current usage coverage among played: **99.458%**;
- RB/WR/TE latest snap coverage among played: **99.379%**;
- stable ID coverage: **99.848%**;
- chronology violations: **0**;
- scheduled Week-6 teams: **28**.

Blocking field:
- target week: **6**
- roster source week: **5**
- `target_week_roster_source_ready = false`
- target-week injury rows available: 0

Interpretation:
the Week-6 player-state data are not yet eligible for an immutable prospective
lock because the exact target-week roster source is not present. Week-5 roster
identity must not be substituted.

No Week-6 opportunity lock was created:
- no share-volatility lock;
- no RB carry/snap lock;
- no RB receiving lock;
- no WR/TE trajectory lock;
- no immutable Week-6 lock pack.

The workflow correctly failed closed before those stages.

Next move:
rerun the same frozen workflow only after exact Week-6 roster authority appears.
Do not loosen the source contract.
