# WEEK 6 INDIVIDUAL OPPORTUNITY LOCK — READINESS RESULT

Date: 2026-10-13

Status: **NOT READY — FRESH REPROBE CONFIRMS EXACT WEEK-6 ROSTER SOURCE STILL UNAVAILABLE. FAIL CLOSED.**

Latest readiness authority:
- workflow run: `38406807851`
- exact head: `e5b3fad39cc67a3bdad8ee0618d4952f8f9661ac`
- diagnostic artifact: `11817249788`
- digest: `sha256:d56de4b59ee3402f351122b9a4f6b027026007a4246d22907cffcb443d403a49`
- result disposition: `PLAYER_STATE_FUTURE_WEEK_NOT_READY`

Prior readiness authority:
- run `38399980031`
- artifact `11812829801`

The fresh reprobe used the already-frozen future-week player-state / opportunity
lock machinery. No source contract or football formula changed.

What is ready:
- immediately prior Week-5 stats: complete for all required teams;
- immediately prior Week-5 snaps: complete for all required teams;
- current usage coverage among played: **99.458%**;
- RB/WR/TE latest snap coverage among played: **99.379%**;
- stable ID coverage: **99.848%**;
- chronology violations: **0**;
- scheduled Week-6 teams: **28**.

Blocking fields:
- target week: **6**
- roster source week: **5**
- `target_week_roster_source_ready = false`
- target-week injury rows available: **0**

Integrity:
- target-week outcomes read: 0
- sportsbook inputs: 0
- OddsAPI calls: 0
- production changed: false

Interpretation:
the Week-6 player-state data are still not eligible for an immutable prospective
lock because exact Week-6 weekly roster authority is not present. Week-5 roster
identity must not be substituted.

No Week-6 opportunity lock was created:
- no share-volatility lock;
- no RB carry/snap lock;
- no RB receiving lock;
- no WR/TE trajectory lock;
- no immutable Week-6 lock pack;
- no Week-6 joint Dirichlet shadow.

Next move:
reprobe the same frozen workflow only after exact Week-6 roster authority
appears. Do not loosen the source contract. Historical research may continue
without using Week-6 outcomes.
