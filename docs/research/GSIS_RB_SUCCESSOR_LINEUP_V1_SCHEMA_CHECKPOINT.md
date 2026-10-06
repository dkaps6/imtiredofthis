# GSIS RB Successor Lineup V1 — Private Schema Checkpoint

**STATUS: SCHEMA / MECHANICS ONLY — NO TARGET OUTCOMES — NO PROSPECTIVE OBSERVATION YET.**

Frozen parent plan:
`docs/research/GSIS_RB_SUCCESSOR_LINEUP_V1_PLAN.md`

Private source authority used only for parser/schema validation:
- snapshot SHA256:
  `779b1bd6c5d4dbd390c992d1e02e007edb80dac7d5e19b8b9045b161000c8a23`
- season/phase: 2026 REG
- source boundary: existing immutable point-in-time archive already used by the
  completed GSIS incremental-information audit.

Private-safe parser result:
- valid offensive exact 11-player lineup rows: **2,118**
- teams represented: **32**
- normalized within-lineup identity collisions: **0**
- raw lineup rows emitted publicly: **0**
- player identities emitted publicly: **0**
- target outcomes read: **false**
- sportsbook inputs read: **false**

This reproduces the prior audited Lineup Detail row/team counts and establishes
that the new successor-weight helper can consume the real private schema without
requiring a new parser interpretation.

Important boundary:
the existing 2026 snapshot is **not** used to backfill a pregame candidate for a
past game. It is parser/schema authority only. The first scientific observation
must come from a future target game with an immutable GSIS snapshot locked
before kickoff under the frozen parent plan.

Implementation:
- `scripts/research/gsis_rb_successor_lineup_v1.py`
- synthetic tests:
  `tests/test_gsis_rb_successor_lineup_v1.py`

The helper:
- conditions lineup exposure on all frozen unavailable RB/FB identities being
  absent;
- weights only surviving successor RB/FBs;
- conserves the already-frozen RB Vacancy V1 vacated share;
- abstains when no GSIS successor exposure exists;
- can write exact candidate rows only to an explicitly supplied private path;
- public audit output contains aggregate integrity counts only.

No production change is authorized.
