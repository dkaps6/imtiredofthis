# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-10-22 — PRISM WEEK-7 LIVE BOARD / CLAUDE EXECUTOR + GPT AUDITOR / PLAYER-FIRST CONTINUITY

**Model:** PRISM — Player Role & Individual Situation Model

### READ THIS DEEP HANDOFF NEXT

`docs/handoffs/NFL_HANDOFF_2026-10-22_PRISM_WEEK7_LIVE_BOARD_CONTROL_ROOM_PLAYER_FIRST.md`

Reusable restart prompt:

`docs/handoffs/NEXT_CHAT_PROMPT_2026-10-22_PRISM_WEEK7_LIVE_BOARD_CONTROL_ROOM.md`

Historical stacked checkpoints were preserved verbatim at:

`docs/handoffs/ARCHIVE_NFL_RESEARCH_HANDOFF_CHECKPOINTS_THROUGH_2026-10-22_PRE_WEEK7.md`

### ABSOLUTE ASSIGNMENT

PRISM is an **INDIVIDUAL-PLAYER PROJECTION MODEL**.

The unit of truth is the named player in his actual current football state for the exact market.

> **THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.**

> **ANOMALIES ARE QUESTIONS, NOT TRENDS.**

Position MAE/RMSE is downstream QA only. Never turn it into the optimization objective.

Do **not** scrap/rebuild PRISM. Current role-state, team-context and environment work is an **additive high-resolution arbitration layer** over substantial qualified player-specific science.

Always distinguish:
- `SOURCE READY`
- `SOURCE BLOCKED`
- `PREDICTIVE CLOSED`
- `QUALIFIED`

A source block is not a predictive failure. A failed new candidate does not erase old qualified science.

### CONTROL-ROOM ROLES

Per user direction:
- **Claude = active executor / research lead**
- **GPT-5.6 = independent auditor / project-control**
- shared lane = GitHub Issue #535
- GPT prefixes: `[GPT AUDIT]`, `[GPT OBJECTION]`
- Claude prefixes: `[CLAUDE STATUS]`, `[CLAUDE QUESTION FOR GPT]`

Before substantive status/audit decisions:
1. read `AGENTS.md`;
2. read only this TOP checkpoint;
3. read the deep handoff above;
4. fetch newest Issue #535 comments;
5. reconcile live `main`, open PRs, exact-head Actions/artifacts.

### LIVE PRODUCTION STATE — WEEK 7 BOARD EXISTS AND IS DURABLE

Current `main` at handoff:
`c8f6f057db7bf6c81201984e2de179ca51c7e882`

PR #693 is merged:
- merge commit `4dc178b59da8708dbea978b8e54e753b1e2f75c5`
- fixes the real nflverse participation schema blocker (`possession_team` / `defenders_in_box`);
- strict box gating remained intact;
- the false intermediate-source provenance sidecar was removed before merge.

Canonical Week-7 live Full Slate:
- run `39849371821` — **SUCCESS**
- job `120071136220`
- head `4dc178b59da8708dbea978b8e54e753b1e2f75c5`
- artifact `12456298985`
- digest `sha256:d1a1fb48e5bb920c5291f35b7457b5870801b689751deb2d742f98c555603a85`
- snapshot mode: `LIVE_SPORTSBOOK_SNAPSHOT`
- workbook pricing status: `CURRENT`
- sportsbook remains downstream-only.

Pregame football availability:
- 14 games / 28 teams;
- 14 eligible, 0 withheld;
- state `NOT_YET_REQUIRED: 14`;
- sportsbook inputs used for availability = 0.

Week-7 live pricing:
- 396 football players;
- 328 priced players;
- 732 compact player-market rows;
- 1,447 exact bookmaker offer rows;
- 2,894 priced side rows;
- core prop identity unresolved rows = 0;
- final manual quarantine removed 0 rows;
- data quality: `FULL_SLATE_DATA_QUALITY_PASS_WITH_DECLARED_LIMITATIONS`.

Durable Week-7 paid boundary on `main` from archive commit `c8f6f057...`:
- `data/market_track_record/boards/2026_wk07.csv` — 2,894 data rows;
- `data/market_track_record/game_odds/2026_wk07.csv` — 252 data rows;
- `data/market_track_record/raw_props/2026_wk07.csv` — 1,447 data rows;
- all carry `source_run_id=39849371821` / source head `4dc178b59...`.

**Do not spend another OddsAPI credit unless the user explicitly authorizes it.** Reuse the durable Week-7 boundary whenever possible.

### CURRENT SCIENCE BOUNDARIES TO PRESERVE

- `PRISM_RETURN_STATE_NEXT_ROLE_PERSISTENCE_V1_QUALIFIED_STATE_ONLY` remains QUALIFIED.
- `PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY` remains SOURCE READY.
- RB carry-count identity bridge: SOURCE BLOCKED, not predictive failure.
- RB receiving team-context reset: SOURCE READY -> PREDICTIVE CLOSED.
- TE receiving team-context reset: SOURCE READY -> PREDICTIVE CLOSED.
- RB rushing Stage-1 participation-authority reset: SOURCE BLOCKED; no predictive outcomes opened.
- Qualified RB Stage-2 conditional allocation exists in research; do not falsely say PRISM cannot represent conditional allocation.
- Direct high-value-share V1 closure is narrow; do not generalize to “scoring-area usage is useless.”
- QB generic pass-yard narrowing remains retracted/closed; do not reopen on Week-6/7 noise.
- Dedicated anytime-TD science is **not certified**. ATD is not a process-backed projection market yet.

Week-7 production itself declares:
- M89/M90 QB mean authority active;
- QB C2 distribution consumed;
- WR-R15 active;
- TE-R5P active;
- RB P3 rushing authority = Week-1 only, so not active in Week 7;
- RB R22 receiving tail = Week-1 only, so not active in Week 7;
- direct WR/CB matchup rows unavailable and gated off, team-scheme coverage still available.

### CLAUDE ROADMAP / RESEARCH FRONTIER

User-approved order from #535:
- **B:** Stage-2 -> production wiring audit, read-only.
- **C:** team scoring-territory drive creation.
- **D:** applicability as a strictly-prior player property.
- **E:** opponent/environment transmission.

Do not subgroup-shop improved players. Freeze applicability criteria before outcomes.

At last control-room evidence, no committed final result proved lane B complete. Re-check #535 before asserting otherwise.

### OPERATIONS / AUDIT STATE

Accepted:
- Week-5 `PRESERVED_PRE_T75_REPLAY` is a valid replay-authority state when it restores acquisition-time **game eligibility only** while current player unavailability remains authoritative.
- Weeks 3/4/6 gate salvage is accepted as **team-layer evidence only**; player-market gate layer remains unknown.
- Week 6 is measured + team-layer clean, but full-gate incomplete and non-poolable with fully gated Weeks 1-2.
- PR #692 artifact-durability work is merged.
- PR #693 production blocker fix is merged.

Outstanding:
- stale workflow-contract CI assertion should be fixed without weakening the live-odds guard;
- `requirements.txt` PyArrow hardening remains separate work;
- truthful final-source TeamForm box provenance remains follow-up;
- PlayerForm semantic migration remains: explicit `alignment_slot`, `usage_rank`, `starter_authority`, migrate consumers one by one;
- #691 is stale/blocked and is superseded by this new canonical continuity checkpoint; do not blindly merge it;
- #690 is audit-cleared but still open against the research branch;
- PR #672 remains large open/draft research lineage; do not merge wholesale.

### BET-SLIP CONSTRUCTION CONTINUITY

Canonical rule:
`docs/research/MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md`

User wants slips built as:
1. PRISM football projection first;
2. compare against actual sportsbook line;
3. validate the named player's exact role/current state/team/opponent/game environment;
4. buy down to an alt threshold with real cushion;
5. construct coherent game-script tickets;
6. use **Safe / Moderate / High / Lottery** tiers;
7. for a “home-run” ticket, increase payout through multiple individually defensible legs, not random TD scorers or nonsense thresholds.

Week-6 SF-WAS manual construction used this method and the user later reported **every placed bet/leg hit**. Treat that as a successful prospective anecdote/process example, **not scientific validation**. The user also corrected that the actual FanDuel payout odds were lower than our mental estimate; next time use the exact displayed FD prices/screenshots rather than estimating SGP payout.

### QUEUE GOVERNANCE

- Betting defaults LOCKED unless the user explicitly shifts into slip construction.
- No paid OddsAPI without explicit authorization.
- Sportsbook downstream only.
- Do not reopen closed science merely to get numbers.
- No names/fuzzy/manual identity bridges where stable provider ID is required.
- No production mutation from source/research work alone.
- Keep old QUALIFIED/CLOSED/SOURCE_BLOCKED results intact.
- On “keep going,” continue the exact active canonical lane after live reconciliation.
