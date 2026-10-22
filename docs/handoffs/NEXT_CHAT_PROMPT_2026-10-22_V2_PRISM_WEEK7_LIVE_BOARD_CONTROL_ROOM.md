Pick up my NFL PRISM project seamlessly from the newest canonical GitHub checkpoint.

Repo: `dkaps6/imtiredofthis`.

Do NOT ask me to re-explain anything. GitHub is canonical. This needs to feel like the exact same assistant continuing, with the same player-first understanding and the same Claude/GPT control-room roles.

READ IN THIS EXACT ORDER:

1. `AGENTS.md`
2. ONLY the newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. `docs/handoffs/NFL_HANDOFF_2026-10-22_V2_PRISM_WEEK7_LIVE_BOARD_CONTROL_ROOM_PLAYER_FIRST.md`
4. newest comments in GitHub Issue #535
5. live `main`, relevant open PRs, and exact-head Actions/artifacts

Do NOT recursively read old handoffs unless the 2026-10-22 deep handoff explicitly points to a specific source file.

NON-NEGOTIABLE ASSIGNMENT:
- PRISM is an INDIVIDUAL-PLAYER PROJECTION MODEL.
- The named player in his actual current football state for the exact market is the unit of truth.
- Position MAE/RMSE is downstream QA, NOT the objective.
- “THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.”
- “ANOMALIES ARE QUESTIONS, NOT TRENDS.”
- Do NOT scrap or rebuild PRISM. Current role/team/environment work is additive.
- Preserve exact distinctions among SOURCE READY / SOURCE BLOCKED / PREDICTIVE CLOSED / QUALIFIED.
- A source block is not a predictive failure.
- Sportsbook lines are downstream comparison only.

CONTROL ROOM:
- Claude = active executor / research lead.
- GPT-5.6 = independent auditor / project control.
- Issue #535 is the shared asynchronous lane.
- If Claude asked GPT a question, answer it there.
- Audit exact commits/runs/artifacts; do not take over Claude's model branch by default.

CURRENT LIVE STATE YOU MUST RECONCILE, NOT ASSUME:
- main at the final pre-handoff refresh: `32b5cbf41f551b452a89c457c1b32e497bd934da`
- PR #693 merged at `4dc178b59da8708dbea978b8e54e753b1e2f75c5`
- Week-7 live Full Slate run `39849371821` SUCCESS
- artifact `12456298985`, digest `sha256:d1a1fb48e5bb920c5291f35b7457b5870801b689751deb2d742f98c555603a85`
- Week-7 paid board/game-odds/raw-props are durable in `data/market_track_record/`
- do NOT make another paid OddsAPI call unless I explicitly authorize it.

IMPORTANT:
The 2026-10-22 deep handoff contains the full science state, Week-7 board state, Week-5 replay ruling, gate-salvage limits, open maintenance items, current PR state, and the bet-slip construction workflow including the successful Week-6 SF-WAS manual parlay night.

Betting/slip construction is a temporary user-directed lane, not the default research state. If I ask for slips, use `docs/research/MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md` and build Safe / Moderate / High / Lottery from PRISM projection + named-player football validation + actual FanDuel thresholds/prices. Do not use random TD legs to manufacture payout.

Start by reconciling live GitHub and then tell me, concisely, what changed since the handoff and what the exact active next action is. Do actual work; do not give me a generic status speech.


FINAL CONTINUITY NOTE:
- At the end of the prior chat, no Claude comment newer than Issue #535 comments `6353044832` / `6353046193` was visible.
- Relevant open PRs were #672 (draft/stale research lineage), #690 (audit-cleared cert-order repair; SOURCE_BLOCKED science unchanged), and #691 (stale/superseded continuity cleanup).
- PR #693 was already merged and Week-7 live Full Slate run `39849371821` had succeeded and been durably archived.
- Re-fetch #535/live GitHub immediately because this note is a checkpoint, not a substitute for live reconciliation.
- The user expects seamless continuation and does not want to reteach the model, the player-first philosophy, or the Claude/GPT role split.
