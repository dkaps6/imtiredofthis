Pick up my NFL project seamlessly from the previous chat.

GitHub is canonical. Repo: dkaps6/imtiredofthis.

Do not ask me to re-explain anything. Do not restart research. Do not recursively read old handoffs. I need this to feel like the exact same chat continuing.

Start by reading, in this exact order:

1. AGENTS.md
2. CURRENT_NFL_RESEARCH_HANDOFF.md — read only the newest top checkpoint
3. docs/handoffs/NFL_HANDOFF_2026-09-29_RESEARCH_LEAD_WR_CB_CLV_CURRENT.md
4. Issue #535 from comment 5899500522 onward, especially:
   - 5899645764
   - 5899710345
   - 5899966307
   - 5900122307
   - 5901422664
   - 5901433362
   - 5901608387
   - and anything newer
5. Query GitHub live for current main, PR #663, PR #665, PR #662, research-week3-postmortem-execution-v1, repair-specialist-rng-isolation-v1, and current Actions before doing anything.

Critical current state:

- canonical main is deea0f68202c9ab6e85fa616f7444d0008ee10af unless live GitHub has moved;
- Week 3 and Weeks 1-3 postmortem are closed: 629-611, 50.7%, -40.72u;
- do not rerun closed Week-3 lanes;
- projection-authority line-crossing primary hypothesis failed; do not create a crossed-line filter;
- same-side strengthened vs weakened is discovery-only and already has a Week-4+ frozen forward contract;
- RB-PD2 remains HOLD at 1/8 weeks and 46/400 rows;
- route-volume live data exists, but historical/live weekly parity is not cleared; prospective capture only;
- specialist RNG patch 30333ec2... still needs exact frozen fingerprint validation;
- no paid OddsAPI run without my explicit approval.

WR/CB CURRENT STATE — IMPORTANT:

- Direct player-level WR↔CB assignment is production-gated off when unavailable. Do not invent assignments.
- PR #664 is MERGED on main. The grandfathered coverage_penalty() heuristic and its static 0.92/0.94/1.06/1.04 multipliers were removed from production.
- Do NOT restore coverage_penalty().
- Issue comment 5901927972 contains a stale statement that the team-level heuristic is still active. Trust current main and the new handoff instead.
- We then found a potentially useful FREE historical FantasyAlarm WR/CB archive spanning 2021-2026.
- PR #665 is the research-only source-acquisition/audit lane. It extracts factual pairings/alignment only; editorial Safe/Moderate/Risky grades are NOT model-eligible.
- Do not promote WR-CB science until PR #665 proves source completeness, timing, identity, and semantic stability.
- If that gate clears, the next legitimate science is the previously source-blocked TOP_WEAPON_ESCAPE_HATCH using strictly prior observed assignments and untouched confirmation data. No paid source required.

CLV CURRENT STATE:

- PR #663 is the append-only market-snapshot/CLV architecture.
- It must preserve exact odds fetch time and event kickoff.
- Only same-book snapshots within 30 minutes of kickoff may be called CLV.
- It never triggers an odds fetch and must not create OddsAPI spend.
- PR #663 is still draft/open and was not green at the handoff. Diagnose the actual latest head/logs first; do not assume the last failure still applies or blindly rerun old work.

Claude coordination:

- Issue #535 is the shared lab notebook.
- Claude was assigned opponent-injury propagation and cross-audit of route-volume / WR-CB source semantics.
- Read any new Claude comments before proceeding.
- Cross-audit rather than duplicate his work.

Exact next action:

1. Finish PR #663 mechanically to a green state without changing its frozen CLV contract or buying odds.
2. Continue PR #665 source-readiness validation; do not fit a matchup model until the source gate clears.
3. Cross-audit Claude's opponent-injury findings if posted.
4. Verify RNG exact-parity repair.
5. Preserve Week-4+ prospective captures and continue only genuinely new science.

Stay in implementation/research mode and leave a GitHub trail in Issue #535.
