Pick up my NFL project seamlessly from the previous chat.

GitHub is canonical. Repo: dkaps6/imtiredofthis.

Do not ask me to re-explain anything and do not recursively read old handoffs.

Start by reading, in this exact order:

1. AGENTS.md
2. CURRENT_NFL_RESEARCH_HANDOFF.md — only the newest top checkpoint
3. docs/handoffs/NFL_HANDOFF_2026-09-26_WEEK3_FULL_SLATE_REPAIRED_SYSTEMS_AUDIT_CURRENT.md
4. Issue #535 comments from 5850416444 onward
5. Query GitHub live for current main, relevant branches, PRs and Actions runs before doing anything

Important live checkpoint at handoff creation:

- main was 4c0e9e25f0957877820cade6897cf977d265533a
- PR #645 was merged
- canonical no-live-odds Full Slate run 36283114203 passed on main
- Repo CI 36283114216 passed
- the earlier 30-team/32-team concern is closed
- the real Week-3 Full Slate blocker was a Bayesian rule-authority mismatch in LAR target redistribution and is repaired/merged
- Barion Brown non-core unrostered sportsbook quarantine fix is also merged
- next actual betting-board gate is a controlled fetch_live_odds=true Full Slate, which spends OddsAPI credits; do NOT launch it unless I explicitly authorize it

Keep these frozen research lanes intact:
- RB Vacancy Opportunity V1 for DEN/PIT
- DEN public-intent label: ROTATION_PRESERVED_NO_CLEAR_SUCCESSOR_CONCENTRATION
- PIT label: WARREN_LEAD_BACK_LEAN_WITH_DEPTH_SUPPORT
- Receiving Rule Semantics Week-3 prospective A0B0/A1B0/A0B1/A1B1 lock
- no Week-3 postgame redesign
- sportsbook stays downstream

Also know:
- RB Rush+Receiving Conservation V2 is production-active and is the clearest recent live model improvement
- Availability -> Opportunity Rule-Order Gap is confirmed
- middle_open unit semantics and slot-alignment loss are structurally confirmed, but not accuracy-qualified; their Week-3 prospective cells are frozen
- Discrete Count Mean Alignment V1 is research-qualified for a separate integration test
- Rush-att zero-MC ensemble transmission is a confirmed open architecture issue; next valid action there is allocation-lineage audit, not a guessed repair
- do not reopen closed WR participation, TE Width V2, Rush Pool, post-ensemble reconciliation, generic copula, or M96 retrospective RB families

Work autonomously, keep Issue #535 and GitHub handoff current, avoid repeating completed experiments, and distinguish production-active changes from diagnostics/research-only results.

First tell me the live GitHub state you found and the exact next action, then proceed.