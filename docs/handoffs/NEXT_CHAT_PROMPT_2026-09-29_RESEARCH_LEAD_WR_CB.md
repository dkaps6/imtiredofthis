Pick up my NFL project seamlessly from the previous chat.

GitHub is canonical. Repo: dkaps6/imtiredofthis.

Do not ask me to re-explain anything. Do not restart completed research. Do not recursively read old handoffs. I need this to feel like the same research lead continuing.

Start in this exact order:

1. Read AGENTS.md.
2. Read CURRENT_NFL_RESEARCH_HANDOFF.md — newest top checkpoint only.
3. Read docs/handoffs/NFL_HANDOFF_2026-09-29_RESEARCH_LEAD_WR_CB_CURRENT.md.
4. Read Issue #535 from comment 5899500522 onward, especially:
   - 5899645764
   - 5899710345
   - 5899966307
   - 5900122307
   - 5901422664
   - 5901433362
   - 5901608387
   - 5901927972
   plus GSIS checkpoint 5881240970.
5. Query GitHub live before editing anything:
   - current main
   - research-week3-postmortem-execution-v1
   - research-wr-cb-free-archive-v1 / PR #665
   - research-market-snapshot-history-v1 / PR #663
   - research-rb-opponent-defender-injury-readiness-v1
   - repair-specialist-rng-isolation-v1
   - PR #662
   - current Actions

Critical reconciled production state:
- live main is deea0f68202c9ab6e85fa616f7444d0008ee10af at the handoff checkpoint;
- PR #664 merged and the legacy static coverage_penalty() heuristic is REMOVED from production;
- do not trust stale Issue wording that says it is still active;
- direct player-level WR-CB assignment remains fail-closed/gated off when unavailable;
- do not pay for WR-CB data;
- do not re-add the old 0.92/0.94/1.06/1.04 coverage multipliers.

Immediate research priority:
Resolve the contradictory WR-CB free-source state by inspecting PR #665 live, not by trusting comment chronology.

PR #665 is trying to recover a free FantasyAlarm historical WR/CB archive spanning 2021-2026. Determine whether the live source audit actually clears the scientific source gate:
- free
- explicit factual WR↔CB pairing/alignment
- historical week/game grain
- stable/machine-reproducible retrieval
- temporal integrity
- adequate completeness
- clean player/team identity

If it fails, close the source lane cleanly and keep direct WR-CB production gated off.
If it clears, proceed research-only to a newly frozen true-assignment test using factual pairing/alignment only. Do NOT restore the deleted legacy heuristic and do NOT use editorial Safe/Moderate/Risky grades as model features.

Claude coordination:
A branch exists named research-rb-opponent-defender-injury-readiness-v1. Query it and Issue #535 before duplicating work. Claude owns opponent-defender injury source readiness; GPT should cross-audit it.

Other current lanes:
- Week 3 / Weeks 1-3 postmortem is fully closed: 629-611, 50.7%, -40.72u.
- RB-PD2 = HOLD at 1/8 weeks and 46/400 rows.
- authority line-crossing primary test failed; do not create a crossing filter.
- authority strengthened-vs-weakened is frozen prospectively for Week 4+ only.
- RB route-volume live source exists; historical/live parity not yet cleared; prospective capture tooling exists.
- PR #663 is append-only downstream market snapshot / T30 CLV infrastructure. Query live because concurrent fixes may have landed. Earlier CI failure was mechanical; replay red was expired artifact run_35282021679.
- RNG branch has narrow semantic-seed-label patch 30333ec2...; do not call PASS until exact frozen fingerprint validation is observed.
- GSIS PR #662 remains research-only; raw GSIS stays private.
- no paid OddsAPI run without my explicit approval.

Stay in implementation/research mode. Take the research lead. Use Issue #535 as the shared lab notebook with Claude. Do not ask me what to do next if the repo already makes the next step clear.