Pick up my NFL project seamlessly from the latest canonical GitHub state.

Repo: dkaps6/imtiredofthis.

Do NOT ask me to re-explain anything. GitHub is canonical. I need this to feel like the exact same conversation continuing. Do NOT recursively read old handoffs. Do NOT restart completed research. Do NOT spend a paid OddsAPI credit without my explicit authorization.

Read in this exact order:

1. AGENTS.md
2. ONLY the newest TOP checkpoint in CURRENT_NFL_RESEARCH_HANDOFF.md
3. docs/handoffs/NFL_HANDOFF_2026-10-15_WEEK6_SEA_DEN_TICKETS_WR_TE_TRUST_CONTINUITY_CURRENT.md

Then reconcile live GitHub before doing any write:

- main
- research branch research-individual-opportunity-roadmap-2026-10-08
- draft PR #672
- Issue #535 newest comments/state
- exact-head GitHub Actions relevant to the current research state

Critical continuity:

- Production Week 6 is already complete. main is around 9b70b12e2fb4bc39f08556f244370068aad9c47b.
- Exactly ONE Week-6 paid OddsAPI acquisition was authorized and already used:
  run 38791377571 / artifact 11999923880 / digest sha256:0916cd1cbec93f9a2529190b3df7c39172eee011a31eb9df429a35cdb9b41f61.
- Final green Week-6 replay reused that paid artifact without refetch:
  run 38792341891 / artifact 12000113891.
- DO NOT pull Week-6 OddsAPI again.

RIGHT NOW we were working on tonight's SEA-DEN game and my ticket creation system.

Use the actual repo process:
docs/research/MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md

I want the normal ticket ladder:
- Core / safe
- Core + juice / moderate
- Aggressive
- High-ceiling / lottery

Do not just sort raw edge. Use our model, player-level football reasoning, role/availability, real/main line vs alt-line cushion, model caveats and intentional ticket construction.

The preserved SEA-DEN core model board includes:
- Sam Darnold U223.5 pass yards, model 212.6
- Cooper Kupp O21.5 receiving, model 28.1
- Evan Engram O20.5 receiving, model 25.8
- Courtland Sutton U3.5 receptions, model 2.56
- Troy Franklin O18.5 receiving, model 22.2
- Horton O3.5 receiving, model 16.1

Previously built working tickets:
SAFE: Darnold U249.5 + Kupp 20+ + Engram 15+
MODERATE: Darnold U239.5 + Kupp 20+ + Engram 20+ + Sutton U4.5 receptions
AGGRESSIVE: Darnold U229.5 + Kupp O21.5 + Engram O20.5 + Sutton U3.5 receptions + Franklin 15+

We had NOT faithfully recovered/finalized the lottery ticket before the chat ended. I specifically told you to use the GitHub ticket-construction process to build it rather than invent it from memory. Resume there.

ATD exception for tonight:
I explicitly allow cautious ATD suggestions, but ATD dedicated model science is NOT certified. Use football knowledge, player role/red-zone/goal-line context, matchup/game environment and the actual offered odds. Label ATD as judgment-only, do not invent model probabilities, and do not make it the foundation of the safe ticket. Reconcile current prices before using an ATD.

Do NOT lose the research lane while doing tonight's game.

Newest research state includes BOTH WR and TE player-level trust work:

TE:
- TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED
- run 38821176218
- Week-6 TE trust shadow frozen prospectively:
  run 38822346088 / artifact 12013018536
- 25 HIGH / 52 MIDDLE / 26 LOW / 2 unsupported
- Week-6 outcomes read 0.

WR — this finished right before the old chat maxed out:
- WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED
- run 38823063793 SUCCESS
- artifact 12013153664
- digest sha256:12fd0de1dc3ebceee4d356669d59f30d96776d8c0800d1fb3ac6a293af457b7c
- exact final WR-R15 parity: 3,224 rows / 211 WRs
- pooled trust association +0.28853
- HIGH_TRUST abs error 14.61 yd vs overall 23.25 vs LOW_TRUST 30.47
- positive player/game clustered confirmation.

Week-6 WR trust shadow also froze successfully:
- WEEK6_WR_TRUST_PREGAME_LOCK_FROZEN
- run 38823520954 SUCCESS
- artifact 12012884262
- digest sha256:e5bc113e507ded651ecc29080c7fdbe4fd55182d1d039d0dd14057ada4b7bb38
- row digest sha256:81d9e950b96d7ac9d5699eba8490fc741a5aab2c938ef29287dc261534346b8f
- 152 WR cohort / 141 supported
- HIGH 35 / MIDDLE 70 / LOW 36 / unsupported 11
- Week-6 outcomes read 0
- sportsbook 0 / OddsAPI 0 / production unchanged.

Canonical model status remains PLAYER_INDIVIDUALIZATION_PARTIAL.

Do not reinterpret HIGH/LOW trust as over/under or bet/fade direction. These are reliability labels around unchanged projections.

Research anti-retest still applies. Do not reopen closed QB generic mean/YPA, WR route mean integration, WR allocation-only bridge, TE width/Dirichlet mappings, target-depth symmetric distribution transform, or other closed lanes.

If I am asking about tonight's bets, prioritize SEA-DEN ticket construction first. If I tell you to go back to research, resume from the WR + TE trust-lock state and anti-retest Issue #535 before opening a genuinely distinct player-level frontier.

Take the wheel and continue.