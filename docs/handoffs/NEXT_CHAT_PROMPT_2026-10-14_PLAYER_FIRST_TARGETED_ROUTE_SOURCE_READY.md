Pick up my NFL project seamlessly from the latest canonical GitHub state.

Repo: dkaps6/imtiredofthis.

GitHub is canonical. Do NOT ask me to re-explain anything. Do NOT restart research. Do NOT recursively read old handoffs. I need this to feel like the exact same conversation continuing.

Read in this exact order:

1. AGENTS.md
2. ONLY the newest TOP checkpoint in CURRENT_NFL_RESEARCH_HANDOFF.md
3. docs/handoffs/NFL_HANDOFF_2026-10-14_PLAYER_FIRST_TARGETED_ROUTE_SOURCE_READY_CURRENT.md
4. docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_FREEZE.md
5. docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_SOURCE_RESULT.md
6. docs/research/WR_PLAYER_OPPORTUNITY_UNCERTAINTY_V1_RESULT.md
7. docs/research/TE_PLAYER_OPPORTUNITY_UNCERTAINTY_V1_RESULT.md
8. newest comments on Issue #535
9. live PR #672 / active research branch / exact-head GitHub Actions

Then reconcile live before any write:
- main
- research-individual-opportunity-roadmap-2026-10-08
- PR #672
- Issue #535 latest comments
- exact-head Actions

Critical continuity:

- This is an INDIVIDUAL-PLAYER projection system. Do not treat position-wide mean MAE as the hypothesis grain.
- Position-level metrics are guardrails, not proof that individual-player mechanisms are exhausted.
- Canonical state remains PLAYER_INDIVIDUALIZATION_PARTIAL.
- WR and TE named-player opportunity uncertainty are CONFIRMED and beat pooled positional uncertainty.
- RB same-player yard-difficulty width remains historically qualified.
- Broad 11/12/OTHER personnel-conditioned opportunity is closed.
- Player x environment response V1 is closed/support-limited.
- WR-R3 exact combined candidate is closed, but do NOT misinterpret that as no WR individual signal.
- Do not reopen generic matchup/position-average/coverage/target-depth/receiver-conversion/efficiency-tail/Q90/NGS/WR-CB/RB-retune lanes unchanged.
- No paid OddsAPI without explicit authorization.
- No production mutation from research.

Newest active lane:

PLAYER_TARGETED_ROUTE_MATCHUP_SOURCE_READY

Canonical source authority:
- run 38637901016 SUCCESS
- job 116105672352
- exact scored head 159b8213a734e3f4ee16fc86fbb51950099f3323
- artifact 11928258474
- digest sha256:b957ff0b37f0b2d8f45232bcd44657928cecc75f5f5e6d1c292ee8e35729228c

Source support:
- 46,666 valid WR/TE targeted-route rows
- 554 valid WR/TE players
- 100% route/PBP join rate on joined source
- 93.18% stable receiver identity coverage on taxonomy-eligible route rows
- 10 route families with >=1,000 pooled rows
- 32/32 defenses supported by Week 8 in each scored season
- 2023 eligible players: 257
- 2024 eligible players: 246
- 2025 eligible players: 253

CRITICAL SEMANTICS:
nflverse participation route is ONLY the route label for the primary/targeted receiver on the play.
It is NOT total routes run, route participation, TPRR, YPRR, or non-target route state.
Do not infer WR-CB assignment.

Immediate next action:
BUILD AND RUN STAGE B EXACTLY AS ALREADY FROZEN.

For each target WR/TE:
- same-player trailing max 8 eligible prior games
- minimum 4 strictly-prior player-games
- targeted-route mix only
- target opponent trailing max 8 defensive games
- strictly-prior opponent route-family completion rate / yards per target / rec yards per target
- compare to strictly-prior league route-family neutral
- derive player-specific matchup_delta = weighted opponent expectation - weighted league neutral

Primary outcomes, WR and TE separately:
- next-game receiving yards per target
- next-game catch rate
- next-game receiving yards

Validation is frozen:
- 2023 development
- 2024 independent confirmation
- 2025 independent confirmation
- player-clustered CI required
- game-clustered CI required
- must show it is not merely a renamed target-depth/air-yard signal
- position aggregate alone cannot qualify it

Do not change route groupings, windows, support thresholds, player filters, or matchup construction after seeing results.

If Stage B passes, freeze a separate full-stack integration candidate beyond M38/WR-R15 or TE-R5P.
If it fails, close the exact route-family interaction with no rescue.

Take the wheel and continue autonomously. Keep updates concise and show me real GitHub progress/results.
