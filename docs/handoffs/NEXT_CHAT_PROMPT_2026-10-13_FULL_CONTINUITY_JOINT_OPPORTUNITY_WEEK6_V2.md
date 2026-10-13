Pick up my NFL project seamlessly from the latest canonical GitHub state.

Repo: dkaps6/imtiredofthis.

Do NOT ask me to re-explain anything. GitHub is canonical. Do NOT recursively
read old handoffs. Do NOT restart completed research. Do NOT spend a paid
OddsAPI credit without my explicit authorization. I need this to feel like the
exact same conversation continuing.

Read in this exact order:

1. AGENTS.md
2. ONLY the newest TOP checkpoint in CURRENT_NFL_RESEARCH_HANDOFF.md
3. docs/handoffs/NFL_HANDOFF_2026-10-13_FULL_CONTINUITY_JOINT_OPPORTUNITY_WEEK6_V2.md
4. docs/research/WEEKS1_5_CONTINUITY_WEEK5_POSTMORTEM_SYNTHESIS_V1.md
5. docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_HISTORICAL_RESULT.md
6. docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_PROSPECTIVE_SHADOW_MECHANICS_RESULT.md
7. docs/research/WEEK6_INDIVIDUAL_OPPORTUNITY_LOCK_READINESS_RESULT.md

Then reconcile live:
- main
- research-individual-opportunity-roadmap-2026-10-08
- PR #672
- Issue #535 newest comments
- exact-head GitHub Actions

Critical continuity:

- Production main was 812ac2cde79fe3846174613a2f2777e57b34b739 at handoff.
- Research is isolated on PR #672.
- Canonical Week-5 preserved Full Slate remains run 37860592615 / artifact
  11585973062. Do NOT acquire new OddsAPI data unless I explicitly approve it.
- Week 5 selected board was 199-151, +26.17u, but raw mean accuracy is NOT
  solved: model MAE 18.93 vs preserved median-line MAE 17.84.
- Week-5 top-40 misses were 19 efficiency-dominant / 14 opportunity-dominant /
  7 mixed. Opportunity uncertainty and efficiency tails are separate problems.
- User-facing betting/ticket outcomes are downstream and must NOT retune football
  science. Watson was a real projection miss plus cross-ticket concentration.
- Individual opportunity-share volatility is now THREE-SEASON CONFIRMED.
- Excess role volatility beyond finite-count sampling survives for RB carry,
  RB target, and WR target. TE target fails the current mechanism gate.
- Mean-identical Beta-Binomial historical PASS:
  - RB carry share
  - WR target share
  RB target failed closed. TE not authorized.
- MAJOR PASS: joint/team-conserving Dirichlet allocation for:
  - RB carry share
  - WR target share
  Historical authority: run 38401069806 / artifact 11815490905 /
  digest sha256:2b336114dc5ef0145c0a40706266da850a0ff1d25c8935958481b996333fce29.
  It preserves exact canonical player means and exact team opportunity totals.
- Prospective joint-shadow mechanics are certified:
  run 38402018307.
- Week-6 lock was last verified FAIL-CLOSED because exact Week-6 roster authority
  was unavailable. Week-5 stats/snaps were complete. NEVER substitute Week-5
  roster identity.
- FIRST operational move: check whether exact Week-6 weekly roster authority is
  now available.
  - If YES: dispatch/freeze the existing Week-6 opportunity lock pack and joint
    Dirichlet RB-carry/WR-target shadow before any Week-6 outcomes.
  - If NO: stay fail-closed and immediately continue the historical
    INDIVIDUAL EFFICIENCY-TAIL UNCERTAINTY lane.
- The next historical efficiency lane must be frozen before scoring, must be
  player-specific and strictly prior, and must NOT tune to Watson/Stafford/
  Coleman/Kincaid or Week-5 misses.
- QB coach/OC intent is a passive prospective side lane only. Historical
  semantic automation is closed. Do not let it take over the roadmap.
- Direct player-specific matchup transforms already failed closed:
  WR/TE depth-YPT V1 and RB run-location YPC V1. Do not retune/rescue them.
- Protect QB M89/M90, QB C2, M38 WR1, WR-R15, TE-R5P, RB receiving-room
  historical improvement, and exact WR/TE trajectory eligibility.
- Current PR-side Backtest W1-2 and Replay Paid Full Slate reds are known
  Polars/pyarrow/NumPy environment failures, not football-science failures and
  not new odds pulls. Repair them mechanically only when useful.

Model vision:

Project every player as an individual football player in his exact game using
current role, team opportunity, individual share, individual efficiency,
offense, opponent/context, injuries/personnel, and player-specific uncertainty.

Do NOT turn the model into generic position averages or generic
defense-v-position multipliers.

Take the wheel and keep moving autonomously. Give me concise progress updates
after real milestones. Do not stop at a roadmap summary; execute the next legal
research step.
