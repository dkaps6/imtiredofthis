Pick up my NFL project seamlessly from the latest canonical GitHub state.

Repo: dkaps6/imtiredofthis.

Do NOT ask me to re-explain anything. GitHub is canonical. Do NOT recursively read old handoffs. Do NOT rerun completed research merely to reacquire context. I need this to feel like the exact same conversation continuing.

Read in this exact order:

1. AGENTS.md
2. ONLY the newest TOP checkpoint in CURRENT_NFL_RESEARCH_HANDOFF.md
3. docs/handoffs/NFL_HANDOFF_2026-10-13_WEEK5_POSTMORTEM_JOINT_OPPORTUNITY_WEEK6_FRONTIER_CURRENT.md
4. docs/research/WEEKS1_5_CONTINUITY_WEEK5_POSTMORTEM_SYNTHESIS_V1.md
5. docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_HISTORICAL_RESULT.md
6. docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_PROSPECTIVE_SHADOW_MECHANICS_RESULT.md
7. docs/research/WEEK6_INDIVIDUAL_OPPORTUNITY_LOCK_READINESS_RESULT.md

Then reconcile LIVE before doing any write:
- main
- branch research-individual-opportunity-roadmap-2026-10-08
- PR #672
- Issue #535 newest comments
- exact-head GitHub Actions

Critical continuity:

- The goal is still true INDIVIDUAL PLAYER projections: role/usage + team opportunity + player share + individual efficiency + player-specific matchup/context + uncertainty. Do not drift back to generic position blending.
- Week-5 selected board was 199-151, 56.86%, +26.17u, but raw model MAE still lost to the preserved market line. The model is NOT solved.
- Week-5 top-40 misses were 19 efficiency-dominant / 14 opportunity-dominant / 7 mixed.
- Ticket construction is a separate downstream defect. Watson was a true projection miss plus cross-ticket concentration; BUF-LAR core lost by Kyren Williams 49 vs 50, a near-threshold miss. Do not use user-ticket outcomes to retune football science.
- Strongest current PASS: three-season player-specific opportunity-share uncertainty. Joint/team-conserving Dirichlet allocation PASSED 2023/2024/2025 for RB carry share and WR target share while preserving exact means/team totals.
- Historical joint authority: run 38401069806, artifact 11815490905, digest sha256:2b336114dc5ef0145c0a40706266da850a0ff1d25c8935958481b996333fce29.
- Prospective joint-shadow mechanics: run 38402018307 SUCCESS. Manual runner: .github/workflows/research-future-week-joint-opportunity-shadow-v1.yml.
- Do NOT extend this family to RB target or TE target; those failed their frozen gates.
- Week-6 lock is currently fail-closed ONLY because exact Week-6 weekly roster authority was unavailable. Week-5 stats/snaps are complete. Never substitute the Week-5 roster.
- FIRST check whether exact Week-6 roster authority is now available. If it is, run/freeze the existing opportunity lock pack and joint RB-carry/WR-target shadow BEFORE Week-6 outcomes.
- If Week-6 roster is still unavailable, keep it fail-closed and immediately continue the next non-time-gated root cause: individual efficiency-tail uncertainty, after an anti-retest audit and a frozen historical diagnostic.
- QB coach/OC intent is a useful passive prospective side lane, not the main roadmap. Historical automation is closed. V1L Monday refresh still had 0 retained sources; do not loosen source rules.
- Player-depth WR/TE direct YPT matchup and RB run-location direct YPC matchup both FAILED CLOSED. Do not tune/rescue/repackage them.
- No new paid OddsAPI pull without my explicit authorization.
- Production Week-5 preserved authority remains run 37860592615 / artifact 11585973062 / digest sha256:bf52aa4bc30bf6e8cc6029a0a86c0f33cdfaa09f587c38a138f23cef6b7436f4.
- Research branch PR-side Backtest W1-2 and Replay Paid Full Slate red checks are currently Arrow/NumPy compatibility failures, not scientific failures. Repo CI / historical input validation / clean benchmark rebuild are green. Fix those mechanically when appropriate; do not confuse them with model science.

Take the wheel and keep moving. Do not stop at a status summary unless something genuinely requires my decision.
