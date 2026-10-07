Pick up my NFL project seamlessly from the previous chat.

GitHub is canonical.
Repo: dkaps6/imtiredofthis.

Do not ask me to re-explain anything.
Do not restart research.
Do not recursively read old handoffs.
Do not reopen closed science.
I need this to feel like the exact same chat continuing.

Start by reading, in this exact order:

1. AGENTS.md
2. CURRENT_NFL_RESEARCH_HANDOFF.md — newest top checkpoint only
3. docs/handoffs/NFL_HANDOFF_2026-10-07_PLAYER_CENTRIC_ALL_POSITIONS_CURRENT.md
4. docs/research/PLAYER_INDIVIDUALIZATION_AUDIT_V1_FULL_STACK_ADDENDUM.md
5. docs/research/RB_PLAYER_STATE_ALLOCATION_SHADOW_V1_WEEK5_LOCK.md
6. docs/research/PLAYER_TARGET_SHARE_TRAJECTORY_V1_RESULT.md
7. docs/research/PLAYER_TARGET_SHARE_TRAJECTORY_SHADOW_V1_WEEK5_LOCK.md
8. docs/research/PLAYER_TARGET_DEPTH_DISPERSION_V1_RESULT.md
9. Issue #535 from comment 6029157126 onward
10. Query GitHub live for current main, relevant branches, PRs, and Actions before doing anything.

My immediate question from the prior chat was: before we run an all-player/all-position replay, which positions are actually finished and which still need player-level work?

The current answer should be:
- QB is largely buttoned up for this player-individualization phase; do not reopen generic QB mean.
- RB is the largest unfinished position. Preserve the Week-5 prospective RB room-allocation shadow and do not violate M96.
- WR/TE opportunity player-state is substantially buttoned up: target-share trajectory is historically CONFIRMED and a Week-5 prospective shadow is already frozen.
- WR/TE efficiency difficulty now has a confirmed football-state correlate: individual target-depth dispersion. The mean-neutral distribution shadow branch exists but has NOT been implemented yet.

Do not start the all-position replay yet if those remaining pieces are unfinished.

Next execution sequence:
1. Freeze and implement the mean-neutral player target-depth distribution/uncertainty shadow on branch research-player-target-depth-distribution-shadow-v1. Preserve exact receiving-yard means, entitlement, team volume, M38, WR-R15, TE-R5P, QB science, and sportsbook separation.
2. Then resolve the remaining RB player-level scope scientifically without violating the M96 retrospective stop or reopening closed RB receiving mean / width families.
3. Once QB/RB/WR/TE are explicitly buttoned up, build the all-position replay.

Important nuance on the four live 2026 weeks:
- exact Target Share Trajectory V1 needs 4 prior same-season games, so Weeks 1-4 cannot exercise that exact frozen feature. Do not weaken it after seeing outcomes.
- target-depth dispersion can potentially be replayed on 2026 Weeks 1-4 using strictly-prior 2025 history.
- the exact trajectory transformation can be integration-tested retrospectively on historical Week-5+ games.

No paid OddsAPI pull without my explicit authorization.
Never claim a run is active unless GitHub Actions shows it is active.
Keep hard-checkpointing results to GitHub so UI timeouts cannot lose continuity.

Continue working rather than just summarizing.