Continue my existing NFL Stuff project from the canonical GitHub paper trail.

Repo: dkaps6/imtiredofthis

GitHub is canonical over chat memory. Do not make me re-explain prior work.

READ ONLY, in this order, to conserve context:
1. AGENTS.md
2. the TOP checkpoint of CURRENT_NFL_RESEARCH_HANDOFF.md
3. docs/handoffs/NFL_HANDOFF_2026-09-25_RECEIVER_ROOM_FRONTIER_CURRENT.md
4. Issue #535 from comment 5834428368 onward, especially final disposition comment 5837631919
5. live main/current research branch state

Do NOT recursively load older handoffs unless the current handoff explicitly points you there.

CURRENT SCIENTIFIC STATE

Receiver Room Targets-Per-Play V1 was promising on untouched 2022-2023 but FAILED CLOSED on the unchanged 2024-2025 confirmation.

Discovery formula:
R_g_play = sum(strict-prior room targets) / sum(strict-prior offensive plays)
candidate room targets = projected offensive plays * R_g_play

2022-2023 discovery:
- run 36172644864
- artifact 10880948137
- result docs/research/RECEIVER_ROOM_TARGETS_PER_PLAY_V1_RESULT.md
- disposition RECEIVER_ROOM_TARGETS_PER_PLAY_V1_SUPPORTED
- all 23 discovery gates passed
- pooled macro MAE 3.337876 -> 3.168127
- pooled macro p90 6.736290 -> 6.434198
- pooled macro abs bias 0.323767 -> 0.253029
- WR/TE/RB_FB pooled MAE all improved

2024-2025 unchanged confirmation:
- first run 36173485673 was MECHANICAL ONLY: wrong helper import caused KeyError prior_history_plays; no scientific output
- bounded repair commit c1b48b90364e2d6f82b66271065cc7b359d725ca restored the already-frozen targets-per-play helper semantics
- authoritative corrected run 36174077739 = SUCCESS mechanically
- job 108200327879
- artifact 10881825507
- digest sha256:3956a8ed9cc2191714787ecab235831b5f9ce39c59f3e08ba260ee2918f2de9c
- canonical result docs/research/RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_CONFIRMATION_RESULT.md
- result commit fb5a9196d02a69e87dd6ec80418bd01865c3600a
- disposition RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_FAILED_CLOSED

Confirmation result:
2024:
- WR MAE 4.561046 -> 4.724467 WORSE
- TE 2.982040 -> 2.881030
- RB_FB 2.389736 -> 2.169780
- macro 3.310940 -> 3.258426

2025:
- WR 4.510085 -> 4.440251
- TE 2.768463 -> 2.728961
- RB_FB 2.302348 -> 2.176637
- macro 3.193632 -> 3.115283

Pooled:
- WR 4.535565 -> 4.582359 WORSE
- TE 2.875251 -> 2.804996
- RB_FB 2.346042 -> 2.173209
- macro MAE 3.252286 -> 3.186854
- macro p90 6.646801 -> 6.651926 WORSE
- macro abs bias 0.404433 -> 0.325998
- summed-room MAE 6.127841 -> 5.983952
- summed-room p90 12.704084 -> 12.049116
- all-room candidate closer 52.27%

Frozen failures:
1. 2024 WR room MAE did not improve
2. pooled WR room MAE did not improve
3. pooled macro p90 did not remain nonworse

All provenance/specialist/no-sportsbook/no-fit gates passed.

NO PLAYER/FULL-STACK INTEGRATION IS AUTHORIZED.

NO RESCUE:
- no excluding 2024
- no WR-specific route/multiplier
- no recency/shrinkage/window search
- no fixed57 blend
- no bias offset
- no WR1/Q4 exemption
- no specialist-order change
- no 2026 outcome fitting
- no sportsbook routing

NEXT ACTION

Start a DIAGNOSTIC-ONLY WR ROOM REGIME-INSTABILITY AUDIT.

Goal:
Explain why WR room targets-per-play appears helpful in 2022/2023/2025 but harmful in 2024, while TE/RB_FB remain directionally stable.

This diagnostic must score ZERO candidate variants and must not tune the failed formula.

Investigate strictly pregame structural explanations such as:
- prior-season -> current-season WR room target-share/rate drift
- top-target / WR-room roster continuity and turnover
- QB change
- offensive coordinator/play-caller change only if a reliable historical leakage-safe source already exists
- speed of within-season room adaptation
- whether 2024 misses are league-wide or concentrated in transition clusters

If no strong structural explanation emerges, close this room-history family and move to another architecture frontier instead of searching weights/windows.

IMPORTANT CLOSED RESULTS / ANTI-RETEST

- uniform Targetable-Dropback player thinning FAILED CLOSED; do not repeat it
- Receiver Room Targetable-Rate/dropback-denominator V1 FAILED CLOSED
- Active-Roster Receiver Room State V1 is NOT SUPPORTED / CLOSED
- WR1-only current-state candidate NOT JUSTIFIED
- One-Pass-State V1 CLOSED
- Hierarchical Receiver Mean Reconciliation V1 CLOSED
- official-attempt receiver pool CLOSED
- C1 group target-mass and C3 joint conservation CLOSED
- generic receiving attempt-semantics C4 CLOSED
- Migration 18/20/21 pass-rate retuning CLOSED
- Rush Pool V1 rescue variants CLOSED
- retrospective RB router/threshold/feature work CLOSED under M96E
- sportsbook information remains downstream only
- no paid OddsAPI pull without explicit approval

Important preserved signal:
Targetable team-volume science still improved team target prediction across 2022-2025, and TE/RB_FB room opportunity remains directionally promising. Do not erase those findings just because the exact all-room targets-per-play formula failed confirmation.

PRODUCTION

Recent receiver research has not changed production behavior. The current main after docs-only continuity merge was 67b9aa349258aff78cb0b588432eab91a3f1da78 before the final handoff refresh.

OPERATING STYLE

Keep working autonomously. Do NOT ask me to invent the next experiment.
Find football-grounded structural explanations yourself.
Freeze any future candidate before scoring.
Leave a complete GitHub/Issue #535 paper trail.
Close loops.
Do not rerun failed formulations.
Keep chat updates concise and GitHub documentation exhaustive.
