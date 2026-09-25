Continue my existing NFL Stuff project from the canonical GitHub paper trail.

Repo: dkaps6/imtiredofthis

GitHub is canonical over chat memory. Do not make me re-explain prior work.

Read ONLY, in this order, to conserve context:
1. AGENTS.md
2. top checkpoint of CURRENT_NFL_RESEARCH_HANDOFF.md
3. docs/handoffs/NFL_HANDOFF_2026-09-25_RECEIVER_ROOM_FRONTIER_CURRENT.md
4. Issue #535 from comment 5834428368 onward, especially 5837470764
5. live branch/run state

Do not recursively load older handoffs unless the current handoff explicitly points you there.

CURRENT ACTIVE PRIORITY

Receiver Room Targets-Per-Play V1 has QUALIFIED on the untouched 2022-2023 temporal screen.

Exact candidate:
R_g_play = sum(strict-prior room targets) / sum(strict-prior offensive plays)
candidate room targets = projected offensive plays * R_g_play

2022-2023 authoritative run:
36172644864
artifact:
10880948137
result:
docs/research/RECEIVER_ROOM_TARGETS_PER_PLAY_V1_RESULT.md
disposition:
RECEIVER_ROOM_TARGETS_PER_PLAY_V1_SUPPORTED

All 23 frozen discovery gates passed. Pooled macro room MAE improved 3.337876 -> 3.168127, macro p90 6.736290 -> 6.434198, macro abs bias 0.323767 -> 0.253029. WR, TE, and RB_FB all improved pooled MAE.

An exact unchanged 2024-2025 confirmation is already running:

Branch:
research-receiver-room-targets-per-play-v1-confirm-2024-2025

Head at handoff:
31e1854be631eeae3b5c6d25aa2c61b3a6c51925

Run:
36173485673

At handoff time it was IN PROGRESS.

FIRST ACTION:
Check run 36173485673. Do NOT duplicate it.

If it passes all 27 frozen gates:
- document the exact run/job/artifact/digest/result in Issue #535;
- commit a canonical result;
- freeze a separate player/full-stack integration plan before any scoring;
- player integration must preserve confirmed room totals and allocate within each room using existing authorized within-room entitlement;
- preserve M38, TE-R5P, and authorized specialist ordering;
- 2024 may use fold-safe WR-R15; retrospective WR-R15 in 2025 remains forbidden;
- catch rate and YPT unchanged;
- QB/rushing/ATD unchanged;
- RB rush+receiving rebuilt through current RB V2;
- protect WR/TE/RB receptions, receiving-yard MAE/p90/bias/Q4 tails;
- zero sportsbook;
- parameters fit 0;
- one candidate;
- historical qualification still requires prospective 2026 confirmation before production.

If it fails scientifically:
close the exact formula. No shrinkage, recency, fixed57 blend, bias offset, room multiplier, WR1/Q4 exemption, specialist change, or sportsbook rescue.

If it fails mechanically:
repair only the bounded mechanical/provenance defect. Do not alter formula/gates/cohort/baseline ordering.

IMPORTANT RECENT CLOSED RESULTS

- Receiver Targetable-Dropback V1 team-level formula improved team target volume independently in 2022, 2023, 2024 and 2025, but the player full-stack translation FAILED CLOSED (run 36147357028 / artifact 10869364930). Do not uniformly thin every player's entitlement again.
- Current-stack compensation audit run 36151138507 proved why: team targets improved but WR room worsened; TE/RB room improved; WR1/Q4 were already underallocated.
- WR1 current-state diagnostic run 36171050355 found strong absolute WR1 state signal, but a fixed-room WR1 candidate was NOT justified. Do not retune M38 or create a WR1-only rescue.
- Receiver Room Targetable-Rate V1 run 36171462750 failed closed only because the fixed57 projected-dropback denominator is negatively biased. Exact dropback-denominator formulation is closed.
- Active-Roster Receiver Room State V1 run 36172146432 / artifact 10880536886 is NOT SUPPORTED and CLOSED; WR room gates failed.
- C1 group target-mass, C3 joint conservation, One-Pass V1, hierarchical receiver mean reconciliation, official-attempt pool, generic attempt-semantics C4, and Rush Pool V1 rescues remain closed.
- Retrospective RB router/threshold/feature research remains closed under M96E.
- Sportsbook information is downstream only.
- No paid OddsAPI pull without my explicit approval.

Production main was 5421ba24b28aeff88e1f6466d93970f01264ebfa before the docs-only handoff. Recent receiver research has not changed production behavior.

Keep working autonomously. I do NOT want to be asked to invent the next experiment. Find the next football-grounded structural question yourself, freeze it before scoring, and leave a complete GitHub paper trail. Close loops. Do not rerun failed formulations. Keep chat updates concise and GitHub documentation complete.
