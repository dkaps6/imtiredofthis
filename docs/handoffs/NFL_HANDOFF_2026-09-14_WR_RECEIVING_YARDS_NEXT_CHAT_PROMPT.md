# NEXT-CHAT PROMPT — WR RECEIVING YARDS

Continue the existing NFL project from GitHub canonical state. Repo: `dkaps6/imtiredofthis`.

Read, in order:
1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. `docs/handoffs/NFL_HANDOFF_2026-09-14_WR_RECEIVING_YARDS_CURRENT.md`
4. `docs/handoffs/NFL_HANDOFF_2026-09-14_WR_RECEIVING_YARDS_EVIDENCE.md`
5. `NFL_MASTER_CONTINUITY_RECORD.md`
6. latest GPT-5.6 + Claude checkpoints in Issue #535.

Do not make me re-explain prior work. GitHub is canonical over chat memory.

Current priority is WR receiving yards. Do not jump to RB. Receptions/opportunity are comparatively healthy; receiving-yard efficiency/translation and ceiling misses are the active problem.

Preserve M38 + `WR_R15_PRODUCTION_MODEL_V1`, #599 receiving weights, M89/M90, and C2. The exact QB-C2 -> WR1 shared-tail lane has already failed under the canonical frozen 884-row contract and independently under Claude's sensitivity pipeline. Do not rescue it with new percentiles/thresholds.

Before any new WR candidate, do a full anti-retest + feature-availability audit across M72/M75/M84, R3/R7/R9-R11, C1/C3, ND3, and the latest WR1 decomposition. Identify only genuinely untested, leakage-safe WR receiving-yard mechanisms.

Collaborate with Claude in Issue #535. Have Claude independently audit prior work and propose/falsify the next mechanism rather than merely agreeing with you. Inspect each other's artifacts directly.

First practical task: build a WR receiving-yard efficiency anti-retest + feature-availability matrix over exact authority identities, then freeze ONE narrow hypothesis with cohort, development/holdout split, metrics and gates before results. Football receiving-yard accuracy/tail behavior is the primary scientific target; market comparison remains downstream diagnostic only.

RB PD2 width work is parked on `research-rb-pd2-yard-difficulty-mc-width-v1` at `3e3da9ec7216ec836b4d593a3b5ac32f524442f0`; run `34876877949` failed mechanically before candidate evaluation. Preserve it for later and do not continue it now.
