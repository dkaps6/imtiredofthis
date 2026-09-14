# CURRENT NFL RESEARCH HANDOFF — READ FIRST

GitHub is canonical; chat memory is secondary.

## ACTIVE RESEARCH CHECKPOINT — 2026-09-14

The user's explicit current priority is **WR receiving yards**. Do not skip ahead to RB until the user changes priority.

Read in this order:

1. `AGENTS.md`
2. `docs/handoffs/NFL_HANDOFF_2026-09-14_WR_RECEIVING_YARDS_CURRENT.md`
3. `NFL_MASTER_CONTINUITY_RECORD.md`
4. latest GPT-5.6 / Claude checkpoints in GitHub Issue #535

The active handoff contains the exact WR authority lineage, PR #600 benchmark state, WR1 identity correction, receiving-yard decomposition, closed QB-C2 shared-tail result, Claude independent artifact lineage, anti-retest rules, the next WR research objective, and the parked RB Weeks-2-18 checkpoint.

### Current WR interpretation

- M38 + `WR_R15_PRODUCTION_MODEL_V1` remain valid production authorities and improved football accuracy in their frozen OOS tests.
- WR receptions/opportunity are comparatively healthy.
- WR receiving-yard translation/efficiency is the active weakness, especially high-efficiency/right-tail games.
- The exact QB-C2 -> WR1 shared-tail selector has been independently tested by GPT-5.6 and Claude and is CLOSED. Do not rescue it with new percentiles/thresholds.
- Before any new WR candidate, perform an anti-retest + feature-availability audit and freeze one genuinely new leakage-safe hypothesis.
- GPT-5.6 and Claude must continue collaborating through Issue #535 and independently challenge each other's design/results.

### Open authority-exact benchmark

PR #600 remains open/mergeable at handoff creation:

- head `1167f9fdadde452deb84d3891097865da2f163d5`
- canonical run `34843204550`
- artifact `10346639168`

Use it as diagnostic evidence; do not train football projections against market lines.

## PARKED RB CHECKPOINT

RB Weeks 2-18 work is intentionally parked while WR receiving yards is active.

Branch `research-rb-pd2-yard-difficulty-mc-width-v1`, head `3e3da9ec7216ec836b4d593a3b5ac32f524442f0`.
Run `34876877949` failed during historical MC-distribution parity reconstruction before candidate evaluation (2022 W08 mismatch `1.0232351709`; 2023 W03 mismatch `2.7762633539`). No RB width scientific result was exposed. Preserve this state for later.

## PRODUCTION CHECKPOINT

The Week-1 Full Slate incident is already repaired/green. Do not reopen paid-live debugging for WR research.

Production handoff:

`docs/handoffs/NFL_HANDOFF_2026-09-11_FULL_SLATE_LIVE_REPAIR_MERGED_CURRENT.md`

No production-science change is authorized by the current WR diagnostic work itself.

Older research ledgers previously carried in this root file remain available in Git history and `NFL_MASTER_CONTINUITY_RECORD.md`; this root file is intentionally kept as a concise pointer to the newest canonical handoff.
