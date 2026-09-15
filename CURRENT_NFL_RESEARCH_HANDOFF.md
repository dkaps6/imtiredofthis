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

## ACTIVE DRAFT PR #562 — RB-PD2 YARD-DIFFICULTY MC-WIDTH V1 — UPDATED 2026-09-15

Branch `research-rb-pd2-yard-difficulty-mc-width-v1`. The historical MC-distribution
parity-reconstruction failure previously blocking candidate evaluation (2022 W08
mismatch `1.0232351709`; 2023 W03 mismatch `2.7762633539`) was root-caused: a
same-job double-build diagnostic (branch `scratch-diag-rb-pd2-w3-2023-mismatch`)
proved the historical-input builders are fully deterministic and the reconstruction
code reproduces `mc_proj` to `~1e-14` against its own same-run inputs. The failure
was an artifact of checksumming against a `component_predictions.csv` downloaded
from a separately-triggered M95Q run rather than a code bug or nflverse drift.
Fixed at commit `780087b1` (`rebuild-distributions` now builds its own in-job
`walk_forward.py` reference and checksums against that). See Issue #535 comment
`5689731492` for the full writeup.

First real end-to-end run: `35037087309`, all jobs green, reconstruction
integrity now `~1e-14`/`~1e-15`. Disposition: `RB_YARD_DIFFICULTY_WIDTH_INTEGRITY_FAILURE`
-- but every science gate (mean-neutrality, point-MAE identity, pooled/high-difficulty
CRPS, coverage, Brier, 3-of-4-season robustness) passed. The sole failure is
`A_parent_panel_matches_556`, a stale exact-row-count check against PR #556's
original frozen artifact (`5607`/`4652` rows); this run's fresh-rebuilt panel has
`5616`/`4657` rows (+9/+5, ~0.16%), while the independent fresh-source identity-set
check passed with zero symmetric difference. Flagged for GPT-5.6 adversarial review
rather than patched unilaterally (Issue #535 comment `5689826445`) -- do not change
this gate without that review landing first.

## PRODUCTION CHECKPOINT

The Week-1 Full Slate incident is already repaired/green. Do not reopen paid-live debugging for WR research.

Production handoff:

`docs/handoffs/NFL_HANDOFF_2026-09-11_FULL_SLATE_LIVE_REPAIR_MERGED_CURRENT.md`

No production-science change is authorized by the current WR diagnostic work itself.

Older research ledgers previously carried in this root file remain available in Git history and `NFL_MASTER_CONTINUITY_RECORD.md`; this root file is intentionally kept as a concise pointer to the newest canonical handoff.
