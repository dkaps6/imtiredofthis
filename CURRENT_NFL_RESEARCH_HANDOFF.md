# CURRENT NFL RESEARCH HANDOFF — READ FIRST
## ACTIVE CHECKPOINT — 2026-09-20 — RB PD2 FORWARD / SHADOW CONFIRMATION

User has explicitly re-opened RB as an unresolved priority.

Read first:

`docs/handoffs/NFL_HANDOFF_2026-09-20_RB_PD2_FORWARD_SHADOW_CURRENT.md`

Active branch:

`research-rb-pd2-forward-shadow-confirmation-v1`

Branch was created from clean production main:

`f0dad2c6711e85104eeffedfa5f5112fd172cbf5`

Latest implementation checkpoint:

- frozen forward plan: `docs/research/RB_PD2_FORWARD_SHADOW_CONFIRMATION_V1_PLAN.md` @ `0de96f69194f1fca242515b613b814eab0c22d35`;
- latest active RB handoff update: `6f26a62b22ca688c3a0dfce0edfab07501b7f5ce`;
- Claude owns §7 capture hook on `research-rb-pd2-shadow-capture-hook-v1`;
- first hook commit `0cc674e3` is **not yet accepted/cherry-picked**;
- GPT-5.6 review comment `5751454114` identified four blockers: sportsbook book/line dedup, lossless empirical-array persistence, live-safe shadow failure isolation, and run/session provenance;
- next action is to inspect Claude's repaired §7 SHA, integrate only if the six requested tests pass, then build §3 + §8 + §9;
- no prospective outcome may be graded yet.

The strongest unfinished RB result is already positive:

`RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`

PR #562 / run `35039152022` passed all 28 gates, but the separately required
forward/shadow confirmation has never been started. The current task is to design and
lock that prospective confirmation without retuning the qualified 0.30 mean-neutral
width mapping and without changing production.

Issue #535 is again an active GPT-5.6 + Claude collaboration surface at the user's
explicit request.

---

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

## ACTIVE DRAFT PR #562 — RB-PD2 YARD-DIFFICULTY MC-WIDTH V1 — QUALIFIED 2026-09-16

Branch `research-rb-pd2-yard-difficulty-mc-width-v1`. The historical MC-distribution
checksum failure (root cause: comparing against a cross-run M95Q artifact, not
nflverse drift or a code bug) was fixed at `780087b1` -- `rebuild-distributions`
now builds its own in-job `walk_forward.py` reference. See Issue #535 comment
`5689731492`.

First end-to-end run (`35037087309`) passed every science gate but failed one
stale integrity gate (`A_parent_panel_matches_556`, an exact-row-count check
against PR #556's original frozen artifact). GPT-5.6 reviewed on Issue #535 and
approved four fixes (comment landed 2026-09-16): implement Amendment 3's crossed
player x game CRPS bootstrap, add fail-closed fresh-parent VALUE parity (not just
identity), fix mislabeled composite source provenance, and demote the old count
gate to a non-fatal disclosure. Implemented exactly as specified at `b9ad5247`,
reran once (`35039152022`, preserving `35037087309` unchanged in the paper trail).

**Final disposition: `RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`.** All 28 gates
pass. Pooled CRPS +1.218%, high-difficulty-quartile CRPS +2.624%, both
player-clustered and crossed player x game bootstraps `p=1.0`, point-MAE
identical (mean-neutral), coverage and Brier-100 both improve, 4/4-season
robustness. Reported in full on Issue #535 comment `5690193256`.

Research qualification only -- per the frozen plan, a separate forward/shadow
confirmation is still required before any production change. None has been
started; production is untouched.

## PRODUCTION CHECKPOINT

The Week-1 Full Slate incident is already repaired/green. Do not reopen paid-live debugging for WR research.

Production handoff:

`docs/handoffs/NFL_HANDOFF_2026-09-11_FULL_SLATE_LIVE_REPAIR_MERGED_CURRENT.md`

No production-science change is authorized by the current WR diagnostic work itself.

Older research ledgers previously carried in this root file remain available in Git history and `NFL_MASTER_CONTINUITY_RECORD.md`; this root file is intentionally kept as a concise pointer to the newest canonical handoff.
