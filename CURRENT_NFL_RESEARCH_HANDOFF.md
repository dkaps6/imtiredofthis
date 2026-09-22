# CURRENT NFL RESEARCH HANDOFF — READ FIRST
GitHub is canonical; chat memory is secondary.

---

## STANDING PROHIBITION — RETROSPECTIVE RB RUSHING RESEARCH IS CLOSED

**This is an operative rule, not a history note. Read it before proposing any RB rushing work.**

The M96 chain ran the RB opportunity/efficiency program to a pre-committed
terminal stop. Its artifacts live on unmerged research branches, so this
prohibition was previously invisible to any session working from `main` — which
is exactly how it came to be violated in Issue #535 (proposal `5754250023`,
retracted in `5754278269`). Recorded here so that cannot recur.

### Terminal disposition

`M96E_FINAL_RETROSPECTIVE_ROUTER_FAILED_STOP` / `AUTONOMOUS_RB_RESEARCH_STOP`
(run `33467630395`, job `99730679349`).

Eight of nine frozen retention checks passed. The only failure was the
predeclared materiality requirement:

- required all-RB rushing-yard MAE gain: **>= `0.150000` yards**
- observed gain: **`0.141791` yards**
- **shortfall: `0.008209` yards**

Read those carefully: `0.141791` is the gain the candidate *achieved*, not the
amount it missed by. The chain was closed by a shortfall of eight thousandths
of a yard, with a working mechanism and eight of nine gates passed -- not by a
failed or broken one.

### The continuation rule — verbatim intent

> Any further retrospective router threshold/feature variants would reuse
> exposed 2025 outcomes and risk overfitting. New RB architecture evidence must
> now come from genuinely prospective/untouched 2026 games or a separately
> justified new-data source that does not retune against the exposed historical
> outcomes.

Exactly two continuations are sanctioned:

1. **genuinely prospective / untouched 2026 evidence** — this is what the RB PD2
   forward/shadow confirmation lane produces. That lane is a sanctioned
   continuation of RB science, not incidental plumbing.
2. **a separately justified new-data source** that does not retune against
   exposed historical outcomes.

Anything else — another router variant, threshold search, feature hunt or
re-decomposition against the exposed 2025 sample — is overfitting, and is
forbidden regardless of how it is framed.

### M96A standing attribution result — do not re-derive this

M96A already performed the opportunity-vs-efficiency attribution
(run `33459376333`, job `99706110345`, artifact `9782611047`,
branch `research-rb-m96a-opportunity-efficiency-attribution`), n = 1,393
RB/FB player-games, 2025:

| Quantity | Value |
|---|---:|
| pregame M94C rush-yard MAE | **21.0312** |
| perfect actual carries, frozen efficiency | **13.3535** |
| opportunity MAE recovery | **7.6777** |
| perfect game efficiency, frozen carries | **14.3055** |
| efficiency MAE recovery | **6.7256** |
| opportunity-dominant share of games | **59.73%** |
| efficiency-dominant share of games | **40.27%** |

Routed **JOINT**: opportunity cleared the component-share gate but missed the
>= 1.0-yard recovery-margin gate by `0.048` yards.

**RB rushing yards is therefore not irreducible** — roughly seven yards of MAE
is recoverable from each factor. The binding question is *which* factor, and
that flips by workload regime:

| Actual carries | Pregame MAE | Perfect carries | Perfect efficiency | Opportunity recovery | Efficiency recovery |
|---|---:|---:|---:|---:|---:|
| 0–5 | 13.288 | 5.245 | 10.027 | **8.043** | 3.261 |
| 6–10 | 21.191 | 13.561 | 13.188 | 7.630 | **8.002** |
| 11–14 | 25.812 | 19.270 | 15.208 | 6.542 | **10.605** |
| 15–19 | 29.764 | 23.749 | 16.989 | 6.015 | **12.775** |
| 20+ | 40.005 | 28.636 | 36.409 | **11.369** | 3.596 |
| 25+ | 49.310 | 37.549 | 54.390 | **11.762** | −5.079 |

Low-volume (0–5) and high-volume (20+, 25+) games are **opportunity** problems.
The 11–19 middle is an **efficiency** problem. Any future new-data justification
should cite this table rather than re-running the attribution.

### Chain lineage and where the artifacts live

| Migration | Disposition |
|---|---|
| M96A — opportunity vs efficiency attribution | `JOINT_ADVANCE_M96B_SEPARATE_WORKLOAD_AND_EFFICIENCY_DISTRIBUTIONS` |
| M96B — modular joint workload × efficiency synthesis | `M96B_MODULAR_SYNTHESIS_COMPLETE`; M95C residual not plug-compatible with M94C |
| M96C — M94C-anchored efficiency residual | `M96C_NO_GLOBAL_WINNER_CONDITIONAL_EFFICIENCY_SIGNAL_SUPPORTED` |
| M96D — pregame conditional efficiency routing | `M96D_PRIMARY_ROUTER_FAILED` |
| M96E — role router with frozen workload-risk guard | `M96E_FINAL_RETROSPECTIVE_ROUTER_FAILED_STOP` |

Full plans and results are preserved on the research branches
`research-current-state` and `research-rb-final-qualification` under
`docs/migrations/M96A_*` through `M96E_*`, with evaluators under
`scripts/backtest/evaluate_rb_m96*`. **Do not merge those branches into `main`
to read them** — use `git show origin/research-current-state:<path>`.

---

## ACTIVE CHECKPOINT — 2026-09-22 — RB PD2 FORWARD LOCK CLOSED; PR #625 MERGED

**Current detailed handoff:**

`docs/handoffs/NFL_HANDOFF_2026-09-21_RB_PD2_FORWARD_LOCK_AND_MODEL_IMPROVEMENT_CURRENT.md`

Read that file for the implementation lineage, but this root checkpoint supersedes
its stale pre-Week-2 status.

### Final validated PR #625 state

PR #625 merged successfully into `main` at:

`2a8b4e5622df3917c672212b16a8189c2bab7921`

Production `main` immediately before that merge was:

`76e5e0452a88018b95ebc212fb0d1b21a2ca90a3`

Active branch:

`research-rb-pd2-forward-shadow-confirmation-v1`

Final **code-bearing** head before this handoff-only documentation update:

`c58860b0dee02d18a93e870d3541e8212987ae31`

PR:

`#625 — RB PD2 forward-shadow implementation validation — MERGED`

The previously open Week-2 history-freshness blocker is **closed**. The real
Week-2 outcomes were certified and the forward-history state was rebuilt and
re-pinned before any Week-3 lock can count.

### Final Week-2 certification evidence

Completed-history certification:

- run `35732450688` = **SUCCESS**
- artifact `10695992072`
- artifact name `rb-pd2-completed-2026-history-v2-week`
- digest `sha256:a84aa005dfdd35d16f891c40c1cafbad5c2779aca3dad16214624d0216e052bb`
- 108 / 108 Week-2 RB/FB football-projection rows received verified outcomes
- 93 rows from the nflverse weekly stats table
- 15 roster-confirmed verified zeroes
- 0 exclusions
- cumulative certified history = 215 rows across Weeks `[1,2]`
- sportsbook inputs used for football projection = 0

The final implementation uses the authoritative nflverse weekly player-stats
table plus weekly roster identity/verified-zero evidence only. A temporary PBP
fallback explored during debugging was removed before the final certification.
The shared GSIS suffix resolver was corrected to handle `III` / `II` / `IV`
longest-token-first, which allowed suffix variants such as Kenneth Walker III
to resolve through the normal stats path.

### Final forward-history evidence

Forward-history rebuild:

- run `35732782075` = **SUCCESS**
- artifact `10696312599`
- artifact name `rb-pd2-forward-history-v1`
- digest `sha256:1205ea152b4702e19d12379c79e29ff7fc20fd2a632a61f7c519980f64b648fb`
- 1,608 state rows
- 169 players
- 889 scoreable rows
- manifest `completed_2026_through_week = 2`
- manifest completed weeks = `[1,2]`
- zero prospective outcomes used

Full Slate now pins that exact immutable Week1+2 forward-history artifact.

### Prospective lock-integrity closure

The second late P1 is also closed. Lock assembly now:

1. writes the NPZ/JSONL lock files first;
2. records the actual persisted timestamp;
3. requires every persisted lock to retain a frozen **15-minute pre-kickoff
   upload buffer**;
4. clears the session lock and fails closed if that post-persistence gate fails.

Regression coverage proves that exactly 15 minutes before kickoff passes and
1 ms later fails.

### Exact-head validation at the final code-bearing head

At `c58860b0dee02d18a93e870d3541e8212987ae31`:

- Repo CI run `35733488365` = **SUCCESS**
- preserved paid Full Slate replay run `35733488509` = **SUCCESS**
- PR was clean/mergeable
- both prior P1 review threads were resolved
- no paid/live OddsAPI pull was triggered

A final Codex exact-head review found only one P2 documentation defect: this
root handoff still described the now-completed Week-2 blocker. This section is
the fix for that documentation defect; it does not change model science or the
validated runtime behavior above.

### What PD2 is — and what it is not

PD2 is a mean-neutral **distribution-calibration** mechanism.

Historical qualification improved pooled CRPS by +1.218% and high-difficulty
CRPS by +2.624%, with better coverage/tail calibration, but **point MAE is
unchanged by design**.

It does not fix RB carries, YPC, or rushing-yard mean accuracy.

The user explicitly wants this implementation finished, then wants the project
back on actual model-metric improvement.

### Authorized next step after PR #625 merge

The merge is complete. Do **not** reopen PR #625 or repeat its Week-2 certification.

1. preserve the final PR/artifact lineage in Issue #535 / the canonical paper trail;
2. return immediately to actual model performance / new-information research;
3. use the real Week-2 slate to build the authority-exact position/market
   scoreboard (including the downstream deduplicated betting hit-rate view);
4. prioritize sanctioned new pregame information rather than reopening exposed
   retrospective RB router variants.

The strongest already-identified RB mean-information lead remains **backfield
teammate availability / injury-created vacancy propagation into rushing
opportunity**, with prior-week snap share as a second live/free lagged
opportunity proxy.

Claude is temporarily unavailable because the user is out of credits. Do not
wait for Claude; GPT-5.6 continues solo.

No paid OddsAPI pull is authorized without explicit user approval.

---

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
