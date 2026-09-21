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

## ACTIVE CHECKPOINT — 2026-09-21 — RB PD2 FORWARD / SHADOW IMPLEMENTATION

The user's current RB priority is the already-frozen RB-PD2 yard-difficulty width
forward confirmation. GitHub remains canonical over chat memory.

Active implementation branch:

`research-rb-pd2-forward-shadow-confirmation-v1`

Validation PR:

`#625 — WIP: RB PD2 forward-shadow implementation validation`

### What is now implemented

- accepted §7 same-process empirical-array capture contract is integrated;
- §3 strict-prior historical difficulty state is implemented;
- §8 exact frozen mean-neutral empirical width transform is implemented;
- §9 immutable pregame lock assembler is implemented;
- true `kickoff_utc` comes from
  `scripts/build/_schedule_utils.py:get_nfl_schedule`, not sportsbook
  `commence_time` and not date-only `team_week_map.gameday`;
- schedule matching verifies canonical team **and opponent**;
- missing/NaN capture identities fail closed instead of becoming literal
  `"nan"` keys;
- identity-source fingerprints are non-empty/equal-or-fail;
- Week-1 P3/STACK1 parity is mechanically recomputed from the underlying
  projection values rather than trusted from a caller boolean.

The three integrity defects raised by the PR #625 automated review
(mechanical Week-1 parity, opponent validation, NaN identity handling) have
been fixed with regression tests and their review threads resolved.

### Certified completed 2026 history

Week-1 completed-history workflow:

- run `35619889214`;
- artifact `rb-pd2-completed-2026-week1-history-v1`;
- artifact ID `10649235022`;
- 107 verified completed Week-1 rows;
- zero prospective observations created.

Combined forward-history workflow:

- run `35620321000` — green;
- artifact `rb-pd2-forward-history-v1`;
- artifact ID `10648116139`;
- artifact digest
  `sha256:52e56ce090576480d9b8cf4ec1999bfaf9e6884e6a0d1266efc075f1afac3af8`;
- 90-day retention;
- 1,500 history rows / 168 players;
- 808 scoreable rows under the frozen prior/reference contract;
- maximum strict-prior reference N = 882;
- includes the 107 certified Week-1 rows.

The Full Slate activation path pins that exact green history run for the first
future lock.

### Prospective activation boundary

Operational start is frozen at:

`2026-09-22T12:00:00Z`

Therefore **no Week-1 or Week-2 2026 game can ever become a prospective
confirmation observation**. The first possible scientific observation is a
future Week-3-or-later eligible player-game.

See:

`docs/research/RB_PD2_FORWARD_SHADOW_ACTIVATION_V1.md`

### Live collection wiring

`.github/workflows/full-slate.yml` now has an opt-in manual input
`rb_pd2_shadow_capture`, default **false**.

When live odds are explicitly requested/available and that input is true:

1. canonical production pricing runs first;
2. §7 captures exact `adjusted_outcomes` in-process;
3. the pinned green history artifact is restored;
4. §8/§9 assemble the immutable pregame candidate lock;
5. the dedicated `rb-pd2-forward-lock-<run_id>` artifact is uploaded
   immediately.

History restore, lock assembly, and research upload are all non-blocking for
canonical production pricing. A research failure invalidates only the research
lock.

No paid OddsAPI pull has been triggered by this implementation work.

### What remains before observation #1

1. finish the latest PR #625 CI / preserved-artifact replay / re-review on the
   exact live-wiring head;
2. land the implementation only after those gates are green;
3. the first future paid Full Slate with `rb_pd2_shadow_capture=true` still
   requires explicit user authorization because it consumes OddsAPI credits;
4. verify its dedicated lock artifact was uploaded before kickoff and is valid;
5. only then may the separate RB mean-information lane formally open in
   parallel.

Claude's new-data readiness inventory is on
`research/rb-new-data-readiness-v1@355ce5c7`. A separate docs/evidence-only
audit of the OUT/DOUBTFUL/IR/PUP self-haircut production behavior was assigned
in Issue #535; do not let that audit mutate the frozen PD2 study.

The M96E standing retrospective stop recorded above remains fully operative.

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
