# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-10-26 — V3 — IDENTITY CARD SUBSTRATE ACCEPTED / AUDIT PRIORITIES 1-4 DONE / RE-AUDIT PENDING / NO PREDICTIVE CLEARANCE

**Model:** PRISM — Player Role & Individual Situation Model

### READ THIS DEEP HANDOFF NEXT

`docs/handoffs/NFL_HANDOFF_2026-10-26_WEEK7_POSTMORTEM_PLAYER_FIRST_ENVIRONMENT_CONTINUITY.md`

Displaced checkpoints, archived verbatim, **not** for resuming:
- `docs/handoffs/NFL_HANDOFF_2026-10-26_V3_ARCHIVED_TOP_IDENTITY_CARD_ACCEPTED_SUBSTRATE.md`
- `docs/handoffs/NFL_HANDOFF_2026-10-26_V2_ARCHIVED_TOP_WEEK7_FORENSIC_PLAYER_FIRST_TRANSMISSION.md`

Do not recursively read older dated handoffs.

### ABSOLUTE ASSIGNMENT

PRISM is an **INDIVIDUAL-PLAYER PROJECTION MODEL**.

> **THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.**

> **ANOMALIES ARE QUESTIONS, NOT TRENDS.**

Position MAE/RMSE is downstream QA only. Identity/state/opponent work is additive; do not scrap/rebuild PRISM.

Always distinguish: `SOURCE READY` / `SOURCE BLOCKED` / `PREDICTIVE CLOSED` / `QUALIFIED`.

### CONTROL ROOM

- Claude = executor / research lead
- GPT-5.6 = independent auditor / project control
- shared lane = Issue #535

Before substantive work: read `AGENTS.md`; read only this TOP checkpoint; read the deep handoff; fetch newest #535; reconcile live `main` / PRs / Actions.

### LIVE STATE

`main` at checkpoint: `b734247ce8d8e8ed8dc50cd3021af5e54e9f1e18` (plus this handoff commit).

Active research branch (user-authorized, genuine descendant of main):

`research-prism-player-first-identity-environment-2026-10-26` @ **`aa70663c`**

Green runs: card `40391557775`, sweep `40391557902`. 68 tests under pandas 1.5.3 / numpy 1.26.4 (the versions CI installs).

Newest #535 comment at checkpoint: **6426050333**

This chat's #535 record:
- `6422827149` corrected card build + source audit
- `6422904132` correction: card did not satisfy the anomaly gate
- `6422987723` correction: my player-level-bridge prediction was wrong
- `6423234263` anomaly gate enforced, incl. a 22% false positive in my own gate
- `6425032876` ledger superseded on main
- **`6425282770` GPT AUDIT — substrate ACCEPTED, predictive BLOCKED, 5-item priority order**
- `6426050333` priorities 1-4 complete, revised fixed target presented for re-audit

**No paid OddsAPI call is authorized. Betting remains CLOSED.**

Carried-forward repo debt:
- Repo CI on main still fails one known stale no-credit exact-string `FETCH_LIVE_ODDS` assertion. **Do not weaken the live-odds guard to make it pass.**
- `claude/nice-cori-4qkwqs` is an **orphan branch with no merge base**; its tooling (Week-6 graders, gate-salvage script, Week-6 result docs) must be ported file-by-file.
- `requirements.txt` leaves `pyarrow` unbounded; workflows pin `pyarrow==17.0.0` individually.

### GPT AUDIT 6425282770 — THE GOVERNING DISPOSITION

`PLAYER_IDENTITY_CARD_V1_DESCRIPTIVE_SUBSTRATE_ACCEPTED__NO_PREDICTIVE_CLEARANCE`

All five A–E blockers from `6420504608` materially closed for the descriptive substrate. The audit also ruled:

1. **Position universe** — keep the complete roster universe in the durable card; **all** opportunity/entitlement/player-projection research must use an explicit `can_receive_opportunity == True` subset; downstream tables must state which universe they use; a non-eligible row may not silently receive carry/target entitlement.
2. **Anomaly gate** accepted as a fail-closed screening layer, *not* a complete state model.
3. **Team-share fields are BLOCKED from predictive role authority** until week-matched. Explicitly: do not gate the numerator while leaving the denominator on all weeks — that mechanically depresses the share.
4. **Threshold sweep authorized** as descriptive only: no outcome inspection, no winner, no tuning.

### AUDIT PRIORITIES 1-4 — COMPLETE, AWAITING RE-AUDIT

**1. Source-qualified player-state resolver.** `load_injuries` available 2026 wk1-7, keyed on `gsis_id` (no bridge). A week-*w* report precedes week *w*, so no look-ahead; a test blocks a later designation explaining an earlier gap.

`UNRESOLVED_CONTEXT` 1,358 → **1,078**; `RETURN_FROM_INJURY` **213**; `INJURY_LIMITED` **67**. 280 of 1,358 (20.6%) now carry a cause.

**The withheld set is unchanged** — verified on real data and by test. An injury-explained week is explained, not representative. Withholding is keyed on a two-valued `gate_disposition`, never on the single label; keying off the label would silently promote explained weeks into usable evidence. CI fails if withheld week-slots drop below explained weeks.

`PARTIAL_GAME` is **SOURCE BLOCKED** for 2026: `load_participation` raises `Season must be between 2016 and 2025` (verified). PBP is no substitute — it names only ball-touchers. CI fails if the label is ever emitted.

Gate version: `PLAYER_STATE_ANOMALY_GATE_V2_INJURY_RESOLVED`, degrading (with CI failure) if the source is unavailable.

**2. Week-matched team-share denominators.** Numerator and denominator now come from the identical accepted player-week set. Arithmetic pinned by test: ungated 0.400, naive-wrong 0.375, week-matched **0.500** — and the test asserts the value is *not* 0.375 so a regression fails loudly. Reconstructible via `player_opp_in_gate_weeks_current` / `team_opp_in_gate_weeks` and rz/i5 equivalents; ungated retained under `*_all_weeks`. **Still descriptive/provisional, still not predictive role authority.**

**3. Descriptive threshold sweep** (`scripts/research/sweep_player_state_threshold_v1.py`):

| ratio | withheld weeks | `UNRESOLVED` | eligible cards w/ usable evidence |
|---|---|---|---|
| 0.30 | 907 | 654 | 2,651 |
| 0.40 | 1,099 | 833 | 2,573 |
| **0.50** (current) | 1,358 | 1,078 | 2,517 |
| 0.60 | 1,643 | 1,347 | 2,417 |
| 0.70 | 2,030 | 1,711 | 2,289 |

Churn between adjacent ratios 192 / 259 / 285 / 387. Spread **362 cards = 13.66%** → `MODERATELY_THRESHOLD_SENSITIVE`.

**This is unresolved by design and must be carried forward as uncertainty, not settled by picking a threshold.** `RETURN_FROM_INJURY` is flat at 213 across the range (absence-driven, not snap-ratio-driven). Named top movers: Tim Patrick (3 weeks), Nick Chubb, DJ Moore, Adam Thielen, Tyler Higbee, Jonnu Smith — ordinary skill players, so the sensitivity is not confined to fringe cases.

Script-level guarantees: `threshold_recommended` hardcoded `None`, `no_winner_selected` `True`, no code path ranks or selects a ratio; CI fails on a recommendation, outcome inspection, a changed swept set, broken monotonicity, or a flat sweep.

**4. Evidence authority + model-facing universe.**

| `evidence_authority` | all cards | opportunity-eligible |
|---|---|---|
| `position_prior_only` | 29,515 | **5,422** |
| `prior_only` | 2,714 | 2,593 |
| `prior+current` | 2,409 | 2,407 |
| `current_only` | 558 | 538 |

**49.5% of opportunity-eligible card-weeks (5,422 / 10,960) have no player-specific opportunity evidence**, so production hands them a position prior and lets them compete in the finite room. That is the materiality of the zero-history audit, and the card previously reported it only as absence (`no_history`). `opportunity_from_population_fallback` answers it directly. `participation_evidence` preserves the observed-zero distinction production's three-value vocabulary cannot express.

Durable card stamped `COMPLETE_ROSTER_UNIVERSE` (35,196 rows); model-facing universe is a named function **and** its own committed artifact stamped `MODEL_FACING_OPPORTUNITY_UNIVERSE` (10,960 rows), not a filter people must remember.

### COMMITTED ARTIFACTS (git, not expiring artifacts — §5a)

```
data/research/player_identity/player_identity_card_v1_2026.csv
data/research/player_identity/player_identity_card_v1_2026_model_facing.csv
data/research/player_identity/player_identity_card_v1_2026_status.json
data/research/player_identity/player_state_threshold_sweep_v1.csv
data/research/player_identity/player_state_threshold_sweep_v1_churn.csv
data/research/player_identity/player_state_threshold_sweep_v1_examples.csv
data/research/player_identity/player_state_threshold_sweep_v1_status.json
```

### IDENTITY BRIDGE — SOURCE-CAPPED, QUANTIFIED

Built at **player level** (`pfr_id` is a player invariant, not weekly). Ambiguity fails closed; 0 ambiguous, 2,520 players mapped. Partial-bridge coverage is **structurally impossible** (0 players, `untrusted_denominator_cards` 0).

Binding constraint is nflverse roster completeness: 48,111 / 64,455 rows (74.6%) carry `pfr_id`; 381 of 384 never-bridged players lack it in **every** row, so the per-week→player-level change recovered only 6 player-weeks — my prediction of "most" was wrong (`6422987723`).

**The statistic that matters: 2,939 / 2,945 = 99.80% coverage on opportunity-eligible cards with current-season opportunity.** `bridge_match_rate` and `offensive_player_weeks_unmatched` are the wrong statistics; the 4,900 unmatched weeks are overwhelmingly offensive linemen. Named exceptions: **Cody White** (WR, LV) and **Treyton Welch** (TE, NO).

### KNOWN GAPS — STATED, NOT HIDDEN

- **1,078 player-weeks remain unexplained.** No injury report explains a benching, blowout rest, healthy inactive or committee rotation.
- **`PARTIAL_GAME` source-blocked for 2026**; nothing available closes it.
- **Team shares week-matched but provisional**, per audit ruling.
- **13.66% threshold sensitivity unresolved by design.**
- Screening thresholds `PARTIAL_PARTICIPATION_RATIO = 0.5`, `MIN_BASELINE_WEEKS = 2` are deliberately **unfitted**.
- Role elasticity and the full teammate-competition map deferred to V2.

### WEEK-7 FORENSIC / TRANSMISSION — UNCHANGED AND STILL OPEN

Week-7 ticket failure was **construction**, not model (`6419138959`, accepted in `6420198257`): ~41–47% naive hit probability on "safe" tickets, 25/57 yardage legs bought-down OVERs against a canonical UNDER direction, SD/clearance ignored, repeated-player concentration, uncertified ATD consuming share.

**Construction failure does NOT close opponent/environment transmission.** The accepted Identity Card does not resolve the Walker / JSN / Kupp / Barner transmission findings by itself.

Transmission facts still open: `def_rush_epa` unconsumed by generic RB rules; explosive-play suppression unconsumed; WR/TE/RB YPT and outside/slot YPT dropped; YBC/stuff dropped; generic game script hardcodes `pass_share = 0.57` (`scripts/modeling/rules_v2.py:98`); lead/trail does not alter the generic split; qualified Stage-2 conditional allocation unwired.

**Environment is not game total / spread / implied points.** That is one input. It is the football situation: opportunity topology, game script, teammate competition, defensive scheme and personnel.

**Kenneth Walker III remains the preferred first generic-skill transmission decomposition case** (GPT version at `cca416bd`, `docs/research/PRISM_WEEK7_WALKER_ENVIRONMENT_TRANSMISSION_DECOMPOSITION_V1.md`).

### STAGE-2 — FAIL CLOSED BY CHRONOLOGY

`PRISM_FUTURE_STAGE2_TRANSLATION_NOT_READY__FAIL_CLOSED` (`6421086265`). Blocker: `target/future stat rows already present; refuse audit: weeks=[7]`. Do not weaken or filter around the guard, do not call Week 7 prospectively ready, **do not reconstruct a Week-7 Stage-2 shadow postgame**, do not reuse Week 7 as a retrospective substitute. The next Stage-2 target must be a genuinely future window frozen before its stats can exist.

### BETTING-LAYER CORRECTIONS — STILL IN FORCE

Until an evidence-backed construction V2 is frozen: alt direction conflicting with canonical main-line direction ⇒ **PASS**; raw yard cushion ≠ safety; use mean + SD + exact threshold + `P(clear)` + exact price; never label ~41% probability "Safe"; surface repeated-player and same-game concentration; no universal anchor across tiers; ATD uncertified and separate. Do not invent exposure caps or threshold rules from Week 7.

### EXISTING SCIENCE BOUNDARIES — PRESERVE

QB M89/M90 + C2 active; WR-R15 active; TE-R5P active; RB P3/R22 Week-1-only and **not** active Week 7; Stage-2 qualified in research but not production-wired; direct WR/CB unavailable/gated; ATD model uncertified; RB rushing Stage-1 source reset `SOURCE BLOCKED`; RB/TE team-context reset candidates `PREDICTIVE CLOSED`.

### EXACT CURRENT NEXT ACTIONS

1. re-fetch #535 (newest `6426050333`) and await / read the re-audit of the revised fixed target `aa70663c`;
2. **no predictive or scoring use of the card until that re-audit clears it**;
3. act on whatever the re-audit rules; if it clears, the first predictive step must be frozen before scoring;
4. Walker-first read-only opponent/environment transmission decomposition;
5. extend to representative WR/TE identities;
6. betting-layer construction V2 spec (direction conflict → PASS, clearance probability, price/EV, exposure caps, probability-aware tier labels);
7. port the stranded `claude/nice-cori-4qkwqs` tooling file-by-file;
8. grade Week 7 once the graders are ported to main.

No paid OddsAPI. No sportsbook upstream. No aggregate-MAE optimization. No post-hoc subgroup rescue. No Week-7 fitting. No reopening closed science without new causal evidence.

The next chat MUST re-fetch #535 and live GitHub before asserting that this remains current.
