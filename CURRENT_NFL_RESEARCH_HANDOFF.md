# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-10-27 — V12 — ROLE ROUTING MECHANISM + ALLOCATOR SUPPORTED / NO WR ROLE READS ZONE / KUPP DOUBLE-PENALTY FRAMING WITHDRAWN / WALKER NOT TO BE REDONE / NO PREDICTIVE CLEARANCE

**Model:** PRISM — Player Role & Individual Situation Model

### WHAT THE NEXT SESSION SHOULD DO FIRST

Read newest #535 **after comment 6446039579**.

**The named-player trace core finding is SUPPORTED** (audit `6445134492`); its
three blockers are **FIXED** at revised executor target
**`4f95723154d5473b362f33c450b6f2d5fd8f6b60`** and awaiting final check. Audit
`6445801491` **supported the mechanism and the allocator chain** and required
two narrow truth corrections plus one caution; all three are landed. The next GPT's first job is to audit
that exact target/code/artifacts, then post the ruling.

The source-wide opponent/environment transmission corrections at research target **`8ed1d1778d388d54e6de9ed37ffc62d2d300e2ea` are CLEARED**. Walker's Week-7 exact case remains CLEARED and must **not** be redone.

The prior disagreement about named-player work is settled: **run/audit the compact JSN/Kupp/Barner named-player role-routing trace** because canonical role assignment is itself an untraced player-first mechanism step.

### WHAT THE TRACE FOUND (target `4f957231`, awaiting final check)

- script `scripts/research/trace_named_player_role_routing_v1.py`; doc `docs/research/PRISM_NAMED_PLAYER_ROLE_ROUTING_TRACE_V1.md`; artifacts `data/research/named_player_routing/`; 13 tests.
- In 2026 W7 all three authorized names are **SEA** players facing **KC**, so the same-offense/same-opponent contrast holds with no substitution. Walker is the **KC** back facing **SEA** in that same game — opposite sides of one matchup. Walker was **not** redone.
- **Answer to the mechanism question: WR transmission is rank-gated; TE and RB transmission is position-gated.** In `simulation_rules.py:201-210` the three WR branches key on the **computed role label** while the TE and RB/FB branches key on **position**. `_wr_role_labels()` labels slot plus only the top two perimeter WRs by target share. So every TE and every RB/FB gets an opponent-coverage target channel automatically, and any other WR falls through the whole chain to `tgt_mult = 1.0` with **no diagnostic recording it**. An unlabeled WR is the only offensive skill position with no opponent-coverage target channel at all.
- **Cooper Kupp misses the WR1_5 label by 0.008206 of raw target share** and is structurally ineligible for every opponent-coverage target gate, while SEA's **third-string TE** and its **fullback** are both eligible by position. JSN = WR1, Shaheed = WR1_5, Kupp/Horton/Foster = unlabeled, Barner eligible via the TE position branch.
- **The cutoff is discontinuous** — rank 2 vs rank 3 is the whole difference between a potential ±15% multiplier and structurally none.
- **The ranking key is inconsistent with the allocated quantity.** `_wr_role_labels` sorts by **raw** `tgt_share` (0.0 default) while production scales **`bayes_tgt_share`**. Montorie Foster is the clean case: raw key `NaN → 0.0` so he sorts last and gets no label, yet his posterior **0.165842** is the **second-largest in the room**, above both the labeled WR1_5 and Kupp.
- **Structural vs circumstantial zero kept apart.** Every SEA WR had 0.0 opponent-attributable movement, but Kupp was ineligible whatever the defense did while JSN was eligible and the gate did not fire. **Man-coverage gate certified NOT fired for KC** by exact argument: WR1 ×0.95 vs WR1_5 ×1.15 would split their `rules`/`bayes` ratios; both are exactly 1.000.
- **RETRACTED FROM V10: the rules/bayes ratio is NOT room normalization.** `apply_rules_to_metrics()` performs no team-room normalization at all, so the ratio **is** the rule-layer matchup multiplier. V10 called it a per-group room scale; that was wrong (audit `6445134492` blocker 1).
- **Because of that, the KC gate state is DECODED, not unknown** — a **source-conditioned inverse reconstruction**, NOT a reading of preserved `coverage_*_rate` values, which were not preserved. Each role's achievable multiplier products are distinct, so an exact match identifies the state uniquely: `coverage_zone_rate >= 0.60` **FIRED**, `middle_open_rate >= 0.50` **FIRED**, `coverage_man_rate >= 0.50` not fired, `pressure_mismatch > 0.075` not fired. Zone is decoded independently by the **TE and RB** groups and man independently by **WR1 and WR1_5**, and they agree — corroboration, not a single-group inference. Residual condition: holds for rows with no injury redistribution/limitation, which the uniform exact ratios evidence but do not prove.
- **Derived:** `volatility_mult = 1.0`; `pass_eff_mult` **bounded to {1.00, 1.01}** (not 0.98, since pressure did not fire). `rules_pass_eff_mult` is emitted by production but was not preserved, so field 8 is **BOUNDED, not resolved**.
- **BIGGEST FINDING: no WR role reads zone coverage at all.** `wr1`/`wr1_5` key only on `coverage_man_rate`; only `te_target_mult` and `rb_rec_target_mult` read zone. KC's zone gate fired, so zone coverage — the most common coverage family in the league — moved every SEA **TE by ×1.265** and every **RB/FB by ×1.200** and could not move one wide receiver. This is an **architectural** asymmetry and stands on its own.
- **RETRACTED FROM V10/V11: "Kupp is reduced twice over" and "his exclusion was not academic this week".** Both labelled WR multipliers were **1.000** because the man gate did not fire, and no WR role reads zone — so had Kupp been labelled WR1 or WR1_5 in this same frozen gate state, his coverage multiplier would **also** have been 1.000. His unlabelled status is **structural eligibility exposure with ZERO realized direct coverage effect** in this matchup: a missing potential channel, not an extra numeric penalty (audit `6445801491` correction 1).
- **The −28.0% M38 reduction must NOT be attributed to the KC zone gate or any opponent context.** The WR hierarchy sharpening stage is keyed on **WR rank**, not on the defense. Both consequences follow from Kupp's rank, but only one is an opponent channel and that one was **inert** this week. The −28.0% is the only realized reduction.
- **Fields 7/10 now read from the REAL allocator**, not inferred: `rules_tgt_share → post_m38_tgt_share → entitlement_tgt_share`. The WR hierarchy stage touches **only WRs** (TE/RB identical pre/post) and moves share from the non-top receivers to JSN: **Kupp −28.0%**, JSN **+29.3%**, Barner 0.0%. **So Kupp is reduced twice** — no coverage multiplier, then −28.0% at sharpening. Net `rules`→`entitlement`: JSN ×0.767, Barner ×0.785, **Kupp ×0.707**. Team mass conserved exactly (both stages sum 1.601148605998); room closes at **0.950000**, residual 0.05, `entitlement_gap` **5.55e-16**, 15 players.
- **The zero claim is SCOPED:** it is only the direct WR coverage-target rule-layer delta. It is **not** total opponent attribution and does **not** make any WR globally opponent-insensitive — shared `pass_eff_mult` can move YPT and the M38/entitlement stages move a WR's share for teammate-structural reasons.
- **NOT ESTABLISHED:** `pass_eff_mult` exactly (bounded above); the decode's no-injury-modifier condition (evidenced, not proven); **total opponent attribution to the final mean** — the traced chain is the target-share path only, and YPT, game script and mean construction are not accounted for; league-wide gate firing frequency; predictive value of consuming `ypt_allowed_*`.
- Guards hardened after audit: the unlabelled-WR fall-through check **raises** instead of recording a boolean; the two-label claim is proven by parsing the assignment statements (`{SLOT, WR1, WR1_5}` exactly, `perim` indices ⊆ `{0,1}`); and **`GATE_SPECS` is now derived from `rules_v2.matchup_multipliers` source** and compared semantically, so the mirror is a verified contract rather than an assertion. The derivation independently recovers all ten production multipliers, including `rb_rush_eff_mult`'s light/heavy-box gates. Mutation tests confirm both guards raise. `eligible_gates()` remains a hand mirror, not an executed production router.
- **Sidecar self-consistency enforced.** `not_established` is now **generated from** the gate decode, and `_self_consistency()` raises before the file is written if they disagree — audit `6445801491` found the committed V2 sidecar calling the TE/RB gates unknown while `gate_decode` had resolved them.
- **CI scope stated:** `repo-ci.yml` triggers only on `pull_request` and pushes to `main`, so the research branch cannot run it and **no PR was opened** without the user's request. Test counts are stated local results. `audit-only.yml` run **`40530603310`** on exact SHA `4f957231` is **green** (compileall + strict audit independently verified); it does **not** run pytest, so the 803 figure stays a local result.
- Zero-history entitlement defect (Foster above Kupp) recorded as an observation only and kept a **separate cause**.

Deep handoff:
`docs/handoffs/NFL_HANDOFF_2026-10-27_V9_SEAMLESS_CONTINUITY_NAMED_PLAYER_TRACE_AUTHORIZED.md`

Paste-ready restart prompt:
`docs/handoffs/NEXT_CHAT_PROMPT_2026-10-27_V9_NAMED_PLAYER_ROLE_ROUTING.md`

Displaced V8 checkpoint archived at:
`docs/handoffs/NFL_HANDOFF_2026-10-27_V9_ARCHIVED_TOP_V8_WALKER_GENERALIZATION_CORRECTED.md`

Do not recursively read older archived handoffs unless a specific lineage question requires one.

### ABSOLUTE ASSIGNMENT

PRISM is an **INDIVIDUAL-PLAYER PROJECTION MODEL**.

> **THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.**

> **ANOMALIES ARE QUESTIONS, NOT TRENDS.**

Position MAE/RMSE is downstream QA only.

Required reasoning order:
1. player identity;
2. current role/state;
3. opportunity allocation;
4. opponent/environment;
5. market-specific outcome;
6. reliability.

GPT = independent auditor / project control. Claude = executor / research lead. Shared lane = Issue #535.

### LIVE STATE AT V9 FREEZE

Before V9 continuity writes:
- `main` = **`e63611436dffa54caef061fbd1ca1c2da347d009`** at V10 write (advanced from `f3eef8b8` by the archive bot only, run `40500557725`; no science superseded)
- research branch `research-prism-player-first-identity-environment-2026-10-26` = **`4f95723154d5473b362f33c450b6f2d5fd8f6b60`**
- newest #535 = **6446039579** (corrected executor target posted; final check pending)
- cleared transmission target `8ed1d1778d388d54e6de9ed37ffc62d2d300e2ea`; governing audits `6441088661` (trace authorized) and `6445134492` (core supported, 3 blockers — fixed) and `6445801491` (mechanism + allocator supported, 2 wording/consistency fixes — landed)
- tests 803 passed (local; see CI scope above), 1 known pre-existing failure (stale no-credit `FETCH_LIVE_ODDS`; **do not weaken the guard**), `audit_repo.py --strict` clean
- betting CLOSED
- no paid OddsAPI authorized

Always re-fetch live state before asserting this remains current.

### IMMEDIATE AUTHORIZED TASK — AUDIT THE DELIVERED TRACE (was: run it)

Question:

> **How does canonical role assignment determine which opponent-transmission path a named player is actually eligible to receive?**

Read-only JSN/Kupp/Barner trace:
1. named player / target-week team / position;
2. canonical role/alignment label;
3. opponent fields present for that role;
4. eligible matchup gates;
5. gates that actually fire;
6. pre-gate -> post-gate opportunity share;
7. finite-room/team-normalization effect;
8. shared efficiency movement such as `pass_eff_mult`;
9. role-specific efficiency signal present-but-stranded;
10. final named-player opportunity/efficiency movement attributable to opponent context.

JSN vs Kupp = same-offense/same-opponent identity-routing contrast. Barner = TE route check.

If exact roster reality makes a name inapplicable, preserve the mechanism question and document the legal substitute.

### CLEARED TRANSMISSION STATE — CORRECT STRENGTH ONLY

Walker W7 exact case:
- Seattle rush-defense context existed;
- `def_rush_epa` did not reach Walker's rushing mean;
- explosive suppression carried but unconsumed;
- RB YPT/YBC/stuff not wired;
- neither 0.60 box gate fired;
- opponent-caused W7 Walker rushing opportunity/efficiency/mean/sigma movement = **0**;
- receiving did move through zone target-share rule.

Source-wide corrected state at `8ed1d177`:
- trace stages transport separately from consumption;
- `def_rush_epa` + `explosive_play_rate_allowed` = **TRANSPORTED-BUT-STRANDED**;
- independent generic ensemble = **MC / ML / State**; Bayesian is inside MC;
- `ml_v2` and `state_v2` are opponent-blind;
- MC can transmit opponent effects where a canonical rule exists and fires;
- box gate is a real RB opponent -> YPC -> MC-mean path;
- therefore correct claim is **outside the box gate**, generic RB rushing mean has no opponent run-defense transmission;
- gate firing frequency = **NOT ESTABLISHED**;
- receiving YPT has shared opponent pressure movement through `pass_eff_mult`;
- missing piece is role-specific opponent efficiency/conversion from `ypt_allowed_*`;
- direct generic-skill volatility multiplier is pressure-driven, but sigma also moves indirectly and promoted QB C2 has separate opponent-aware distribution logic;
- WR3+ / lower perimeter WRs can receive no canonical WR target multiplier;
- 108/88/63-style counts are literal source-reference inventory, not causal transmission counts;
- certified TeamContext dataflow = 17 fields populated / 13 read.

### MATERIALITY / ZERO-HISTORY — DO NOT REOPEN

A+B CLOSED.

Production-faithful W7 trace:
- 396 football-universe rows / 28 teams
- true zero-history / `position_prior_only` = **34**, across 21 teams
- `current_games=0` = **69**
- all fallback posteriors nonzero
- zero-history rush top-five = **13 rows / 11 teams**
- final rush allocation mass = **1.824446**
- zero-history target entitlement = **1.264686 / 26.6 = 4.7545%**
- **17** players >= .05
- priced-board fallback 13 players is sportsbook availability only and cannot bound football allocator population.

Retracted — never restate:
- 30.4% allocation-moving
- 1,730 zero-prior
- SEA 1.42 production prior mass
- “at most 13 could enter allocator”
- 36× framing

### IDENTITY CARD

Accepted descriptive substrate; no predictive promotion.

- 35,196 complete-roster cards
- 10,960 model-facing cards
- 3,053 players / 32 teams
- A CLOSED: injury contamination cannot be bypassed by NO_BASELINE
- B CLOSED: `TARGET_TEAM_TENURE_SHARE_V1`, participation-only tenure authority
- `PARTIAL_GAME` source-blocked for 2026
- threshold sensitivity **13.66%**, unresolved by design
- team shares descriptive/provisional, not predictive role authority

### DURABILITY

Forward explanatory-state archive is on `main` since `00a9ae21`.

Week 7 explanatory state recovered to git.

Residual gap:
- Weeks 1–6 explanatory state unavailable;
- current preservation retained opponent/environment column names but not values.

Future fix:
preserve opponent values + `rules_*_mult` so gate firing frequency can be measured on future runs.

### STILL OPEN

- hardcoded generic `pass_share = 0.57`
- lead/trail not altering generic pass/rush split
- future gate-frequency measurement once values are preserved
- Inside-5 TD design audit / future legal prospective test
- betting construction V2 spec
- orphan grading-tool port
- Week-7 grading after grader restoration

### FAIL-CLOSED / DESIGN-ONLY

Stage-2 Week 7 remains chronology fail-closed. No postgame shadow.

Inside-5 TD preregistration remains design-only:
- Week 7 generated hypothesis and cannot test it
- `MIN_I5_SUPPORT=10`
- inside-5 vs RZ correlation 0.890
- clip not main issue
- no scoring/promotion yet

### BETTING

Betting CLOSED.

Future construction rules remain:
- main-line direction conflict => PASS
- cushion != safety
- use mean + SD + exact threshold + P(clear) + price/EV
- probability/dependency-aware tiers
- surface repeated-player/same-game concentration
- no arbitrary exposure cap invented from W7
- ATD separate/uncertified

### NON-NEGOTIABLE BOUNDARIES

- no paid OddsAPI without explicit user approval
- no sportsbook upstream
- no Week-7 fitting
- no post-hoc subgroup rescue
- no named-player exception
- no aggregate-positional-MAE chase
- no reopening A/B
- no weakening `FETCH_LIVE_ODDS` guard to make CI green
- no wholesale merge of orphan `claude/nice-cori-4qkwqs`
- no frequency claim from source wiring alone
- Week 7 identifies mechanisms; it does not score fixes

### EXACT NEXT QUEUE

1. Re-fetch #535 after **6441088661**.
2. Audit Claude's compact JSN/Kupp/Barner role-routing target if posted; otherwise wait for/coordinate that exact authorized read-only executor item.
3. Fix future trace preservation to retain opponent values + multipliers.
4. Trace hardcoded pass-share / lead-trail game-script path.
5. Audit/freeze Inside-5 TD design before any legal test.
6. Betting-layer V2 spec.
7. Port orphan grading tooling file-by-file.
8. Grade Week 7.

After the next substantive milestone, supersede this root checkpoint in place and archive V9.
