# NFL HANDOFF — 2026-09-26 — WEEK-3 FULL SLATE REPAIRED / SYSTEMS-AUDIT LOCKS CURRENT

GitHub is canonical. Chat memory is secondary.

This handoff is intentionally compact enough for a new chat to have room, but complete enough to resume without recursively reading old handoffs.

## Read order for the next chat

Read only:

1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md` top checkpoint
3. this file
4. Issue #535 comments from `5850416444` onward
5. live `main`, plus any branch/run explicitly named below

Do **not** recursively ingest older handoffs unless this file explicitly sends you there.

---

# 1. CANONICAL PRODUCTION STATE

Current canonical main:

`4c0e9e25f0957877820cade6897cf977d265533a`

Merged commit message:

`Merge PR #645: repair Week 3 Full Slate rule authority`

PR #645 is merged and is the canonical Week-3 Full Slate repair.

PR #644 is closed/superseded because its useful identity fix was included in #645.

## What PR #645 fixed

Two separate confirmed production blockers were combined:

### A. Non-core unrostered sportsbook identity failure

Live Week-3 run `36275289734` paid for odds and failed on one row:

- Barion Brown
- NO
- `player_anytime_td`

He is a real active Saints WR but absent from the Ourlads scrape. Anytime TD is not one of the five priced markets.

Production now:
- still fails closed if an unrostered player appears on a **priced** market;
- quarantines an unrostered player on a **non-core / non-priced** market as `UNROSTERED_NONCORE_PLAYER`;
- still fails if the entire model-facing roster authority is clearly broken.

This keeps sportsbook identity defects from destroying a paid slate when the bad row cannot affect the board.

Side debt:
`manual_roster_overrides.csv` is orphaned because its only consumer, `repair_live_prop_identity_v1.py`, is no longer called from `full-slate.yml`. This is real technical debt but **not the current blocker**.

### B. Full-football Bayesian rule authority mismatch

The farthest real Week-3 failure was **not** the 30-team / 32-team guard.

Run `36276421905` passed the availability-aware eligible-team seam, then failed on:

`rules_tgt_share max_abs_diff=0.014064361123645175`

Root cause was localized to LAR.

Puka Nacua:
- existed in PlayerForm / ModelContext;
- raw target share = `0.3090024331`;
- Bayesian target share = `0.2647748824`;
- did **not** have a sportsbook-shaped metrics row.

The legacy injury target-redistribution path built its Bayesian share map only from sportsbook-shaped metrics rows. Therefore:
- full-football simulation used Puka's Bayesian share;
- pricing/rule replay fell back to his raw share;
- six LAR recipient target shares drifted after team redistribution.

Repair:
- load the complete PlayerForm Bayesian baseline once;
- pass that complete football authority into injury redistribution;
- preserve row-local fallback behavior for other callers;
- no coefficient, threshold, ensemble, synthesis, M38, WR-R15, TE-R5P or sportsbook-upstream change.

Exact failed-artifact replay:
- run `36281702615` = SUCCESS
- head `87c6ba73c67000de4d2953047e65c89e96527a8a`
- artifact `10919081863`
- digest `sha256:e3a9829738041cb61bf77d6de3a6eb180d0d4ac3bee91baf4caff2ef3a69039d`

Replay output:
- 30 teams / 15 games
- 421 football players
- 362 priced players
- 3,304 priced rows
- 833 player-market keys checked
- priced distribution misses = 0
- every football-assumption diff = 0.0, including `rules_tgt_share`
- sportsbook rows used to define football universe = 0
- RB Rush+Receiving V2 pathwise gap = 0.0

Canonical result:
`docs/production/WEEK3_FULL_SLATE_RULE_AUTHORITY_REPAIR_V1_RESULT.md`

## Post-merge certification

Canonical clean-main no-live-odds Full Slate:
- run `36283114203`
- event = push to main
- `FETCH_LIVE_ODDS=false`
- conclusion = **SUCCESS**

Repo CI on the same merged head:
- run `36283114216`
- conclusion = **SUCCESS**

Therefore:

**The current partial Week-3 football stack is mechanically certified on main.**

Do **not** reopen:
- the old 30-team / 32-team suspicion;
- Claude's reverted certified-slate direct edit;
- Claude's reverted game-exclusion mechanism;
- the old LAR drift diagnosis.

Those loops are closed.

## Exact next operational gate

To produce the actual Week-3 betting board, the next action is:

**controlled Full Slate with `fetch_live_odds=true` from canonical main.**

That consumes OddsAPI credits.

Do **not** launch it without explicit user authorization.

---

# 2. CURRENT LIVE PRODUCTION MODEL IMPROVEMENT

## RB Rush+Receiving Conservation V2 — PRODUCTION ACTIVE

This remains the clearest recent model-quality improvement actually live in Full Slate.

Non-Week-1 RB `rush_rec_yards` is built from the final standalone rush + receiving components rather than an independently inconsistent combo mean.

Historical 2024-2025, 2,787 RB player-games:
- MAE `27.6853 -> 25.5718`
- RMSE `39.8437 -> 36.4030`
- p90 AE `64.7742 -> 57.2604`
- 30+ yard misses `841 -> 768`
- candidate closer 59.13%

Replicated:
- 2024 MAE `28.2918 -> 25.9442`
- 2025 MAE `27.0783 -> 25.1991`

Week-2 observational confirmation:
- MAE `30.3620 -> 28.9014`
- signed bias `+20.1339 -> +9.8670`

Do not reopen V2 absent a concrete defect/new evidence.

---

# 3. FROZEN WEEK-3 RB VACANCY EXPERIMENT

Branch/history exists under the prospective RB vacancy work. Do not redesign.

Exact no-odds Week-3 Full Slate authority:
- run `36204768034`
- artifact `10892728623`
- digest `sha256:f52b36fb7a9c929fadca473bd8303823fb9f63bd42c884341cd6c3b52c26ed67`
- source main at lock: `f7d2011b73950488ea209124ba895b92c401b2b1`
- sportsbook upstream = 0

Exact baseline/candidate lock:
- run `36205758768`
- artifact `10893588588`
- digest `sha256:54437c69f0a66c2c9bb97c520f3e8d9c933e0203dac39fe1fde5352dafa3eeea`
- 25k MC
- seed 42
- outcomes attached = 0
- only `rules_rush_share` differs
- YPC / ML / State / ensemble weights unchanged

## DEN
Jonah Coleman definitive unavailable.

Vacancy share:
`0.357142857`

Frozen transfer:
- JK Dobbins +0.142857143
- RJ Harvey +0.214285714

Baseline -> candidate:
- Dobbins rush att `12.2933 -> 12.2741`; rush yd `47.9515 -> 47.7897`
- Harvey rush att `7.6729 -> 8.2607`; rush yd `35.5790 -> 39.7878`

## PIT
Rico Dowdle definitive unavailable.

Vacancy share:
`0.304347826`

Frozen transfer:
- Riley Nowakowski +0.012338425
- Jaylen Warren +0.292009401

Baseline -> candidate:
- Warren rush att `12.2887 -> 13.1161`; rush yd `49.9476 -> 56.8201`
- Nowakowski `2.5245 -> 2.2339`; yd `16.4666 -> 14.2082`
- Heidenreich `2.5242 -> 2.1700`; yd `16.4420 -> 13.7046`

Important:
finite top-five rushing allocation means changing one share redistributes finite room. Grade **all active RB/FBs on affected teams**, not just direct recipients.

After games are final:
1. grade Vacancy V1 independently;
2. do not redesign the transfer;
3. only then compare against public-intent labels.

---

# 4. FROZEN PUBLIC-INTENT LABELS

Branch:
`research-public-intent-week3-prospective-v1`

Current branch head at handoff:
`0170f89a42ca5c10e57c06685cd2736474dd45a2`

Contract:
`docs/research/PUBLIC_INTENT_PROSPECTIVE_CAPTURE_CONTRACT_V1.md`

Week-3 capture:
`docs/research/PUBLIC_INTENT_WEEK3_PROSPECTIVE_CAPTURE_V1.md`

Saturday addendum:
`docs/research/PUBLIC_INTENT_WEEK3_PROSPECTIVE_CAPTURE_V1_SATURDAY_ADDENDUM.md`

Frozen labels:

### DEN
`ROTATION_PRESERVED_NO_CLEAR_SUCCESSOR_CONCENTRATION`

Lead:
`NONE_CLEAR`

Confidence:
`MEDIUM_HIGH_NO_CONCENTRATION`

Saturday update:
- Coleman placed on IR;
- FB Adam Prentice elevated;
- no additional tailback elevated;
- label unchanged.

### PIT
`WARREN_LEAD_BACK_LEAN_WITH_DEPTH_SUPPORT`

Lead:
`JAYLEN_WARREN`

Confidence:
`MEDIUM`

Saturday update:
- Travis Homer signed active;
- Lew Nichols elevated;
- Dowdle OUT;
- Warren QUESTIONABLE;
- label unchanged;
- confidence not upgraded to workhorse certainty.

After games:
- grade RB Vacancy V1 first;
- then attach actual carry/snap absorption to frozen labels;
- never rewrite labels after outcomes;
- DEN/PIT alone cannot fit or promote a public-intent coefficient.

---

# 5. SYSTEMS-INTEGRITY AUDIT — IMPORTANT FINDINGS

Canonical ledger on research branch:
`docs/research/MODEL_SYSTEM_INTEGRITY_AUDIT_2026_09_26.md`

The project deliberately pivoted away from endless feature hunting toward:
- implementation/data-lineage bugs;
- math/architecture contradictions;
- weighting/blending errors;
- rule-order/availability transmission errors;
- semantic/unit defects;
- distribution-support defects.

## A. Availability -> opportunity rule-order gap CONFIRMED

Result:
`AVAILABILITY_OPPORTUNITY_RULE_ORDER_GAP_CONFIRMED`

Result doc:
`docs/research/AVAILABILITY_OPPORTUNITY_RULE_ORDER_AUDIT_V1_RESULT.md`

Authority:
- run `36275905038`
- artifact `10917451964`
- digest `sha256:340b5992a58ff591baf7a4ccf92c36482c1e22a082fabdc14e2f4e7dbcb1f6fb`

Week-3:
- 11 definitive-unavailable skill players
  - 6 WR
  - 3 TE
  - 2 RB/FB
- surviving eligible roles = 0
- surviving PlayerForm = 0
- surviving ModelContext = 0
- production-reachable `rules_injury_redistribution` rows = **0**

Meaning:
definitive unavailable players are correctly removed **before** the legacy WR vacancy rule can observe them.

Important correction:
vacated opportunity does **not** create huge residual.
Affected teams normalize survivor mass back to:
- modeled target probability = 0.95
- target residual = 0.05
- modeled rush probability = 0.95
- rush residual = 0.05

The unresolved problem is **successor identity/concentration**, not missing team mass.

Do not resurrect the old 60/30/10 rule or fit postgame transfer coefficients.

## B. Receiving-rule semantic defects CONFIRMED structurally

Plan:
`docs/research/RECEIVING_RULE_SEMANTICS_INTEGRITY_V1_PLAN.md`

Stage-1 result:
`docs/research/RECEIVING_RULE_SEMANTICS_INTEGRITY_V1_STAGE1_RESULT.md`

Run:
- `36276736046`
- artifact `10917254062`
- digest `sha256:c7ff51d17adfaf0f96e18808cec5948fa314f126f5c8b0392c8bed5568fa66fa`

### Defect A — middle-open unit mismatch

Production rule interprets `middle_open_rate` as 0-1 and compares to 0.50.

Week-3 source values are percentage points.

Evidence:
- 32/32 team rows have `middle_open_rate > 1`
- current gate effectively fires universally

Frozen A1 repair:
- values [0,1] unchanged;
- values (1,100] divided by 100;
- threshold remains exactly 0.50;
- no multiplier change.

A1B0 changes:
- 59 target-share rows
- all 59 TE
- median abs delta 0.012218
- max 0.028647
- 286 downstream entitlement rows
- non-target rules unchanged exactly.

### Defect B — slot alignment lost before rule labeling

PlayerForm preserves `alignment_position`, then normalizes `position` to generic WR and `role` to model hierarchy.

PlayerContext does not carry the alignment field.

Week-3:
- WR rows = 161
- preserved SWR rows = 56
- production SLOT labels = 0
- current WR1 = 30
- WR1_5 = 30
- unlabeled WR = 101

Frozen B1 repair carries existing SWR alignment through role labeling only.

A0B1:
- 56 SLOT labels restored
- 56 WR target-share rows changed
- median abs delta 0.014586
- max 0.029649
- 424 entitlement rows changed
- non-target rules unchanged exactly.

Combined A1B1:
- 77 target-share rows changed
- 59 TE / 18 WR
- 424 entitlement rows changed
- max entitlement delta 0.048108

### Stage-2 retrospective scoring CLOSED on source availability

Result:
`docs/research/RECEIVING_RULE_SEMANTICS_INTEGRITY_V1_STAGE2_RESULT.md`

Disposition:
`HISTORICAL_SOURCE_UNAVAILABLE_PROSPECTIVE_ONLY`

Source audit:
- run `36280290109`
- artifact `10918628297`
- digest `sha256:9ac7e1fd80adee0360c099087cc7aad3b7d801045a0e6ad5d9f57fe83df37ed2`

Historical 2025 maintained authority:
- 1,088 team-week rows
- no `middle_open_rate`
- no man/zone rate fields
- 8,688 player-week rows
- 3,231 WR rows
- source = weekly rosters
- historical alignment-like WR rows = 0
- SWR rows = 0

Week-1 long-retention artifact proves pregame LWR/SWR/RWR existed, but does not preserve enough upstream state for exact A/B.
Canonical Week-2 parent artifact expired.

Therefore:
- semantic defects remain real;
- A1/B1 are **not accuracy-qualified**;
- no Stage-3 production integration is authorized.

### Week-3 prospective receiving-semantics lock

File:
`docs/research/RECEIVING_RULE_SEMANTICS_INTEGRITY_V1_WEEK3_PROSPECTIVE_LOCK.md`

Commit:
`0170f89a42ca5c10e57c06685cd2736474dd45a2`

The exact pregame cells are frozen:
- A0B0 current
- A1B0 middle-unit repair
- A0B1 slot repair
- A1B1 combined

After games are final:
- attach actual targets/receptions/rec yards;
- do not recompute cells;
- score target-share AE, receptions MAE, rec-yard MAE, p90;
- pooled WR/TE, WR, TE, frozen slot subgroup;
- changed-row diagnostics;
- test one-game/team concentration.

No postgame rescue.

---

# 6. OTHER OPEN SYSTEMS-AUDIT LANES

## Discrete Count Mean Alignment V1

Issue #535 checkpoint `5850475777`.

Disposition:
`DISCRETE_COUNT_MEAN_ALIGNMENT_V1_QUALIFIED_FOR_INTEGRATION_TEST`

Historical corrected replay:
- run `36276366140`
- artifact `10916528395`
- digest `sha256:0563d974f8c25de290aabec9c92665c8e509fce99c2b83c0ec10e9c5a5c69102`

Candidate:
integer-preserving largest-remainder mean alignment for:
- `receptions`
- `rush_att`

No coefficient/threshold fit.

Results:

Receptions CRPS:
- 2024 `0.997154 -> 0.975917`
- 2025 `0.937276 -> 0.916401`
- pooled `0.967082 -> 0.946027`

Rush-att CRPS:
- 2024 `1.037831 -> 1.034776`
- 2025 `0.987607 -> 0.984803`
- pooled `1.012596 -> 1.009666`

Archived receptions lines:
- Brier `0.269746 -> 0.265809`
- log loss `0.745967 -> 0.734915`
- both improve independently in 2024 and 2025.

Current continuous alignment fractionalizes:
- 76.38% of receptions draws
- 31.58% of rush-att draws

Integration plan frozen:
`docs/production/DISCRETE_COUNT_MEAN_ALIGNMENT_V1_INTEGRATION_PLAN.md`

This is **research-qualified only**; do not assume production promotion.

Do not combine with the zero-MC issue below.

## Rush-att zero-MC ensemble transmission

Diagnostic:
`docs/research/RUSH_ATT_ZERO_MC_ENSEMBLE_TRANSMISSION_AUDIT_V1.md`

Issue #535 checkpoint `5850463671`.

Confirmed historical architecture:
`run_pricing_v2.py` can compute a nonzero calibrated ensemble mean while final distribution/model mean stays at zero when `mc_proj == 0`.

2024-25:
- 7,137 rush-att rows in zero-MC/nonzero-ensemble state

QB:
- 520 rows
- actual carries >0 in 420
- forced-zero MAE 2.2481
- intended ensemble MAE 1.5108
- ensemble closer 77.88%

RB:
- 161 rows
- actual >0 in 93
- MAE 1.4472 -> 1.2684

Blanket WR/TE injection gets worse, so **no generic repair is authorized**.

Most important lineage clue:
blocked QB/RB rows are usually reported as top-five by `rules_rush_share`, yet positive-MC rates collapse sharply by rank:
- rank 1: 100%
- rank 2: 98.7%
- rank 3: 75.9%
- rank 4: 36.2%
- rank 5: 4.0%
- rank 6+: 0%

That does not look like literal documented top-five selection.

Next valid action:
**exact allocation-lineage audit**
selector inputs -> selected membership -> multinomial probability -> realized MC mean -> lookup.

Do not invent a repair before the lineage audit.

---

# 7. CLOSED / DO-NOT-REPEAT LANES

Do not repeat or rescue these without genuinely new evidence:

- Receiver Room Targets-per-Play V1: failed blind 2024-25 confirmation.
- WR Anchor / Role-Transmission V1: participation leader materially worse than M38 anchor.
- TE Receiving-Yards Width V2: failed frozen CRPS/80%-coverage gates.
- Rush Pool Evidence Guard V1 integration: failed full-stack science.
- Rush Post-Ensemble Reconciliation V1: mechanically exact, scientifically bad; 24/33 gates failed.
- residual/top-five normalization rescue: current Week-3 residual state does not support the historical seam as a live normalization target.
- generic copula idea: mean-neutral dependence change cannot fix standalone marginal mean props.
- historical missing receiving-market weights: rec_yards/receptions already have promoted heldout weights; RB rush+rec handled by V2.
- retrospective M96 RB rushing search: terminal stop remains in force.

Do not launch another broad feature sweep just because production gains are limited.

---

# 8. ORDER OF OPERATIONS AFTER SUNDAY GAMES

These are separate experiments. Do not mix them.

1. Grade frozen RB Vacancy Opportunity V1.
2. Grade DEN/PIT public-intent labels against realized carry/snap concentration.
3. Grade Receiving Rule Semantics Week-3 A0/A1/B0/B1 prospective lock.
4. Preserve all original pregame labels/cells; no rewrites.
5. If a sample is insufficient, continue prospective capture rather than fit tiny-sample coefficients.
6. Keep sportsbook information downstream of football science.

---

# 9. USER PROCESS / GOVERNANCE

The user explicitly wants:
- actual model/science progress, not admin loops;
- bugs/math/weight/rule audits in parallel with feature research;
- hypotheses frozen before scoring;
- no hindsight rescue;
- failures closed cleanly;
- no repeated experiments;
- GitHub canonical over chat;
- Issue #535 paper trail;
- brief check-ins while work is happening;
- no paid OddsAPI pull without explicit approval.

When the user says "timed out," recover from the last GitHub checkpoint and continue. Do not ask them to re-explain.

---

# 10. EXACT NEXT ACTION

At this handoff moment:

- canonical Full Slate no-live-odds = GREEN on main;
- no current football-stack blocker remains from tonight's Week-3 repair;
- actual live betting board still requires a new controlled `fetch_live_odds=true` run;
- that run spends OddsAPI credits and therefore requires the user's explicit authorization.

Until they authorize it, do not launch a paid Full Slate.

If they authorize:
1. verify live main still equals or descends from `4c0e9e25f0957877820cade6897cf977d265533a`;
2. run canonical Full Slate with live odds;
3. do not modify model science while debugging any operational failure;
4. if it fails, diagnose the farthest real failure before editing anything;
5. preserve paid artifact immediately.

Research locks remain untouched while doing this.
