# NFL HANDOFF — 2026-09-26 — FULL SLATE REPAIRED / SYSTEMS-AUDIT + WEEK-3 PROSPECTIVE LOCKS CURRENT

GitHub is canonical. This handoff is the immediate execution checkpoint and supersedes older handoffs for current work.

## 0. Memory-efficient start rule

A new chat should read only, in this order:

1. `AGENTS.md`
2. the **top checkpoint only** in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this file:
   `docs/handoffs/NFL_HANDOFF_2026-09-26_FULL_SLATE_REPAIRED_SYSTEMS_CURRENT.md`
4. Issue #535 comments:
   - `5851005226` — receiving semantics Stage 2 / prospective lock
   - `5851180199` — Week-3 Full Slate rule-authority repair exact replay
   - `5851353477` — repair merged + clean-main Full Slate PASS
   - then only comments posted after those
5. live `main`, relevant branch heads, and current workflow status.

Do **not** recursively read older handoffs unless this file explicitly directs you to one.

---

# 1. Current production authority

Canonical main:

`4c0e9e25f0957877820cade6897cf977d265533a`

Commit:
`Merge PR #645: repair Week 3 Full Slate rule authority`

PR #645:
**MERGED / CANONICAL**

PR #644:
superseded by the combined #645 production repair. Do not restart #644 as a separate lane.

Post-merge canonical checks on the exact main head:

- Full Slate no-live-odds run `36283114203`: **SUCCESS**
- Repo CI `36283114216`: **SUCCESS**
- Archive Market Track Record `36283268620`: **SUCCESS**

This proves the current **30-team / 15-game Week-3 partial football slate** can complete through canonical Full Slate from clean main with `FETCH_LIVE_ODDS=false`.

The earlier idea that a literal 32-team guard was the active blocker is superseded. The availability-aware partial-slate seam is functioning in the canonical no-odds run.

---

# 2. What actually broke the Week-3 Full Slate, and what is fixed

There were two confirmed production blockers.

## 2.1 Non-core sportsbook roster mismatch

Original live Week-3 run `36275289734` fetched odds and failed because Barion Brown / NO / `player_anytime_td` was not represented in the Ourlads-shaped roster authority.

He was a genuinely active player, but:
- the row was in a non-priced market;
- it could never contribute to the production board;
- it still killed the paid slate.

Canonical behavior now:
- unrostered players on the five priced markets still fail closed;
- unrostered **non-core / non-priced** sportsbook rows are quarantined instead of killing the slate;
- the fix was verified against the preserved real Week-3 odds without another OddsAPI request.

This does not feed sportsbook information upstream into football.

Side note:
`manual_roster_overrides.csv` remains orphaned from the canonical Full Slate path. That is an architecture/maintenance concern, but it is not the current blocker and should not be casually rewired.

## 2.2 LAR target-share rule-authority drift

The farthest real live failure was run:

`36276421905`

Preserved artifact:
`10917037948`

Digest:
`sha256:738422833937cd40ddbd7dd85ebf01741d11aec079eb681a7f8cf87818058f90`

Failure:
`rules_tgt_share max_abs_diff=0.014064361123645175`

It localized to six LAR recipients.

Root cause:
- Puka Nacua existed in PlayerForm / ModelContext / the complete Bayesian football authority;
- Puka had **no sportsbook-shaped metrics row**;
- the injury redistribution logic reconstructed Bayesian target shares only from the pricing-shaped rows it received;
- pricing therefore fell back to Puka's raw target share while the full-football simulation used his Bayesian posterior;
- the vacancy redistribution amount differed, which renormalized six LAR recipients.

Exact Puka values:
- raw PlayerContext target share: `0.3090024330900243`
- Bayesian target share: `0.2647748823867376`

Canonical repair:
- load the complete PlayerForm Bayesian baseline once;
- apply that complete football authority through the rule layer;
- injury redistribution can no longer depend on whether a player happens to have a sportsbook offer;
- legacy row-local behavior remains for callers that do not provide the full authority.

No coefficient, injury percentage, 60/30/10 rule, M38, WR-R15, TE-R5P, ML/State weight, QB/RB synthesis, or sportsbook-upstream behavior changed.

Exact failed-artifact replay:
- run `36281702615`: **SUCCESS**
- head `87c6ba73c67000de4d2953047e65c89e96527a8a`
- artifact `10919081863`
- digest `sha256:e3a9829738041cb61bf77d6de3a6eb180d0d4ac3bee91baf4caff2ef3a69039d`

Replay output:
- 30 teams
- 15 games
- 421 football players
- 362 priced players
- 59 football players with no priced offers
- 3,304 priced rows
- 833 priced player-market keys checked
- 0 priced-distribution misses
- **every protected football-assumption difference = 0.0**
- sportsbook rows used to define football universe = 0
- sportsbook inputs used to generate football distributions = false

Result doc:
`docs/production/WEEK3_FULL_SLATE_RULE_AUTHORITY_REPAIR_V1_RESULT.md`

---

# 3. Exact operational state for tomorrow's board

The football stack is mechanically green on canonical main.

The next operational gate for an actual Week-3 betting board is:

**one controlled Full Slate run from current main with `fetch_live_odds=true`.**

That consumes OddsAPI credits.

No post-repair paid/live run has been launched yet.

The user did **not** authorize another paid OddsAPI pull after the final repair during this chat. Obtain explicit authorization before launching it.

When authorized:
1. verify live main is still `4c0e9e25...` or understand any newer merged commit;
2. do not rerun old diagnostics first;
3. launch canonical `.github/workflows/full-slate.yml` with live odds enabled;
4. monitor through football build, live odds gate, pricing, certifications, board/workbook artifact;
5. if it fails, diagnose the **new farthest failure** rather than reopening Barion/Puka/32-team theories;
6. do not spend a second paid pull until the first new failure is understood and a preserved artifact can be reused.

The user explicitly needs the board for Sunday, so once authorization exists this is an operational priority.

---

# 4. Major production science currently live

## RB Rush+Receiving Conservation V2

Production-active and protected.

Non-Week-1 RB:
`rush_rec_yards = final rush_yards + final rec_yards`

Historical 2024-2025, 2,787 RB player-games:
- MAE `27.6853 -> 25.5718`
- RMSE `39.8437 -> 36.4030`
- bias `+14.3388 -> +9.3891`
- p90 AE `64.7742 -> 57.2604`

Final stable certification run:
`36009823313`

Do not reopen absent a concrete defect.

---

# 5. Systems-integrity pivot — why this matters

The user explicitly asked us to stop assuming that the answer to months of weak improvement was simply "more features."

Current program:
look for places where valid football information is:
- lost;
- overwritten;
- double-counted;
- scaled incorrectly;
- combined with inconsistent weights;
- removed before a downstream rule can use it;
- transformed into physically inconsistent distributions.

This is a **parallel systems-audit lane**, not an excuse to abandon the frozen Week-3 prospective experiments.

Canonical audit ledger on research branch:
`docs/research/MODEL_SYSTEM_INTEGRITY_AUDIT_2026_09_26.md`

Do not drift back into broad feature sweeps without a specific structural reason.

---

# 6. Confirmed systems findings

## 6.1 Availability -> Opportunity Rule-Order Gap

Disposition:
`AVAILABILITY_OPPORTUNITY_RULE_ORDER_GAP_CONFIRMED`

Authoritative run:
`36275905038`

Latest successful repeated run:
`36279453173`

Week-3 preserved state:
- 30 current eligible teams
- 11 definitive-unavailable RB/FB/WR/TE players:
  - 6 WR
  - 3 TE
  - 2 RB/FB
- unavailable surviving eligible roles: 0
- unavailable surviving PlayerForm: 0
- unavailable surviving ModelContext: 0
- production-reachable `rules_injury_redistribution` rows: **0**

Interpretation:
definitive-unavailable players are correctly removed **before** opportunity rules.
The legacy definitive-WR redistribution rule therefore cannot observe them in production.

Important correction:
the missing opportunity does **not** become a giant residual.
Affected teams already exceed the explicit-player 95% allocator cap, so production does:

`remove unavailable player -> lose vacancy-specific identity -> generic survivor normalization to 95%`

The unresolved issue is **successor identity/concentration**, not team-volume conservation.

No repair was promoted from this diagnostic.

Result:
`docs/research/AVAILABILITY_OPPORTUNITY_RULE_ORDER_AUDIT_V1_RESULT.md`

---

# 7. Frozen RB Vacancy Opportunity V1 — DO NOT TOUCH BEFORE OUTCOMES

This experiment is separate from production and separate from public-intent labels.

Immutable Week-3 football authority:
- no-odds Full Slate run `36204768034`
- artifact `10892728623`
- source main `f7d2011b...`

Vacancy baseline/candidate lock:
- run `36205758768`
- artifact `10893588588`
- 25,000 MC draws
- seed 42
- outcomes attached = 0
- sportsbook inputs = 0

Qualifying events:

## DEN
Jonah Coleman definitive unavailable.

Frozen transfer:
- vacancy share `0.357142857`
- JK Dobbins +`0.142857143`
- RJ Harvey +`0.214285714`

Baseline -> candidate:
- Dobbins rush att `12.2933 -> 12.2741`; rush yds `47.9515 -> 47.7897`
- Harvey `7.6729 -> 8.2607`; rush yds `35.5790 -> 39.7878`

## PIT
Rico Dowdle definitive unavailable.

Frozen transfer:
- vacancy share `0.304347826`
- Riley Nowakowski +`0.012338425`
- Jaylen Warren +`0.292009401`

Baseline -> candidate:
- Warren `12.2887 -> 13.1161`; rush yds `49.9476 -> 56.8201`
- Nowakowski `2.5245 -> 2.2339`; rush yds `16.4666 -> 14.2082`
- Heidenreich `2.5242 -> 2.1700`; rush yds `16.4420 -> 13.7046`

Finite top-rushing-pool normalization means a transfer can move other active backs too. Grade **all active RB/FBs on affected teams** postgame.

Do not:
- redesign the transfer;
- add QUESTIONABLE/DOUBTFUL;
- alter YPC;
- use target-game outcomes to change the lock.

Postgame: grade frozen Vacancy V1 first.

---

# 8. Frozen Week-3 RB public-intent labels — grade only after Vacancy V1

Branch:
`research-public-intent-week3-prospective-v1`

Frozen label files:
- `docs/research/PUBLIC_INTENT_WEEK3_PROSPECTIVE_CAPTURE_V1.md`
- `docs/research/PUBLIC_INTENT_WEEK3_PROSPECTIVE_CAPTURE_V1.json`
- Saturday addendum:
  `docs/research/PUBLIC_INTENT_WEEK3_PROSPECTIVE_CAPTURE_V1_SATURDAY_ADDENDUM.md`
- reusable contract:
  `docs/research/PUBLIC_INTENT_PROSPECTIVE_CAPTURE_CONTRACT_V1.md`

DEN:
`ROTATION_PRESERVED_NO_CLEAR_SUCCESSOR_CONCENTRATION`
lead = `NONE_CLEAR`

PIT:
`WARREN_LEAD_BACK_LEAN_WITH_DEPTH_SUPPORT`
lead = Jaylen Warren
confidence = MEDIUM

Saturday evidence did not change either frozen label.

Postgame order:
1. grade RB Vacancy V1 independently;
2. attach actual carry/snap absorption to the public-intent labels;
3. never rewrite the frozen labels;
4. DEN/PIT alone cannot fit/promote a public-intent coefficient.

---

# 9. Receiving Rule Semantics Integrity V1 — two real semantic defects, NOT production-promoted

Branch current frozen head:
`0170f89a42ca5c10e57c06685cd2736474dd45a2`

## Defect A — middle_open units

Production compares `middle_open_rate >= 0.50` as if 0-1.

Week-3 source arrives in percentage points.

Stage-1 evidence:
- 32/32 team-context rows had `middle_open_rate > 1`;
- current gate was effectively always true.

Frozen A1 only:
- values [0,1] unchanged;
- values (1,100] divide by 100;
- threshold remains 0.50.

Stage-1 A1B0:
- 59 changed target-share rows
- all 59 TE
- median abs delta 0.012218
- max 0.028647
- 286 downstream entitlement rows changed
- non-target rules exact invariant.

## Defect B — slot alignment lost

PlayerForm preserves `alignment_position`, then normalizes public position to WR.
PlayerContext does not carry the alignment.
Slot rule looks for SWR/SLOT and therefore labels **zero** current WRs as SLOT.

Stage-1:
- 161 WR rows
- 56 true SWR rows in PlayerForm
- A0 SLOT labels = 0

Frozen B1 carries the already-existing SWR alignment through the label step only.

A0B1:
- 56 SLOT labels restored
- 56 WR target-share rows changed
- median abs delta 0.014586
- max 0.029649
- 424 entitlement rows changed
- non-target rules exact invariant.

Combined A1B1:
- 77 target-share rows changed
- 59 TE / 18 WR
- 424 entitlement rows changed.

Stage-1 authoritative run:
`36276736046`

Artifact:
`10917254062`

Digest:
`sha256:c7ff51d17adfaf0f96e18808cec5948fa314f126f5c8b0392c8bed5568fa66fa`

## Stage 2 retrospective result

Disposition:
`HISTORICAL_SOURCE_UNAVAILABLE_PROSPECTIVE_ONLY`

Authoritative source audit:
- run `36280290109`
- artifact `10918628297`
- digest `sha256:9ac7e1fd80adee0360c099087cc7aad3b7d801045a0e6ad5d9f57fe83df37ed2`

Maintained historical team-week source:
- 1,088 rows
- no middle_open_rate/man/zone fields.

Maintained historical pregame universe:
- 8,688 player-week rows
- 3,231 WR rows
- 0 historical SWR/alignment rows.

Week-1 long-retention artifact proves live alignment existed (63 LWR / 61 SWR / 57 RWR) but does not retain enough upstream rule state for exact replay.
Exact Week-2 parent artifact expired.

Therefore:
- defects are confirmed;
- accuracy improvement is not historically qualified;
- Stage 3 production integration is **not authorized**.

## Week-3 prospective accuracy lock

File:
`docs/research/RECEIVING_RULE_SEMANTICS_INTEGRITY_V1_WEEK3_PROSPECTIVE_LOCK.md`

Commit:
`0170f89a42ca5c10e57c06685cd2736474dd45a2`

Pregame cells are already frozen:
- A0B0 baseline
- A1B0 middle-unit repair
- A0B1 slot carry
- A1B1 combined

After games are final:
- join actual targets/receptions/rec yards;
- grade the unchanged cells;
- pooled WR/TE + WR + TE + frozen slot subgroup;
- p90;
- changed rows;
- team/game concentration;
- no rescue/redesign after outcomes.

---

# 10. Discrete Count Mean Alignment V1 — qualified research, not production yet

Separate systems lane.

Disposition:
`DISCRETE_COUNT_MEAN_ALIGNMENT_V1_QUALIFIED_FOR_INTEGRATION_TEST`

Authoritative replay:
- run `36276366140`
- artifact `10916528395`
- digest `sha256:0563d974f8c25de290aabec9c92665c8e509fce99c2b83c0ec10e9c5a5c69102`

Problem:
production multiplicatively mean-aligns integer count distributions, creating fractional receptions/rush attempts.

Frozen candidate:
integer-preserving largest-remainder alignment.

Results:
Receptions CRPS pooled:
`0.967082 -> 0.946027`

Rush-att CRPS pooled:
`1.012596 -> 1.009666`

Receptions archived-line:
- Brier `0.269746 -> 0.265809`
- log loss `0.745967 -> 0.734915`

Current production fractional support:
- receptions: 76.38% of draws
- rush_att: 31.58%.

Integration plan is frozen:
`docs/production/DISCRETE_COUNT_MEAN_ALIGNMENT_V1_INTEGRATION_PLAN.md`

Plan commit:
`1ff9ccca6417e74eb58ae18c9bf2391da0e6eb51`

No production promotion yet.
Do not combine this with the separate zero-MC issue during integration.

---

# 11. Rush-att zero-MC transmission finding — diagnostic only

Separate file:
`docs/research/RUSH_ATT_ZERO_MC_ENSEMBLE_TRANSMISSION_AUDIT_V1.md`

Confirmed architecture:
pricing can compute a nonzero calibrated ensemble projection but final distribution/model mean remains zero whenever MC mean is zero, because production only mean-aligns when `mc_proj > 0`.

Historical exact authority:
- 7,137 rush-att rows in zero-MC/nonzero-ensemble state.

QB:
- 520 blocked rows
- actual carries >0 in 420
- forced-zero MAE 2.2481
- intended ensemble MAE 1.5108
- ensemble closer 77.88%.

RB:
- 161 blocked rows
- actual >0 in 93
- MAE 1.4472 vs 1.2684.

WR/TE worsen under blanket injection, so **no generic repair is authorized**.

Next legitimate step in this lane:
exact allocation-lineage audit:
selector inputs -> selected membership -> multinomial probability -> realized MC mean -> lookup.

Do not design a repair before that lineage is understood.

---

# 12. Confirmed rushing architecture contradiction — repair family closed

`POST_SPECIALIST_CROSS_MARKET_CONSISTENCY_V1_CONTRADICTION_CONFIRMED`

Final independently blended RB rush_att / rush_yards can produce a football state inconsistent with joint-MC YPC/opportunity.

But:
`RUSH_POST_ENSEMBLE_RECONCILIATION_V1_FAILED_CLOSED`

Forcing final markets back to MC team carry mass made 2024/2025 accuracy materially worse because historical MC player carry mass itself was badly underallocated.

Do not rescue with:
- partial factor;
- cap/floor;
- RB-only;
- QB exclusion;
- high-volume router;
- alternate factor per market.

The contradiction is real; that exact repair family is closed.

---

# 13. Closed research families — do not repeat

Do not restart:
- Receiver Room Targets-per-Play V1 confirmation;
- WR participation replacing M38;
- WR Anchor / Role Transmission;
- TE Width V2;
- Rush Pool Evidence Guard production integration;
- Rush Post-Ensemble Reconciliation;
- residual-normalization / sparse-top5 historical rescue;
- generic copula idea for standalone marginal props;
- old three-market missing-weight experiment;
- M96 retrospective RB rushing router family.

If proposing something adjacent, first prove it is not one of these under a new name.

---

# 14. What tomorrow changes

Week-3 target games are Sunday Sep 27, 2026.

Before games are final:
- do not grade RB Vacancy V1;
- do not grade public-intent labels;
- do not attach outcomes to receiving semantic A/B;
- do not modify any frozen pregame cell.

After games are final:
1. grade RB Vacancy V1;
2. grade DEN/PIT public intent against realized usage;
3. grade Receiving Rule Semantics frozen A/B;
4. record all results before designing any follow-up.

These are independent analyses. Do not use one result to rewrite another lock.

---

# 15. Immediate next-chat priorities

Priority depends on the user's next instruction.

## If the user wants the actual Sunday betting board
The football stack is green.
Get explicit approval for one paid live-odds Full Slate, then run canonical Full Slate from current main and follow the new farthest failure/output.

## If the user wants to continue model science before outcomes
Stay in systems-integrity mode.
Best unresolved structural lane is the rush-att zero-MC allocation lineage.
Do not touch frozen Week-3 prospective experiments.

## If games are final
Grade the three frozen prospective programs in the exact order in §14.

---

# 16. Collaboration / behavior requirements

- GitHub is canonical.
- User does not want to re-explain.
- Do not recursively reread old handoffs.
- Every substantive hypothesis/test/result/disposition goes in GitHub + Issue #535.
- Finish loops before switching.
- Do not report a research finding as production unless actually merged/certified.
- Do not rerun failed experiments without genuinely new evidence.
- No paid OddsAPI call without explicit user approval.
- Sportsbook data remains downstream only.
- User wants actual model improvement, not scoreboard/admin loops.
- When a run fails mechanically, repair only the mechanical issue; do not silently change science.
- When science fails frozen gates, close it. No rescue search.
