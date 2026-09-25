# NFL HANDOFF — 2026-09-25 — RECEIVER ROOM FRONTIER / TARGETS-PER-PLAY CONFIRMATION ACTIVE

Repo: `dkaps6/imtiredofthis`

GitHub is canonical over chat memory.

This handoff is intentionally compact enough for a fresh GPT-5.6 chat to load without exhausting context, but complete enough to continue seamlessly. Do **not** recursively load older handoffs at startup.

---

## 0. READ ORDER FOR NEXT CHAT

Read only:

1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md` top checkpoint
3. **this file**
4. Issue #535 from comment `5834428368` onward, especially `5837470764`
5. live branch/run state named below

Then work immediately.

Older docs/results are reference-only unless this handoff explicitly points to one.

---

# 1. CURRENT CANONICAL / PRODUCTION STATE

Physical `main` before this docs-only handoff:

`5421ba24b28aeff88e1f6466d93970f01264ebfa`

Recent receiver research has **not changed production**.

Protected production science still includes:

- QB mean: M89/M90 / `QB_PASS_SYNTHESIS_V1`
- QB distribution: `C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1`
- WR: M38 + WR-R15 where authorized
- TE: TE-R5P
- RB Rush+Receiving Conservation V2 production-active
- availability-first current roster plumbing
- sportsbook downstream only

No paid OddsAPI pull was authorized or used in this research sequence.

---

# 2. ACTIVE PRIORITY RIGHT NOW

## Receiver Room Targets-Per-Play V1 — 2024-2025 unchanged confirmation

Branch:

`research-receiver-room-targets-per-play-v1-confirm-2024-2025`

Current branch head:

`31e1854be631eeae3b5c6d25aa2c61b3a6c51925`

Frozen plan:

`docs/research/RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_CONFIRMATION_PLAN.md`

Active run at handoff time:

`36173485673`

Status at handoff creation:

**IN PROGRESS**

Do not duplicate it.

### Exact unchanged formula

For room g in WR / TE / RB_FB:

`R_g_play = sum(strict-prior room targets) / sum(strict-prior offensive plays)`

`candidate room targets = projected offensive plays * R_g_play`

History:
- prior season regular-season games for team T;
- completed current-season regular-season games strictly before target week;
- league fallback only if zero eligible prior team plays.

No:
- shrinkage;
- recency weighting;
- pseudo-count;
- fitted coefficient;
- bias offset;
- room multiplier;
- WR1/Q4 exception;
- sportsbook input.

### Confirmation baseline ordering

Critical:

- 2024: M38 explicit entitlement -> fold-safe TE-R5P -> authorized fold-safe WR-R15
- 2025: M38 explicit entitlement -> fold-safe TE-R5P only
- **WR-R15 retrospective 2025 application is forbidden**

Candidate is still room-level only. No player yards/receptions/rushing/QB/ATD changes in this phase.

### Frozen confirmation disposition

Pass only if all 27 frozen gates pass.

If pass:
`RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_CONFIRMED`

If any gate fails:
`RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_FAILED_CLOSED`

No rescue.

---

# 3. WHY THIS LANE EXISTS — RECENT SCIENCE CHAIN

This is the important reasoning trail. Do not restart it.

## A. Shared Pass-State Coherence V1 — structural contradiction CONFIRMED

Run:
`36073919850`

Artifact:
`10839371645`

Disposition:
`SHARED_PASS_STATE_COHERENCE_V1_STRUCTURAL_DIVERGENCE_CONFIRMED`

Current production can simulate a C2 QB passing game and a separate canonical receiver game on the same draw.

Current Week-3 audit showed:
- median corr(final C2 QB pass yards, canonical modeled receiver total): ~`0.5575`
- median canonical-vs-C2-shadow individual receiver array corr: ~`0.0399`
- median player p90 draw gap: ~`30.84 yd`
- Q4 p90 draw gap: ~`49.53 yd`
- canonical receiver engine could produce zero receptions + positive receiving yards on ~`16.42%` median player draws
- exact C2 completed-pass process: 0%

This was real architecture evidence.

## B. One-Pass-State Integration V1 — FAILED CLOSED

Run:
`36078172403`

Artifact:
`10841142451`

Result:
`docs/research/ONE_PASS_STATE_INTEGRATION_V1_RESULT.md`

Disposition:
`ONE_PASS_STATE_INTEGRATION_V1_FAILED_CLOSED`

Positive:
- receiving-yard macro MAE `16.234770 -> 16.171014`
- 2024 and 2025 both improved
- WR/TE/RB pooled MAE each improved
- RB rush+receiving MAE improved both seasons

Failed:
- rec-yard macro p90 worsened
- Q4 MAE/p90 worsened
- receptions MAE microscopically worse
- RB combo p90 worsened

Interpretation:
shared state has signal, but wholesale receiver marginal replacement over-corrects high-authority receivers.

Do not rescue One-Pass with Q4/position thresholds or carveouts.

## C. Hierarchical Receiver Mean Reconciliation — FAILED CLOSED

Run:
`36081448987`

Artifact:
`10842317747`

Disposition:
`HIERARCHICAL_RECEIVER_MEAN_RECONCILIATION_V1_FAILED_CLOSED`

Positive mean signal:
- macro rec-yard MAE `16.234770 -> 16.106198`
- WR/TE/RB each improved
- RB combo MAE improved both seasons

But:
- macro p90 worsened `34.326908 -> 35.125815`
- 40+ miss rate worsened
- absolute bias worsened
- Q4 MAE/p90 worsened
- RB combo p90 worsened

This reinforced that aggregate correction can improve average accuracy while damaging alpha/high-authority tails.

No threshold rescue.

---

# 4. OPPORTUNITY PARTITION DISCOVERY

## Opportunity Partition Semantics V1 — CONFIRMED

Run:
`36141882752`

Artifact:
`10867051736`

Canonical result:
`docs/research/OPPORTUNITY_PARTITION_SEMANTICS_V1_RESULT.md`

Disposition:
`OPPORTUNITY_PARTITION_SEMANTICS_V1_CONFIRMED`

Repo semantics prove:
- `rules_pass_rate` is dropbacks / plays
- QB pricing converts dropbacks to official attempts
- canonical receiver targets are allocated directly from unconverted dropbacks
- canonical rushing pool is `plays - dropbacks`

Current Week-3 magnitude:
- median pass attempts/dropback ~`0.8731`
- median receiver target-pool inflation vs official attempts ~`14.54%`
- median excess pool ~`4.094 opportunities/team`
- median named target mass from non-attempt dropbacks ~`3.884 targets/team`

Important:
this semantic mismatch is real, but it did not fully explain QB/receiver yard reconciliation.

Rushing/scramble correction was deliberately kept separate.

---

# 5. OFFICIAL ATTEMPTS FAILED; TARGETABLE DROPBACKS EMERGED

## Receiver Official Attempt Pool V1 — FAILED CLOSED

Run:
`36142802477`

Artifact:
`10868246439`

Disposition:
`RECEIVER_OFFICIAL_ATTEMPT_POOL_V1_TEAM_CALIBRATION_FAILED_CLOSED`

Official attempts did not improve official-attempt prediction:
- pooled attempts MAE `6.459381 -> 6.578260`
- p90 worsened

But receiver team-target prediction improved:
- 2024 target MAE `6.505208 -> 6.171176`
- 2025 `6.671869 -> 5.972131`
- pooled `6.588539 -> 6.071654`

So “official pass attempts” was the wrong receiver denominator, but there was clearly a targetable-opportunity signal.

## Receiver Targetable-Dropback V1 — TEAM LEVEL FOUR-SEASON SUPPORT

Exact frozen formula:

`R_T = sum(strict-prior team targets) / sum(strict-prior team dropbacks)`

`candidate target pool = projected dropbacks * R_T`

Same cumulative-count formula improved team target MAE independently in:
- 2022
- 2023
- 2024
- 2025

2022-2023 pooled:
- MAE `6.629337 -> 6.159377`
- RMSE `8.147865 -> 7.890454`
- p90 `13.480627 -> 12.640581`

2024-2025 pooled:
- MAE `6.588539 -> 6.225454`
- RMSE `8.166840 -> 7.972908`
- p90 `13.150607 -> 13.124031`

The team-volume science remains valid.

---

# 6. TARGETABLE FULL-STACK PLAYER TRANSLATION — FAILED CLOSED

Branch:
`research-receiver-targetable-dropback-v1-full-stack`

Run:
`36147357028`

Job:
`108111725439`

Artifact:
`10869364930`

Result:
`docs/research/RECEIVER_TARGETABLE_DROPBACK_V1_FULL_STACK_RESULT.md`

Disposition:
`RECEIVER_TARGETABLE_DROPBACK_V1_FULL_STACK_FAILED_CLOSED`

Integrity/provenance all passed.

Player results:

Receptions macro MAE:
`1.290256 -> 1.290744`

- WR `1.440661 -> 1.451920`
- TE `1.285236 -> 1.290029`
- RB `1.144872 -> 1.130283`

Receiving-yard macro MAE:
`16.237196 -> 16.196135`

- WR `21.972878 -> 21.981999`
- TE `15.599941 -> 15.640045`
- RB `11.138769 -> 10.966360`

But:
- macro rec-yard p90 `34.211309 -> 35.543674`
- macro abs bias `2.511945 -> 4.489480`
- Q4 MAE `26.542664 -> 26.865304`
- Q4 p90 `55.510605 -> 58.121353`
- RB combo MAE worsened both seasons
- RB combo p90 worsened

Raw MC already worsened WR/TE, so ensemble was not the culprit.

Also confirmed:
`tgt_share = player targets / team targets`

So multiplying entitlement by targetable rate was mathematically valid; no denominator double-count.

No V1 rescue.

---

# 7. CURRENT-STACK RECEIVER COMPENSATION AUDIT — KEY BREAKTHROUGH DIAGNOSTIC

Run:
`36151138507`

Artifact:
`10871109942`

Disposition:
`CURRENT_STACK_RECEIVER_COMPENSATION_AUDIT_V1_COMPLETE`

It explained where the team-level signal was getting lost.

Team target MAE:
`6.591825 -> 6.225521` — improved

But room behavior:

- WR room MAE `4.551774 -> 4.893041` — worse
- TE room MAE `2.869768 -> 2.720220` — improved
- RB/FB room MAE `2.342531 -> 2.283111` — improved

WR:
- WR1 targets `2.782160 -> 3.000603` — worse
- WR2+ targets `1.882355 -> 1.835673` — improved

Q4:
- targets `2.578471 -> 2.740431` — worse
- deterministic yards `25.936887 -> 26.314885` — worse

2024 WR room bias:
`-0.354 -> -3.299 targets`

2025 WR room bias:
`+0.813 -> -2.109`

Interpretation:
- team target volume is too high;
- WR room / top-end receiver opportunity is too low;
- TE/RB room opportunity is relatively high;
- uniform team thinning exposes this compensation.

This is why the research moved to room opportunity.

---

# 8. WR1 CURRENT-STATE DIAGNOSTIC — SIGNAL REAL, WR1-ONLY CANDIDATE REJECTED

Branch:
`research-wr1-current-state-anchor-diagnostic-v1`

Successful run:
`36171050355`

Artifact:
`10880365550`

Result branch head:
`14a0c22fa6ed10597a372d204b098a887bee5f0e`

Verdict:
`candidate_justified = false`

Discovery 2022-2023 only.

Absolute WR1 team-share signal was strong:

M38 WR1 team target-share MAE:
`0.082427`

Validated blend-4 current-state:
`0.071169`

State-gap vs needed correction:
- sign agreement `70.40%`
- targetable-hurts cohort sign agreement `89.69%`

But room-conserved WR1 interpretation failed:

M38 WR1 within-room MAE:
`0.127663`

State-normalized within-room:
`0.152165`

Therefore:
- real WR1 absolute-share state signal
- **not** a WR1-within-fixed-room solution
- no WR1-only candidate
- no M38 multiplier retune from this result

---

# 9. ROOM TARGETABLE-RATE V1 — FAILED CLOSED BUT IDENTIFIED UPSTREAM DENOMINATOR BIAS

Branch:
`research-receiver-room-targetable-rate-v1`

Run:
`36171462750`

Artifact:
`10880311081`

Result:
`docs/research/RECEIVER_ROOM_TARGETABLE_RATE_V1_RESULT.md`

Disposition:
`RECEIVER_ROOM_TARGETABLE_RATE_V1_FAILED_CLOSED`

Formula:

`R_g = prior room targets / prior team dropbacks`

`candidate room targets = projected dropbacks * R_g`

It improved almost every room error/tail metric:

Pooled:
- macro MAE `3.3379 -> 3.1853`
- macro p90 `6.7363 -> 6.5594`
- closer rate `52.18%`

But macro absolute bias worsened:
`0.3238 -> 0.8003`

Attribution:
- omitted OTHER targets ~0.03/game only
- summed room candidate bias ~`-2.401 targets/game`
- team targetable candidate bias ~`-2.398`
- fixed 57% projected dropbacks are low vs actual by ~`2.85/game` in 2022 and `3.40/game` in 2023

Therefore exact dropback-denominator formulation is CLOSED.

Do not “fix” it with a bias offset or pass-rate retune.

---

# 10. ACTIVE-ROSTER RECEIVER ROOM STATE V1 — NOT SUPPORTED

Branch:
`research-active-roster-receiver-room-state-v1`

Run:
`36172146432`

Job:
`108194031342`

Artifact:
`10880536886`

Digest:
`sha256:10aca2c157a6269ada4487787b1c5e671a5df2a51aec34d3854161b4a1ee54de`

Result commit:
`ac553f584281091373eca70cbd161778bcceb65d`

Result:
`docs/research/ACTIVE_ROSTER_RECEIVER_ROOM_STATE_V1_RESULT.md`

Disposition:
`ACTIVE_ROSTER_RECEIVER_ROOM_STATE_V1_NOT_SUPPORTED`

Pooled:
- macro composition MAE `0.081330 -> 0.080858`
- macro oracle targets `2.450679 -> 2.442002`
- WR composition `0.092917 -> 0.093338` — worse
- WR oracle target MAE `2.818050 -> 2.836944` — worse
- TE/RB_FB modestly improved

WR gates failed in both 2022 and 2023.

Coverage was incomplete, especially full WR room state.

No rescue:
- no WR-only route
- no coverage threshold
- no current-games threshold
- no 2024-2025 peek
- no position-specific weighting

---

# 11. CURRENT WINNER: RECEIVER ROOM TARGETS-PER-PLAY V1

Branch:
`research-receiver-room-targets-per-play-v1`

Result branch head:
`602e0c86eaac9e38a2ec65e3dc68e46203dac3e3`

Run:
`36172644864`

Job:
`108195660546`

Artifact:
`10880948137`

Digest:
`sha256:961fcaa805e13818ade3473ee87799558c97c17317fab0913b2d893fd04835ff`

Result:
`docs/research/RECEIVER_ROOM_TARGETS_PER_PLAY_V1_RESULT.md`

Disposition:
`RECEIVER_ROOM_TARGETS_PER_PLAY_V1_SUPPORTED`

2022-2023 only. All 23 frozen gates passed.

### Pooled

Macro room MAE:
`3.337876 -> 3.168127`

Macro p90:
`6.736290 -> 6.434198`

Macro abs bias:
`0.323767 -> 0.253029`

Rooms:
- WR MAE `4.649635 -> 4.482915`
- WR p90 `9.517370 -> 9.017551`
- TE MAE `2.814755 -> 2.591002`
- TE p90 `5.493863 -> 5.173002`
- RB_FB MAE `2.549238 -> 2.430465`
- RB_FB p90 `5.197636 -> 5.112040`

Summed room:
- MAE `6.213005 -> 5.974676`
- abs bias `0.899338 -> 0.759088`
- p90 `12.609038 -> 12.322572`

All-room candidate closer:
`54.02%`

### 2022

Macro MAE:
`3.440250 -> 3.209260`

WR:
`4.761250 -> 4.498155`

TE:
`2.828880 -> 2.594462`

RB_FB:
`2.730621 -> 2.535164`

### 2023

Macro MAE:
`3.235879 -> 3.127145`

WR:
`4.538430 -> 4.467731`

TE:
`2.800682 -> 2.587555`

RB_FB:
`2.368523 -> 2.326150`

### Why this is distinct from C1

C1 is closed and must stay closed.

C1:
- preserved existing modeled receiver total
- redistributed WR/TE/RB_FB composition
- used last-8 team history + fixed 105-target league pseudo-count
- failed player-level protection

Targets-per-play V1:
- forecasts **absolute room target opportunity**
- cumulative strict-prior room targets / offensive plays
- does not preserve old total receiver mass
- does not use C1 pseudo-count/window
- does not alter QB/pass-rate production

---

# 12. EXACT NEXT ACTION

## First: inspect run 36173485673

Do not create another confirmation run unless the existing run is mechanically invalid.

### If it passes

1. Document exact run/job/artifact/digest/result in Issue #535.
2. Commit canonical 2024-2025 confirmation result.
3. Freeze a **new player/full-stack integration plan before scoring**.
4. Player integration must:
   - preserve each confirmed room target total;
   - allocate within room using already-authorized current within-room entitlement proportions;
   - preserve M38;
   - preserve TE-R5P;
   - preserve 2024 WR-R15 authority;
   - no retrospective WR-R15 in 2025;
   - leave catch rate and YPT unchanged;
   - leave QB pass yards unchanged;
   - leave rush attempts/rush yards unchanged;
   - leave ATD unchanged;
   - re-evaluate RB rush+receiving through current RB V2;
   - explicitly protect WR/TE/RB receptions + receiving-yard MAE/p90/bias/Q4 tails;
   - sportsbook inputs 0;
   - parameters fit 0;
   - one candidate.

5. If player/full-stack qualifies historically, require prospective 2026 confirmation before production promotion.

### If it fails scientifically

Close exact targets-per-play formula.

Do not:
- add shrinkage/recency;
- blend with fixed57;
- add room multipliers;
- exempt WR1/Q4;
- change specialist ordering;
- add bias offsets;
- use sportsbook;
- fit 2026.

Move to a genuinely different hypothesis.

### If it fails mechanically

Repair only the exact bounded mechanical/provenance defect. Do not alter formula, gates, baseline ordering, cohort, or science.

---

# 13. IMPORTANT CLOSED / DO-NOT-RETEST LANES

Do not reopen:

- Rush Pool Evidence Guard V1 production integration
- TE Width V2
- C1 group target-mass calibration
- C3 joint group-mass + conservation
- One-Pass-State V1
- hierarchical receiver mean reconciliation V1
- official-attempt receiver pool V1
- uniform targetable-dropback player thinning V1
- receiver room targetable-rate/dropback-denominator V1
- active-roster receiver room-state V1
- WR1-only current-state anchor candidate
- WR-R11 NGS residual target formulation
- generic TE target-pool boost
- generic attempt-semantics C4
- Migration 18/20/21 pass-rate retune
- fixed57 pass-rate replacement
- retrospective RB router/threshold/feature research closed by M96E
- Rush Pool V1 RB/QB/OTHER carveout rescues
- sportsbook-upstream routing
- global SD rescale

Failed experiments may generate new hypotheses, but their exact formulations remain closed.

---

# 14. STANDING SCIENCE LESSONS

These should guide the next chat:

1. **Team opportunity can improve while player projections worsen.**
   Always test the full chain.

2. **WR/Q4 underprojection is real and can coexist with excessive total team target volume.**
   Do not assume a global reduction should hit every player equally.

3. **TE/RB room volume has often benefited from opportunity corrections while WR room/top-end has not.**

4. **Absolute WR1 current-state information exists, but fixed-room WR1 rerouting failed.**
   Do not convert that into an M38 retune.

5. **The fixed57 pass/dropback production state is protected.**
   Targets-per-play bypasses it for receiver room opportunity; it does not replace it globally.

6. **Room targets-per-play is currently the cleanest signal because it fixes the denominator without retuning broader game/pass science.**

7. Every candidate must preserve:
   - leakage safety
   - sportsbook independence
   - frozen before scoring
   - season stability
   - player/tail protection
   - explicit fail-closed behavior

---

# 15. IMPORTANT AUTHORITIES / ARTIFACTS

- One-Pass V1: run `36078172403`, artifact `10841142451`
- Hierarchical mean reconciliation: run `36081448987`, artifact `10842317747`
- Opportunity semantics audit: run `36141882752`, artifact `10867051736`
- Official attempt pool: run `36142802477`, artifact `10868246439`
- Targetable 2024-25 team confirmation: run `36145789611`, artifact `10869935285`
- Targetable full stack: run `36147357028`, artifact `10869364930`
- Compensation audit: run `36151138507`, artifact `10871109942`
- WR1 current-state: run `36171050355`, artifact `10880365550`
- Room targetable-rate: run `36171462750`, artifact `10880311081`
- Active-roster room state: run `36172146432`, artifact `10880536886`
- Room targets-per-play 2022-23: run `36172644864`, artifact `10880948137`
- Active confirmation now: run `36173485673`

Latest Issue #535 checkpoint written by this chat:

`5837470764`

---

# 16. COLLABORATION / OPERATING RULES

The user is mentally exhausted from months of experimentation and wants the model/science process to carry more of the decision burden.

Do not ask the user to invent the next experiment.

Do:
- investigate architecture;
- find contradictions;
- audit why valid upstream signals are lost downstream;
- freeze one football-grounded candidate at a time;
- keep GitHub paper trail current;
- close loops before switching;
- challenge weak ideas;
- preserve positive sub-signals from failed candidates without rescuing the failed candidate itself.

Chat updates should be concise. GitHub handoffs/results should be exhaustive.

---

# 17. HANDOFF STATE

At the moment this handoff was written:

- production behavior unchanged;
- room targets-per-play 2022-23 = **SUPPORTED**
- active-roster room-state V1 = **NOT SUPPORTED / CLOSED**
- WR1-only state candidate = **NOT JUSTIFIED / CLOSED**
- targetable dropback team-level signal = still valid
- targetable dropback full-stack = **FAILED CLOSED**
- current priority = **2024-2025 unchanged room-targets-per-play confirmation**
- run `36173485673` = **IN PROGRESS**
- no player-level integration has been authorized yet

Continue from that exact state.
