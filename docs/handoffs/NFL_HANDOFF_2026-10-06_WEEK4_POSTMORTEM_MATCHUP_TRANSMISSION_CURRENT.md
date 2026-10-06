# NFL HANDOFF — 2026-10-06 — WEEK 4 POSTMORTEM / FOOTBALL MATCHUP TRANSMISSION CURRENT

Status: **CURRENT CROSS-CHAT CONTINUITY AUTHORITY**

GitHub is canonical. This handoff exists so the next chat can continue the exact current research state without re-explaining, restarting, recursively loading old handoffs, or reopening closed science.

---

## 1. User's current concern / research direction

The user's core concern is no longer merely calibration or aggregate averages.

The user explicitly challenged whether the model is doing enough **football**:

> projections should express the specific player, current role/usage, offense, opposing defense, position-specific matchup, injuries, QB quality, committee/bellcow structure, receiving hierarchy, and current in-season game environment.

The motivating example was Bijan Robinson vs New Orleans in Week 4. The user's point was not "fit Bijan after he smashed." The point was that a football-aware model should already know, pregame, when:
- a defense is weak against the run or a position;
- a defense is strong against a position;
- an offense is run/pass tilted;
- a player is a bellcow versus committee;
- a WR room is concentrated or dual-threat;
- injuries/vacancies change opportunity;
- QB quality changes receiving/pass environment;
- current-season matchup state has moved away from prior-year assumptions.

This concern is valid enough that **football matchup transmission is now the primary active root-cause lane**.

Do not answer the user by retreating to generic "improve averages" language. The exact active question is:

> Does current production fail to transmit already-available, pregame football knowledge into the final player projection strongly enough?

---

## 2. Live canonical repository state at handoff

Repo:
`dkaps6/imtiredofthis`

Production `main` before this continuity-doc update:
`d9b55721adfbf3c9885bfa8839e0fef19f6252ad`

Important:
- no Week-4 postmortem research has been promoted to production;
- canonical production behavior is still the Week-4 production stack;
- no new live OddsAPI acquisition was used in the postmortem;
- never spend another OddsAPI pull without fresh explicit user authorization.

Primary research branches:
- `research-week4-postmortem-execution-v1`
  - current head before this handoff: `f3ccd5e464fe02ba4397dc677faf07c2d3da25b8`
- `research-distribution-right-tail-asymmetry-v1`
  - current head: `80bc5a2768c9be031757e830e4d8ae8d958d5a93`
- `research-opportunity-state-conflict-uncertainty-v1`
  - current head: `eabbe74f24c7c426c8330ec1d59e8088de773c0f`
- `research-football-matchup-transmission-v1`
  - current head: `8670f39393035dac2fedbd5ec8c40e9a75a1bd50`

Issue #535 contains the detailed prospective/frozen audit trail. Latest matchup continuity comments at handoff:
- `6021335699`
- `6021798748`
- `6022435377`

---

## 3. Week-4 board authority — never reacquire just to analyze

Paid live Week-4 Full Slate:
- dispatcher run `36934550541` = SUCCESS
- paid run `36934563481`
- paid artifact `11197726111`
- digest `sha256:202aa5205e71ad0acedf1910f505c2b362f564d8593a0f82dd0928fb45175aae`

Paid run failed only at downstream QB-C2 lineage stamping after live odds/pricing existed.

Canonical recovered Week-4 final board:
- recovery run `36935917903` = SUCCESS
- artifact `11197776900`
- digest `sha256:3f058570037ca016a5cbf1fa79e6e6abc4d845384de4dbdfaf477c8f0a3160a8`
- 3,460 side rows
- 1,730 priced offers
- 873 player-market rows
- 16 games
- pricing CURRENT
- unresolved-position rows = 0
- Tyson Bagent and Marcus Mariota quarantined from final board only

Use this recovered board for Week-4 betting/postmortem analysis. Do not fetch live odds again merely to reproduce Week 4.

---

## 4. Week-4 canonical postmortem — complete

Research branch:
`research-week4-postmortem-execution-v1`

Core successful settlement/postmortem run:
- `37482840038` = SUCCESS
- artifact `11421298874`
- digest `sha256:1b323783d6d3cc7fe172c84f11b89d10bee5e92773264a8d4c615b4489124d21`

Expanded failure-mode run:
- `37485600879` = SUCCESS
- artifact `11423535484`
- digest `sha256:59185d2aa4178971f6b0cc806eddd172ca0e7fbae559ad44da37a2b99656d791`

All 16 Week-4 games final.
nflverse player stats / rosters / snap counts covered all 32 teams.

Settlement integrity:
- first run correctly failed closed on four Jadarian Price DK rows rather than treating DNP as a zero-stat UNDER win;
- exact official OUT/IR/nonparticipation evidence was attached;
- those rows settled VOID;
- final unresolved rows = 0.

Week-4 selected board:
- 440 settlement rows
- 427 decided
- 13 voids
- **220-207**
- **51.52%**
- **-3.38u**
- **-0.79% ROI**

By position:
- QB 27-24, +0.02u
- RB 78-75, -3.74u
- WR 78-78, -2.73u
- TE 36-29, +3.15u

By market:
- pass_yards 12-12, -1.36u
- rush_yards 45-36, +4.06u
- rec_yards 75-71, -4.31u
- receptions 71-70, +1.19u
- rush_rec_yards 17-18, -2.95u

Projection quality:
- model MAE 18.89
- selected line MAE 18.32
- model closer than selected line 46.84%
- model signed bias -6.34
- market-line signed bias -3.83

Calibration:
- Week-4 mean stated fair probability 66.48%
- realized win rate 51.52%
- 70-100% stated band n=151
- mean stated 79.06%
- realized 59.60%

No simple Week-4 slice is promoted.

---

## 5. Weeks 1-4 cumulative state — closed factual record

Weeks 1-3 were NOT regraded. The exact frozen Weeks-1-3 artifact was concatenated with newly graded Week 4.

- W1: 204-205, -20.92u
- W2: 192-198, -22.06u
- W3: 233-208, +2.26u
- W4: 220-207, -3.38u

Cumulative:
- **849-818**
- **50.93%**
- **-44.10u**
- **-2.65% ROI**
- model MAE 17.89
- selected line MAE 16.86
- model closer than line 46.07%

70-100% cumulative confidence band:
- n=653
- mean stated probability 80.52%
- realized 53.60%

Cumulative clustered slice analysis:
- 57 eligible slices
- **0 BH-FDR survivors**

Do not create:
- top-N rescue;
- 20+ raw-edge rule;
- UNDER-only rule;
- position carveout;
- arbitrary edge threshold.

Both downstream selector studies remain terminal NULL:
1. Market-Relative Bet Selector V1
2. Market Offer Residual Probability V1

Raw edge is not validated confidence.

---

## 6. Manual selection lesson — do not overlearn the fun tickets

Prospectively:
- Thursday PIT-CLE four-leg ticket hit.
- Monday ATL-NO three-leg ticket hit.
- larger Sunday process-backed pool went **10-12 (45.5%)**.

Therefore:
- Thursday/Monday contained legitimately good reads;
- subjective "clean football disagreement" curation is **not** yet a validated selector;
- do not use those wins to post-hoc certify human curation.

The user's real goal remains:
> identify the genuinely good football spots without taking every giant model edge.

---

## 7. Failure-Mode Atlas — critical current diagnosis

Doc:
`docs/research/WEEKS1_4_FAILURE_MODE_ATLAS_V1.md`

Canonical run:
`37485600879`

Across 1,667 decided Weeks 1-4 rows:
- model mean error: **-5.61**
- model median error: **-0.43**

Interpretation:
the overall low mean is NOT mainly a universal -5/-6 yard center shift. The median is near zero. A relatively small number of huge upside outcomes drag the mean down.

Define diagnostically:
`z = (actual - model_proj) / model_sd`

Rows with |z| > 2:
- 270 / 1,667 = **16.20%**
- actual above model: **93.33%**
- selected side UNDER: **83.70%**
- loss rate: **82.59%**
- units: **-178.96u**

Rows with |z| > 3:
- 116 / 1,667 = 6.96%
- actual above model: 98.28%
- selected UNDER: 85.34%
- loss rate: 87.07%
- units: -86.10u

The outcome-defined |z|<=2 complement was +134.86u, but this is NOT a pregame selector because membership uses realized outcomes.

Extreme +2SD vs -2SD counts:
- pass_yards: 8 vs 5
- rec_yards: **89 vs 4**
- receptions: **68 vs 0**
- rush_rec_yards: **33 vs 2**
- rush_yards: **54 vs 7**

This is a major asymmetric high-side miss problem.

Confidence bands become worse exactly where tail failure rises:
- .90-1.00 stated-probability band:
  - n=98
  - mean stated 94.19%
  - realized 56.12%
  - >2SD miss rate 42.86%
  - model closer rate 35.71%

All-row probability scoring:
- fair_prob Brier 0.28629
- book no-vig Brier 0.24996
- constant .50 Brier 0.25000
- fair_prob log loss 0.80865
- no-vig log loss 0.69309
- fair_prob AUC 0.5380

Late stack is NOT the primary aggregate source:
- all rows raw MC MAE 18.42 -> final 17.89
- W4 raw MC 19.75 -> final 18.89
- W4 pass_yards MC 70.86 -> final 61.76

Do not rip out QB M89/M90 based on W4 explosions.

Receiving yards are different:
- all-weeks MC MAE 22.73
- final 22.75

The late stack does essentially nothing to repair receiving-yard base error.

---

## 8. Rush+receiving remains a distinct center problem

Rush+receiving is not just a tail issue.

All rush+rec:
- mean error -16.44 yards
- median error -8.82

RB-only:
- mean combo error -16.60
- median -8.82

Component decomposition:
- rushing component mean error -10.95
- rushing median -3.67
- receiving component mean -4.04
- receiving median -0.12

Thus RB rush+receiving low bias is primarily a rushing-authority/volume issue.

Do not globally +8 yards. This is a structural football lane.

---

## 9. PlayerForm vs Bayes opportunity mismatch — known, but obvious fixes are CLOSED

Historical audit already established:
`BAYESIAN_CURRENT_STATE_TRANSMISSION_SYSTEMIC_MISMATCH_CONFIRMED`

For RB rush share / WR target share / TE target share:
- PlayerForm faster current-season blend beat downstream Bayes in both 2024 and 2025.

But exact obvious production-order mean replacement was then tested:
Opportunity Authority Priority V1 = **FAILED CLOSED**.

Do NOT rescue with:
- RB-only replacement;
- TE-only replacement;
- different Bayes weights;
- position exception;
- retuned group strengths/caps.

Week-4 examples merely illustrate the known mismatch:
- Bijan current .635 -> PlayerForm .616 -> rules .542
- Jonathan Taylor .795 -> .758 -> .647
- Chuba .612 -> .458 -> .407
- Aaron Jones .612 -> .530 -> .470

Those do not authorize a new mean rule.

---

## 10. Opportunity State Conflict Uncertainty V1 — NULL / CLOSED

Branch:
`research-opportunity-state-conflict-uncertainty-v1`

Canonical run:
- `37492425193` SUCCESS
- artifact `11426127514`
- digest `sha256:51ef028a66ffffa01608b6c5dee76429b80d6bdf9a718915b5a352e747984f4a`

Question:
does `abs(PlayerForm opportunity - Bayes opportunity)` identify uncertainty?

Result:
`OPPORTUNITY_STATE_CONFLICT_UNCERTAINTY_NULL`

It generally ran backward:
- RB 2024 Spearman -0.1364; Q4-Q1 Bayes AE -0.04192; CI entirely negative
- RB 2025 Spearman -0.0897; AE delta -0.02568; CI entirely negative
- WR both years near-zero/slightly negative
- TE 2025 significantly negative

Therefore:
- do NOT widen distributions because PlayerForm and Bayes disagree;
- do NOT invert/rescue the signal;
- do NOT use low disagreement as betting selector.

Exact idea is CLOSED.

---

## 11. Historical right-tail asymmetry — SIGNAL CONFIRMED

This is newer than the original Week-4 postmortem and is important.

Branch:
`research-distribution-right-tail-asymmetry-v1`

Frozen plan:
`docs/research/DISTRIBUTION_RIGHT_TAIL_ASYMMETRY_V1_PLAN.md`

Canonical run:
- `37493425352` SUCCESS
- head `80bc5a2768c9be031757e830e4d8ae8d958d5a93`
- artifact `11427113291`
- digest `sha256:5051111ebc20b79a334cfd1dc08ec8d8d52107e59acb023f7f5fefe4e5b771dd`
- scoreable rows: 40,640
- football-only
- sportsbook inputs 0
- parameters fit 0
- projection mean changed = false

Disposition:
`DISTRIBUTION_RIGHT_TAIL_ASYMMETRY_SIGNAL_CONFIRMED`

Replicated in BOTH 2024 and 2025:
- **rush_yards**
- **rec_yards**
- **receptions**
- **rush_rec_yards**

Not replicated:
- pass_yards (2024 pass, 2025 fail)

Examples q90-vs-q10 upper-minus-lower tail-break imbalance:
- rush_yards:
  - 2024 +4.02pp, CI [2.87, 5.19]
  - 2025 +2.74pp, CI [1.63, 3.87]
- rec_yards:
  - 2024 +10.40pp, CI [8.81, 12.02]
  - 2025 +8.35pp, CI [6.91, 9.83]
- receptions:
  - 2024 +11.35pp, CI [10.17, 12.56]
  - 2025 +9.88pp, CI [8.74, 10.99]
- rush_rec_yards:
  - 2024 +12.56pp, CI [10.72, 14.40]
  - 2025 +9.17pp, CI [7.51, 10.82]

This is genuine historical confirmation that the production-aligned distributions underrepresent upper outcomes in these markets.

What it authorizes:
- only a **separately frozen mean-neutral asymmetric-distribution candidate**;
- candidate must preserve means and prove better distribution/probability scoring on genuine holdout/prospective data.

What it does NOT authorize:
- automatic OVER bets;
- UNDER ban;
- skew factor fit on Week 4;
- threshold/top-N rescue;
- mean boosting;
- production change.

At handoff, no committed RESULT.md was observed on the branch yet; canonical run/artifact above are authority. Writing a frozen result doc is a reasonable secondary continuity task.

---

## 12. Opponent-defender injury source V1 — closed

Issue checkpoint `6020600654`.

Branch:
`research-rb-opponent-defender-injury-source-v1`

Canonical run:
- `37494717496` SUCCESS
- artifact `11426662979`
- digest `sha256:8c2d28a674b92d4462eb26cd4285306e711306ebdf56d82709808dc7a80b4f16`

Raw nflverse injury source had strong identity/role coverage, but frozen V1 required >=95% defensive-front game/report-status completeness.

Observed report-status completeness:
- 2023 45.07%
- 2024 43.90%
- 2025 42.50%
- 2026 34.69%

Disposition:
`SOURCE_PARITY_NOT_CLEARED`

Do not rescue by switching to practice-status after seeing result.
Do not treat blank report status as active/healthy.
Do not treat it as T-75 official inactive authority.

---

## 13. FOOTBALL MATCHUP TRANSMISSION V1 — PRIMARY ACTIVE FRONTIER

Branch:
`research-football-matchup-transmission-v1`

Frozen plan:
`docs/research/FOOTBALL_MATCHUP_TRANSMISSION_V1_PLAN.md`

Plan freeze was recorded before historical scoring.
Current branch head at handoff:
`8670f39393035dac2fedbd5ec8c40e9a75a1bd50`

Phase A canonical run:
- `37504432928` SUCCESS
- artifact `11431601141`
- digest `sha256:bc20084e4dd9a5272ff9bc6acb922ce5395b40243a8f5380b8a88d42b151bdd8`
- outcomes used = 0
- sportsbook refetch = false
- production change = false

Phase A disposition:
`DETERMINISTIC_TRANSMISSION_AUDIT_COMPLETE`

### The exact architecture finding

Production DOES collect meaningful football matchup data.

Week-4 TeamForm/source contains:
- plays / pace
- current PROE / pass tendency
- pass-rate offense/faced
- success rate
- def_rush_epa
- pass-defense EPA/YPA/success
- explosive-play allowed
- pressure
- box rates
- yards-before-contact per RB rush
- stuff rate
- WR/TE/RB YPT allowed
- outside/slot YPT allowed
- coverage man/zone/middle
- player role/usage/injury state

But generic RB/WR/TE projection transmits only a narrow subset.

Phase A feature classifications included:
- AVAILABLE_BUT_DROPPED_BEFORE_TEAM_CONTEXT = 9
- TEAM_CONTEXT_PRESENT_BUT_NOT_USED_BY_RULES = 2
- AVAILABLE_BUT_NOT_CONSUMED_GENERIC = 1
- GENERIC_SIM_FALLBACK_BYPASSED_BY_RULES_PASS_RATE = 1
- USED_TO_COMPUTE_SCRIPT_METADATA_ONLY = 1
- plus QB-specialist-only families

Specific confirmed gaps:

1. **def_rush_epa**
   - present in TeamContext
   - NOT used by generic `matchup_multipliers()` or simulation.

2. **explosive_play_rate_allowed**
   - present and current-season repaired
   - NOT used by generic matchup rules/simulation.

3. **position-specific defensive receiving efficiency**
   - `ypt_allowed_wr`
   - `ypt_allowed_te`
   - `ypt_allowed_rb`
   - outside YPT
   - slot YPT
   - exist upstream
   - are DROPPED before canonical TeamContext
   - therefore cannot affect generic player projection.

4. **YBC/stuff run-defense fields**
   - exist upstream
   - are dropped before TeamContext.

5. **offensive PROE / pass tendency**
   - exists
   - generic simulation would use PROE only if `rules_pass_rate` were absent
   - but production rule layer always populates `rules_pass_rate`.

6. **hardcoded generic pass/rush split**
   - `project_game_script()` sets `pass_share = 0.57`
   - Phase A source assertions confirm the hardcoded 0.57 path.
   - lead/neutral/trail probabilities are calculated but NOT consumed by simulation_v2 to alter the pass/rush split.
   - thus generic skill-position volume can effectively shadow actual offense pass/run tendency.

7. **RB matchup adjustment**
   - generic rushing matchup currently changes YPC mainly via coarse box thresholds:
     - light box >= .60 => x1.07
     - heavy box >= .60 => x0.94
   - it does NOT jointly express role concentration × run-defense quality × expected team rushing environment.

8. **QB exception**
   - M89/M90 is a separate richer specialist path with defensive pass context.
   - do not conflate QB with generic RB/WR/TE architecture.

### Bijan Week-4 architecture illustration — NOT a candidate

Exact pregame ATL @ NO state:
- ATL neutral pass rate: 46.5%
- ATL true PROE: -0.1050
- Bijan PlayerForm rush share: 0.61579
- Bijan Bayes/rules rush share: 0.54207
- production rules pass rate: 0.57
- projected ATL plays: 59.5449
- rules team rush attempts: 25.6043
- Bijan Bayes YPC: 4.8741
- NO light box: 0.391
- NO heavy box: 0.250
- Bijan rules rush-efficiency matchup multiplier: **1.00**

Production therefore applied **no rushing matchup efficiency boost** from NO's broader run-defense context.

A purely mechanical transmission illustration using already-existing pregame ATL neutral pass rate (46.5%) + PlayerForm rush share (0.6158), while holding Bayes YPC unchanged, yields:
- ~31.86 team rushes
- ~19.62 Bijan carries
- ~95.62 rushing yards

This is NOT an authorized formula/candidate. It merely quantifies how much pregame football state can be lost through the canonical path.

Do not say "95.6 was the correct projection" or tune toward Bijan's actual outcome.

---

## 14. Anti-retest: generic matchup boosting has already been studied

Do not respond to the architecture finding with:
"bad run defense => boost RB X%."

Prior RB matchup work:
- M95A = RB role × defensive rushing vulnerability truth test
- it found strong descriptive football truth:
  - established/workhorse RBs did better against leakage-safe weak run defenses;
  - combined workhorse weak-vs-strong descriptive gap ~+13.73 rushing yards;
  - 100+ rushing-yard rate gap ~+11.93pp.
- however generic prospective role+defense / interaction models were mixed and often worse out of year.

M95B:
- compact offense × defense matchup engine
- defensive additions produced only small/inconsistent incremental gains over role+offense
- no stable production promotion.

QB generic defensive matchup:
- M56 richer static defensive matchup = failed signal screen.
- M83 conditional defensive adaptive gameplan = `NO_DEFENSIVE_ADAPTATION_MECHANISM`.

Therefore CLOSED:
- arbitrary defense-vs-position multiplier;
- fantasy-points-allowed shortcut;
- generic weak-defense RB boost;
- reopening M95A/M95B with new weights because Bijan hit;
- retired WR coverage_penalty heuristic.

The genuinely OPEN question is **transmission/integration**, not "does matchup matter?"

---

## 15. Exact next action — this is where next chat should resume

Primary next action:
continue `research-football-matchup-transmission-v1` from completed Phase A into the already-frozen **Phase B / Phase C historical audit**.

Do NOT rerun Phase A.

Phase B — historical residual signal audit:
- leakage-safe 2024 Weeks 2-18
- leakage-safe 2025 Weeks 2-18
- zero 2026 outcomes
- zero sportsbook lines/odds
- no candidate coefficient fitting

Question:
> Conditional on the existing production player/role projection, do strict-prior defensive matchup variables explain next-game residual error in the expected football direction?

Primary markets:
1. RB rush yards
2. RB rush+receiving yards
3. WR receiving yards
4. TE receiving yards
5. RB receiving yards
6. QB pass yards control

Predeclared matchup families:
- RB rushing: def_rush_epa, stuff rate, YBC allowed, box rates
- receiving: position-specific YPT allowed, outside/slot YPT, pass EPA/YPA/success, coverage
- volume/game environment: pace/plays, PROE/pass tendency, pass/rush tendency faced, pressure

Phase C — usage × matchup interactions:
- RB rush opportunity × run-defense weakness
- WR target/route opportunity × WR/outside/slot matchup weakness
- TE target/route opportunity × TE/zone/middle weakness
- RB target/route opportunity × RB receiving matchup weakness

No target-week realized usage.
No threshold search.
No top-N search.
No bellcow-only post-hoc carveout.
No sportsbook input.

Replication gate:
a matchup family is genuinely missing football signal only if:
1. expected-sign association in BOTH 2024 and 2025;
2. clustered uncertainty supports direction;
3. incremental to existing production role/opportunity;
4. same semantics available live;
5. not a restatement of a closed family.

If a family passes, freeze a **separate full-stack integration candidate** before scoring it.
Diagnostic PASS != production promotion.

Secondary continuity tasks:
1. write/freeze a RESULT.md for Right-Tail Asymmetry V1 from canonical run `37493425352` if still absent;
2. write/freeze a Phase-A matchup result doc from `37504432928` if still absent;
3. preserve right-tail candidate work as separate from the football-matchup integration lane;
4. do not let calibration research replace the user's primary football-layer question.

---

## 16. Other frozen/open research state — do not accidentally reopen

Projection-authority move-direction forward confirmation:
- Week-4 Observation #1:
  - strengthened n=111, 48.65%, -6.94u
  - weakened n=227, 52.86%, +1.95u
- point direction consistent with discovery
- bootstrap CIs cross zero
- support only 1/8 weeks, 111/400 strengthened, 227/400 weakened
- disposition `FORWARD_OBSERVATION_ONLY_INSUFFICIENT_SUPPORT`
- no shrink-to-market rule.

GSIS RB successor:
- private V2 allocation lock valid
- allocation SHA `38a79c295a85f5f58c7a77473aae9ea62122da0c5f06d99920f868f1ef3e6b4e`
- event-audit SHA `2d01fa956319210a420ac0617e9d2a422abb7dbc101e8610974df8c5536c0461`
- required pregame three-arm projection lock was not persisted
- Week 4 therefore not scientifically scoreable
- do not reconstruct after outcomes
- next eligible event must persist three-arm projection lock before kickoff.

RB-PD2:
- no valid Week-4 prospective capture found
- do not manufacture Observation #2
- HOLD remains.

WR/CB:
- public/historical source lane remains constrained
- retired `coverage_penalty` stays retired
- do not invent player-specific WR-CB assignment without reproducible source.

ATD:
- executable football-only output exists
- dedicated science remains NOT certified
- no calibrated 2+/3+ TD count distribution
- do not treat ATD probabilities as production-certified betting probabilities.

---

## 17. User interaction / tone continuity

The user is frustrated because an enormous amount of research/backtesting has not yet produced the reliable model he expects.

Do not patronize or tell him to be patient.

He specifically wants:
- strong football reasoning;
- objective/unbiased analysis;
- continuous forward movement;
- no repeated research;
- no post-hoc rescues;
- no pretending a hit proves validation;
- no generic status-only answer when actionable work exists.

When updating him:
- explain what was discovered in football terms;
- distinguish data present vs data actually consumed;
- be explicit about what is closed and why;
- keep moving without asking him to re-explain.

The key framing:
> We did build a football model, but the Phase-A audit shows that the production architecture is not transmitting all of the football context it already possesses into the generic skill-position projection. The next task is to prove historically which missing transmissions actually add signal, then integrate only those that replicate.

---

## 18. Start rule for the next chat

Read only, in this order:

1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md` — newest top checkpoint only
3. this file:
   `docs/handoffs/NFL_HANDOFF_2026-10-06_WEEK4_POSTMORTEM_MATCHUP_TRANSMISSION_CURRENT.md`
4. `docs/research/FOOTBALL_MATCHUP_TRANSMISSION_V1_PLAN.md`
5. Issue #535 from comment `6021335699` onward, especially `6021798748` and `6022435377`
6. query live GitHub for current main, the four research branches above, PRs, and Actions before editing

Do NOT recursively load old handoffs unless a specific anti-retest question requires targeted lookup.

Then immediately continue Football Matchup Transmission V1 Phase B/C.