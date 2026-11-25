# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-11-20 — V30 — **NO PREGAME FILTER PICKS OUT WINNING MODEL BETS ON 2024-25 LINES; THE PRICE ALREADY PREDICTS HIT RATE; THE RB-UNDER RULE AND THE PASSING-YARDS EDGE BOTH FAILED OUT OF SAMPLE**

**Repo:** dkaps6/imtiredofthis. **Branch:** `claude/nice-cori-4qkwqs`.
- V29 is archived verbatim at `docs/handoffs/ARCHIVE_NFL_RESEARCH_HANDOFF_V29_PRE_V30_2026-11-20.md`.
  Its items 1-15 are kept below, still current except where V30 says otherwise.
- V28 is at `docs/handoffs/ARCHIVE_NFL_RESEARCH_HANDOFF_V28_PRE_V29_2026-11-19.md`. Its §0
  mandatory reading before any card still binds.

### V30, added 2026-11-20 (read first)

**Start with `docs/research/WHAT_WE_KNOW_2026_W11.md`:** the one-page settled summary, §1-9.

16. **Team and player identity, and how FanDuel builds lines**
    (`TEAM_PLAYER_IDENTITY_2026_V1.md`).
    - Teams' efficiency, pass lean and quality repeat; volume and beating the spread do not.
    - Player shares repeat.
    - **A FanDuel line ≈ the season average shaded 5-16%,** with about 0.04 weight on the last
      3 games.
    - Player-level misses vs the line **reverse** between halves.
17. **Alt ladders** (`ALT_LADDER_GENERIC_SHAPE_EXPLORATORY_V1.md`): rung prices are about 95%
    explained by market + rung/main ratio, i.e. generic. Exploratory only.
18. **RB-under rule and role-change rule on 2024-25 lines**
    (`ROLE_CHANGE_AND_RB_UNDER_HISTORICAL_RESULTS_V1.md`): **both FAIL.**
    - RB-under: 2024 60.2%, 2025 47.7%.
    - Blind overs lose 7-12% in every market in both seasons.
    - The weekly tracker (`RB_UNDER_TRACKER.md`) continues as a record only, with **no
      demonstrated edge**. Week 11 picks are logged.
    - Scheduled check-ins: Sun 11-22 (inactives) and Tue 11-24 (grade week 11).
19. **Passing-yards edge, weeks 9-10:** 49.1%, -7.2% (n=112), after the 2026-11-04 retraction.
    Dead.
20. **Selective pick identification** (`SELECTIVE_PICK_IDENTIFICATION_RESULTS_V1.md`),
    preregistered, fit on one season and scored on the other: **all six filters FAIL.**
    - Filters tried: model unders, big edges, winning cells, price, the model's record on the
      player, and a logistic classifier.
    - The classifier hit 58.5% and lost 1.4%: it picked -150 bets.
    - **Hit rate tracks the price-implied rate in every band.**
    - The model's direction is real but small: an under the model calls OVER loses 7.9%, vs
      -2.3% when it agrees.
    - **Data defect:** the full-stack/base benchmark `game_id` is the prior season's game.
      Team, opponent, line and actual are correct. Do not cluster or join on it.
21. **What is left that the model cannot do by itself**, not yet tested:
    - **line shopping:** the better of the FanDuel and DraftKings numbers on the same player;
      the 2026 boards carry both books;
    - **timing:** news the line has not absorbed yet.
    - Each needs its own preregistration. Check the prior work first:
      `VEGAS_CONFIRMATION_LIKELIHOOD_V1_PLAN.md` mentions cross-book prices.

### V30 additions, 2026-11-22 to 11-24 (week 11 closed)

22. **Week 11 is fully graded** (`WK11_2026_BACKTEST_V1.md`).
    - **Sunday card v2:** SAFE +158 won 3/3. MODERATE, MEDIUM, HIGH and LOTTERY lost.
    - **The three legs the fixed G7 would have removed all missed:** Purdy 10+ rush (0),
      Hubbard 10+ rec (0), Warren 10+ rec (1).
    - **The operator's MNF ticket won 6/6** (about +511). The card's entry lost.
    - **Screened rushing unders went 1-3** in their first forward week.
    - **Line movement:** the model's Thursday picks at Thursday prices made +0.4%; at Sunday
      prices -1.2%. Thursday lines that later moved toward the model hit 58.9% (n=158).
    - **Only week 11 can be checked.** Week 10's Thursday board was TNF-only.
23. **Operator rules added** (`YOUR_RULES_PLAIN_ENGLISH.md`):
    - **23:** no leg type on a ticket without a positive graded record (≥10 bets). The record
      is `data/market_track_record/graded/w9_w11_2026_leg_type_record_v1.csv`; update it every
      week.
    - **24:** think about each game like a football person first; the model checks and sizes
      the read.
      - **It is NOT a rule or checklist.** Every game is different.
      - The operator corrected an attempt to turn it into four fixed questions.
      - Do not template it.
24. **G7 matchup fix** (commit 246d8f6). Legs are judged against the defense versus the
    player's own position group (QB / RB / WRTE). Previously QB rushing used RB run defense.
    Regression tests are in `tests/test_build_alt_cards_gates_v1.py`.
25. **SHIPPED 2026-11-24 into the model** (`docs/production/RB_MODEL_FIXES_PROMOTION_2026_11_24.md`,
    commits c88c49f and 716ddae on `claude/nice-cori-4qkwqs`; every 2026 paid Full Slate run has
    been dispatched from this branch):
    - **`RB_CARRY_SHARE_RECENCY_V2`** (PlayerForm, `scripts/modeling/rb_carry_recency_v2.py`):
      RB/FB with 3+ games feeds Bayes 0.15 x season-to-date + 0.85 x last-3 snap games, only
      when all three window games are state-labelled `HEALTHY_FULL` (anomaly gate). Otherwise
      season-to-date is kept.
      - **V1 was replaced the same day.** The PR #716 review found:
        - V1's test changed every position;
        - production's window skipped zero-touch games;
        - there was no anomaly gate.
      - Preregistered retest (`RB_CARRY_SHARE_RECENCY_END_TO_END_RESULTS_V2.md`) PASSED:
        - 2025: +0.046 [+0.022, +0.070];
        - 2026: +0.040 [+0.020, +0.060];
        - lead backs better in both.
      - The gate withholds 12-17% of RB rows.
      - Production and the test share one module.
    - **`RB_RECEPTIONS_OPPONENT_ADJ_V1`** (pricing): RB receptions mean x (1 + 0.1932 x D),
      with a new context step in Full Slate.
    - **Validation:**
      - 17 tests;
      - repo and readiness audits;
      - the real pricing loop on week-11 inputs (35 RB players adjusted, nothing else touched);
      - week-12 no-odds dry run (V1) passed PlayerForm and the context step; a local real-data
        week-12 run of V2 applied 72 RBs, withheld 33 (Barkley, Montgomery, Hall, Emmett
        Johnson among them) and left 41 with fewer than 3 games.
    - **Both are small.** The week-11 forward checks were flat.
    - **NOT fixed:** the Emmett Johnson class of miss.
      - Cause: Bayes gives zero-history backups the group-mean share. Before week 11, Carter
        (0 games) had 0.32 and Ott (1 game) 0.26, above Johnson's 0.23.
      - The fix is the allocative depth-chart prior (D1), in its frozen prospective test.
      - **Next decision:** D1 at the week-14 interim.
    - **Replay limitation:** a Full Slate replay of a completed week fails closed at
      "Ourlads role universe is empty". Logged for infrastructure.
    - **Merged to main 2026-11-24** (PR #716, merge commit 24c8bc2), at the operator's request.
      - Main now carries recency V2, the RB receptions adjustment and every earlier branch fix.
      - The PR's `replay` check stays red: its pinned week-5 artifact 11584521127 is gone
        (404). This is pre-existing and the check is not disabled.
      - Week-12 V2 dry run (45467603678): 68 applied, 31 withheld by the gate, 40 with fewer
        than 3 games. It stopped at the readiness audit only because the week-12 injury report
        publishes Wednesday.
26. **Week 12 (Thanksgiving week: Wed GB@LA, three Thu games, Fri DEN@PIT, Sun, Mon):**
    - **Paid pull 45685322803** (2026-11-25 19:27Z, operator's go): full week, alt ladders,
      event window 2026-11-25..12-01, 150 credits.
      - It failed closed at the live-prop identity gate on Chris Brooks (GB RB4, ACT) and
        Michael Pittman Jr. (PIT, RES/reserve).
      - Fixed with a verified roster row (Brooks) and a quarantine row (Pittman; revisit if PIT
        activates him before Friday).
    - **New: `reuse_paid_odds_regate`** re-runs the live-odds gate on a failed paid run's saved
      capture with **no OddsAPI request**. Use it instead of re-buying whenever a paid run dies
      at the gate.
      - Re-gate run **45687281991**: 1,628 priced side-rows, 14 of 16 games.
      - NE@LAC and WAS@ARI had no props posted yet. BAL@HOU had only 8 rows.
    - **Archived to main:**
      - paid capture: e7c6a1d (`raw_props`, `game_odds`, `alt_props/2026_wk12*.csv`);
      - board: 8a17975 (`boards/2026_wk12.csv`).
    - **Credits: 214 left.**
    - **Caveat:** at capture, 22 of 32 teams showed "No injuries reported" on NFL.com, mostly
      Sunday teams that had not filed Wednesday reports yet. Sunday games need a Sunday-morning
      re-pull before any Sunday card.

### V29 items (2026-11-19)

### Reverse-engineering study (`docs/research/REVERSE_ENGINEER_2026_W1_10_V1.md`)
Script `scripts/research/reverse_engineer_season_v1.py`; rows `data/research/reverse_engineer/`.
Boards weeks 1, 2, 6-10 (3-5 lost).
1. **PRISM's picked side:** no market/side positive in both weeks 6-10 and the out-of-sample
   weeks 1-2. Consistent only in losing on RB/WR OVER calls (RB rush OVER -14.9% / -21.6%).
   Week 6-10 "winners" (QB rush UNDER +20%, model/line 0.8-0.9 +4.5%) failed out of sample.
2. **Game plans:** rush attempts track the final margin (r 0.54; won by 15+ = 30.8 rushes,
   lost by 15+ = 20.2) far more than the spread (r 0.14; fav 7+ 27.1 vs dog 7+ 24.2).
   Offense neutral PROE is stable (odd/even weeks r 0.54); **what a defense allows is not
   (r 0.00)**, so team-vs-defense-type game plans are not measurable.
3. **Production game script:** constant pass share 0.57 for all teams. Team average + spread
   + total cuts team rush-attempt MAE 6.02 -> 5.51 walk-forward, better in 7/7 weeks;
   dropbacks not improvable. A production candidate, needs its own promotion test.
4. **Market pattern (every FanDuel offer, model ignored):** RB rushing UNDER when the team is
   an underdog by 3+: **61.4% hit, +15.5%, 6 of 7 weeks above breakeven, DraftKings
   61.3%**. Median actual 30 vs line 39.9. After removing partial/role-cut games: 59.3%,
   +11.6%, game-clustered CI [-6%, +28%]. **Lines under 60: 63.7%, +19.9% (n=91); 60+ lines
   (stars): 48%, no edge.** Alt RB rush OVER rungs on underdogs weeks 9-10: -30.4%.
5. **Not a ticket rule.** Presented to the operator as a proposal on 2026-11-19; only he can
   add it to the CURRENT RULES block.
6. **Part 2, player production (same doc §5-9):** role/volume repeats (snap 0.82-0.93,
   target share 0.80-0.89, carry share 0.90), efficiency does not (YPT 0.25-0.31, catch rate
   ~0, YPC 0.21). Best weight on the last 3 games over 2024-2026: **RB carries 0.70 (stable
   0.75/0.65/0.50)**, receivers ~0.2. **RB rushing UNDER with 2+ flags** (underdog 3+, YPC
   < 95% league, volume projection > 1.2x line; `rb_rush_situation_flags_v1.py`): 83 bets,
   66.3%, +24.6%, 6/7 weeks; out-of-sample weeks 1-2 62.5%; ex-injury-exits 60.6%, CI
   [-8.9%, +35.7%]; lines < 30 do not work (51%). Week 11 candidates (line >= 30): Taylor,
   Skattebo, Jeanty, Aaron Jones.
7. **Validity audit (same doc §10).** The part-1 team-volume claim was **invalid** (fitted vs
   unfit comparator, the flaw PR #558's review caught). Re-tested
   (`team_volume_market_increment_v2.py`): spread/total add nothing (confirms #558); the
   team's own fitted tendency beats the constant 0.57 split by 9-13% on rush attempts. A
   name-join defect (snap counts to stats) was fixed with pfr→gsis IDs; conclusions held.
8. **RB carry-share recency, preregistered and PASSED** (`RB_CARRY_SHARE_RECENCY_RESULTS_V1.md`):
   PlayerForm has no recent-games component; adding last-3 at a=0.85 (fit 2023-24) improves
   next-game carry share on 2025 (+0.0043 [+0.0016,+0.0070]) and 2026 (+0.0055
   [+0.0022,+0.0088]). Candidate only. **Caveat:** that test scored PlayerForm's blended share, which
   the simulator does not consume (Bayes rebuilds from the prior/current split).
9. **End-to-end recency test, preregistered and PASSED**
   (`RB_CARRY_SHARE_RECENCY_END_TO_END_RESULTS_V1.md`): offline reproduction of Bayes →
   top-5 → 0.95 cap (it reproduces the known -3.5 carries/team mass deficit). Expected RB
   carries MAE gain +0.073 [+0.035,+0.112] (2025), +0.046 [+0.013,+0.078] (2026); lead backs
   improve. **Small** (1-2% of MAE).
10. **Corrections (same evening).**
    * Role-shrinkage V2 **was run**: `PRISM_ROLE_STATE_FRONTIER_RESULTS_V2.md`, 8 of 9 gates,
      K6 on its own G7 threshold.
    * The coupled fixes **were tested**: `PRISM_PLAYER_EVIDENCE_SHRINKAGE_RESULTS_V1.md`.
      Evidence-precision weighting is closed negative. The allocative prior is a strong
      candidate never given its own preregistration, and its universe/K was outcome-known.
11. **Allocative prior, pregame universe, production chain:** two preregistered runs.
    * **RB-only** (`RB_ALLOCATIVE_PRIOR_PREGAME_RESULTS_V1.md`): RB carry MAE -14 to -16%,
      **K3 fired** (team RB mass worse; carries moved to an over-allocated QB block).
    * **All blocks** (`ALLOCATIVE_PRIOR_ALL_BLOCKS_PREGAME_RESULTS_V1.md`): RB carry MAE
      3.256→2.761 (2025, +0.495 [+0.419,+0.573]) and 3.457→2.854 (2026, +0.603
      [+0.460,+0.748]); lead-back bias +3.7→+1.75; QB mass fixed (K4 clear). **K3 fired
      again, smaller** (team RB mass worse by 0.14/0.21 carries). The remaining loss is the
      0.95 cap/residual and non-playing backs. **Not passed; nothing promoted.**
    * **Operator reopened** the closed "residual/top-five normalization" family
      (2026-11-19, "Yes you should reopen it"). The closure's premise ("not missing team
      mass") was falsified by the mass-budget study.
12. **Allocative prior + team budget 1.0 (D1)**
    (`ALLOCATIVE_PRIOR_PLUS_BUDGET_NORMALIZATION_RESULTS_V1.md`): RB carries MAE -19%
    (2025, +0.623 [+0.519,+0.725]) and -21% (2026, +0.736 [+0.558,+0.908]); team RB mass
    bias +1.8→+0.2 / +2.1→+0.5; lead-back bias +3.7→+0.7; committee backs improve. The
    budget-only control harms committee backs and QBs. **K4 fired narrowly** (2025 team QB
    mass worse by 0.024, CI crosses 0; the gate had no noise tolerance). Not passed.
    **Prospective test frozen before week 11 kickoff**
    (`ALLOCATIVE_PRIOR_BUDGET_PROSPECTIVE_PREREGISTRATION_V1.md`): weeks 11-18, frozen
    tables/code hashes, NI margins 0.25 carries; interim after week 14 (no decision), final
    after week 18. QB over-projection (~1 carry/team) decomposed: it is mass on QBs who do not
    play (QB1 carries accurate under D1); the prior-weight fix is closed by
    `PRISM_PRIOR_ANCHORING_RESULTS_V1.md`, so no QB study is opened.
13. **D1 → rushing yards vs FanDuel** (`D1_RUSH_YARDS_VS_MARKET_RESULTS_V1.md`): primary null
    (+0.18 yds, CI [-1.10,+1.50], n=221). D1 fixes the level (bias +10.6→+2.5) but its median
    residual is -4.1. **When D1 is above the line, the actual went over only 38%**: a second,
    independent sighting of the "book already cut him" RB-under pattern. Never decide sides
    by mean vs line; RB rushing is right-skewed.
14. **Receiving: allocative target prior + budget (TD1)** (`TARGET_ALLOCATIVE_PRIOR_BUDGET_RESULTS_V1.md`):
    WR+TE target MAE -7/-8% (2025 +0.106 [+0.083,+0.130]; 2026 +0.129 [+0.087,+0.168]); WR1
    under-projection removed (T3 lands it at ~0). The allocative prior is the whole gain. PASS as
    written, but NI-2 (team mass) was uninformative by construction; block mass shows **WRs still
    short ~1.5-2 targets/team-game and TEs over**, likely the rules layer's TE x1.265 / RB x1.20
    target multipliers (measured on wk 8/9/11 replay inputs). Prospective test frozen before
    week 11 (`TARGET_ALLOCATIVE_PRIOR_BUDGET_PROSPECTIVE_PREREGISTRATION_V1.md`). The multiplier question
    is already settled: D2 (delete multipliers) was withdrawn in `PRISM_JOINT_D1_D2_D3_RESULTS_V1.md`
    because it worsens yards. TD1 must be validated on yards (`TD1_REC_YARDS_VS_MARKET_*`).
15. **TD1 → receiving yards/receptions vs FanDuel** (`TD1_REC_YARDS_VS_MARKET_RESULTS_V1.md`):
    primary null (rec_yards -0.19, CI [-0.76,+0.40]; n=715). TD1 removes level biases
    (receptions +0.56→+0.06 = the "half a reception low" defect; WR1 rec_yards +13.7→+1.2) but
    yardage MAE does not improve; the line has the lowest MAE in both markets. **Consistent
    across rushing and receiving: the allocative+budget fixes are opportunity/level fixes, not
    yardage-accuracy fixes, and sides must come from the distribution, not mean vs line.**

### Week 11
* Paid full-week pull run **44580190319** (Thu-MNF, alt ladders), archived by 44581177692 to
  `data/market_track_record/{boards,alt_props,raw_props,game_odds}/2026_wk11*`. First attempt
  44578946345 failed closed at the live-prop identity gate (Andrew/Drew Ogletree; 141 credits
  lost, not replayable); fixed with a verified `manual_roster_overrides.csv` row.
  **218 credits left.** Sunday props were thin at capture (NYJ 8 rows, LAC 20); a weekend
  re-pull is expected.
* TNF IND @ HOU card: `docs/research/TNF_2026_WK11_IND_HOU_CARD_V1.md`. SAFE +158, MODERATE +170,
  HIGH +610 (naive). MEDIUM not buildable and LOTTERY UNSOLVED under the two-ticket cap and the
  never-above-projection rule (all eight eligible players at their deepest rungs = +1,638).
  Every ticket is priced below its blended chance. Pattern play shown separately: Taylor
  UNDER 51.5 rush (operator's call; conflicts with PRISM 55).

---

## CARRIED FORWARD FROM V28 (2026-11-16; still binding unless V29 above says otherwise) — **WEEK 10 CLOSED AND GRADED. THE OPERATOR'S TICKET RULES ARE NOW WRITTEN DOWN; READ THEM BEFORE BUILDING ANY CARD**

**Repo:** dkaps6/imtiredofthis. **Branch:** `claude/nice-cori-4qkwqs` (not yet merged to `main`).
V27 is archived verbatim at `docs/handoffs/ARCHIVE_NFL_RESEARCH_HANDOFF_V27_PRE_V28_2026-11-16.md`;
its model-research sections are carried forward below unchanged.

### 0. Before building ANY ticket card — mandatory reading, in order
1. `docs/research/YOUR_RULES_PLAIN_ENGLISH.md` — the **CURRENT RULES** block: every operator
   rule with his words and date, plus the list of rules a session invented on 2026-11-15
   that are **not** his (one leg per game; "model probability must beat the price" as a
   hard filter; tier minimums +500/+1200/+3000; an exact-match teammate check).
2. `docs/research/MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md` and the binding override
   `docs/research/evidence/2026-11-01_week8_sunday_tickets/WEEK8_USER_BINDING_TICKET_PAYOUT_AND_CONSTRUCTION_OVERRIDE_20261101.md`.
3. `docs/research/PRISM_ALT_CUSHION_STRATEGY_2026_W9.md` (+ addenda) and
   `PRISM_ALT_FOOTBALL_INTEGRATION_2026_W9.md` — the Nov 9 integrated alt-line rule.
4. `docs/research/WK10_2026_BACKTEST_V1.md` — what week 10 says.
5. `AGENTS.md` section 10.

**Why this block exists.** On 2026-11-15 the session worked from a compressed chat summary,
rebuilt the Sunday card five times in two hours, reapplied a rule the operator removed on
Nov 9 (one leg per game), and added filters he never gave. The operator's trust is very low
and he asked for a different helper. **Never rebuild his rules from memory or a summary.**

### 1. Week 10 results (`WK10_2026_BACKTEST_V1.md`)
* TNF: nine-gate builder cards 4/4; ad hoc rebuilds 1/9.
* Sunday: every version lost except **v3 SAFE +152 (7/7)**, which was replaced 23 minutes
  later. SNF: 0 of 14 tickets across five versions.
* Biggest single cause: **Rodgers on four Sunday tickets and every SNF ticket**; he was pulled
  in a 38-18 blowout (143 yds, 82% snaps). Concentration, plus a game script (PIT secondary
  lost two starters pregame) that was visible and not used.
* Whole-board rule comparison (2,947 FanDuel over rungs): blind **-16.5%**; the only rule near
  breakeven was **nine gates + own season beats the price, +1.1% (n=38, 17 players)**. Wider
  pools all lost. One week; not proof.
* Exclusions: QB-change teams -48%, toughest-quartile matchups -43%, teammate-context -24% —
  all correctly excluded. **Mahomes (news over the official report) +57% and Keenan Allen
  (team change) +59% were wrong this week.** Official club reports should outrank headlines.

### 1b. Whole-slate grading (`PRISM_W10_COMPLETE_GRADING_V1.md`, week 10 closed incl. MNF)
* PRISM-picked side: **-11.3% vs -5.5% coin flip (-5.7 pts, CI [-12.8, +1.5])** — reverses
  weeks 6-8 (+3.5 pts). Overs -20.8%, unders -6.6% (FanDuel).
* **The week-9 cushion edge did not replicate:** every cushion band lost; 1.5-2.0x at
  -150..-400 went **-15.1%** (week 9: +13.5%). Do not treat the cushion rule as a proven edge.
* Passing yards is the only market where PRISM beat the line (MAE 67.1 vs 69.3; alt rungs +11.7%).
* Star compression: top-quartile lines PRISM bias **-17.6** vs Vegas -5.9. Missed role changes:
  Emmett Johnson (PRISM 6 vs book 73.5, actual 101).
* MNF: all five tickets lost — script inverted (LAC ran, Brissett threw 49); a QB change is a
  whole-offense context change, not passing-only.

### 2. Tooling state (all under `scripts/research/`, research only)
* `build_slate_alt_legs_v1.py` — board + alt ladders -> rungs; `team_change_state()`.
* `build_alt_cards_v1.py` — nine gates. **G7 fixed 2026-11-15**: the cut was one quantile over
  all markets mixed (failed every rush rung, never a pass rung); now per market.
* `audit_card_legs_v1.py` — per-leg checklist; `teammate_context_values()` is directional
  (competitor back today / QB losing >= 30% of targets) with a volume floor for key teammates.
* `validate_ticket_card_v1.py` — ticket rules; refuses UNRESOLVED_CONTEXT / DNP_INACTIVE;
  edge bases `model`, `history`, `cushion`.
* `build_weekly_alt_card_v1.py` — **still enforces one leg per game; do not use its ticket
  assembly** until it is rebuilt from the CURRENT RULES. Its `--out-checks` CSV is useful.
* Operator-facing export: `data/research/sun_2026_wk10_model_outputs_fanduel.csv` (every
  FanDuel rung with model, book line, price, season record, checks failed).

### 3. Next actions, in order
1. **Build one card program from the CURRENT RULES** (no per-game quota; two-ticket player cap
   enforced; same-game legs priced by the real quote; game-script and concentration checks;
   single-game slates built separately). Test it on weeks 9 and 10 before week 11.
2. **Build the week-11 card the night before**, not two hours before kickoff; Sunday morning
   only applies inactives and price changes, through the program.
3. Grade MNF (LAC@BAL) when it posts; add it to the week-10 rows.
4. Model defects still open (V27 §6): ceiling compression on stars (JSN model 77 vs book 91;
   Chase 62 vs 87.5) and overconfident distributions.

### 4. Paid odds
Sunday wk10 paid run 43730255663; Sunday alt ladders archived at
`data/market_track_record/alt_props/2026_wk10_sun.csv`. Credits were 174/500 before Sunday;
the Sunday pull's cost was not recorded — check before the next paid run.

---

## CARRIED FORWARD FROM V27 (2026-11-10; still binding unless V28 above says otherwise) — **WEEK 9 CLOSED AND FULLY GRADED: THE TABLE'S PROFITABILITY CLAIM IS WITHDRAWN, THE NINE-GATE BUILDER IS THE ONLY THING THAT WORKED, AND IT CANNOT SERVE A LOTTERY TIER**

**Model:** PRISM — Player Role & Individual Situation Model. **Repo:** dkaps6/imtiredofthis.
**Branch:** `claude/nice-cori-4qkwqs`, synced to `main`.

**V26 is superseded, not retracted.** Archived verbatim at
`docs/handoffs/ARCHIVE_NFL_RESEARCH_HANDOFF_V26_PRE_V27_2026-11-10.md`. Everything it
promoted still stands. Its §8 items 1–3 are now all **done**: the alpha rule shipped,
week 9 is fully graded, and the per-cell audit ran. V26's §4 known inconsistency is
closed. **One V26 number is corrected below: the alt table is not "bettable" anywhere.**

---

### 1. WEEK 9 IS CLOSED. THE COMPLETE SLATE MADE EVERY CONCLUSION WORSE

`docs/research/PRISM_W9_COMPLETE_GRADING_V1.md`. Durable rows, committed not uploaded:
- `data/market_track_record/graded/w9_2026_alt_board_graded_complete_v1.csv` (5,115 rungs)
- `data/market_track_record/graded/w9_2026_gated_survivors_graded_v1.csv` (76 survivors)

The alt board covered **13 of 15 games** (two never had coverage: Thursday, and the
09:30 London window which kicked before the 09:08 capture). V26 graded 11 of them.
The two late games (TB@CHI, BUF@MIN — both low-total, defensive) graded at **-28.1%**
and moved the board from **-13.4% to -15.7%**.

**This is the second time a partial-slate conclusion moved materially on completion.
Partial-slate numbers must not be written into a document as findings again.**

### 2. THE TABLE AUDIT'S PROFITABILITY CLAIM IS WITHDRAWN. ITS RANKING RULE SURVIVED

Addendum appended to `docs/research/PRISM_ALT_TABLE_SUPPORT_AUDIT_V1.md`.

| market | n | signed | ROI as published | **ROI complete** |
|---|---|---|---|---|
| `rec_yards` | 1651 | +0.0329 | +2.9% | **-0.2%** |
| `pass_yards` | 781 | +0.0298 | -0.7% | **-6.8%** |
| `rush_yards` | 818 | -0.0557 | -18.5% | -22.0% |
| `rush_rec_yards` | 379 | -0.0673 | -33.9% | -28.8% |
| `receptions` | 1486 | -0.0225 | -30.0% | -30.7% |
| **all** | 5115 | | -13.4% | **-15.7%** |

**SURVIVED, unchanged:** market ordering identical; the sign rule at
**`corr(signed error, ROI) = +0.860`**; the 1.6x depth rule at **-53.5%** (n=701) against
-55.1% cited; every §4 exclusion.

**WITHDRAWN:** the word "bettable." **At table face value no market returned a profit.**
The sign rule is a reliable ranking and exclusion device, not a profitability
certificate. `rec_yards`/`pass_yards` are the only cells worth building on, and **only
with independent football gates above them.**

**ROI was negative in every price band** (<=-300: -5.0% … >+300: **-46.5%**) and every
depth band. The board has no profitable region at face value, only differently-bad ones.

**Decile calibration is actually decent** — no decile off by more than 4 points. The
board still loses 15.7% because a 3.6-point overstatement at the bottom (predicted 5.2%
vs actual 1.6%) is fatal when sold at +300 and longer. **Small tail errors do all the
damage, exactly where long tickets are built.**

### 3. THE NINE-GATE BUILDER IS THE ONLY THING THAT WORKED — AND IT IS NOT YET AN EDGE

Paired on the same pool (FanDuel, builder price window), so these are the same
takeable offers:

| | n | hit | ROI |
|---|---|---|---|
| blind (book + price only) | 1162 | 64.4% | **-5.3%** |
| **gated (all nine gates)** | **76** | **88.2%** | **+4.4%** |

`rec_yards` +0.5% blind → **+6.2%** gated (n=53); `rush_yards` -15.8% blind → **+0.3%**
gated (n=23); `pass_yards`/`receptions` emitted zero legs.

Cluster bootstrap by game (13 games, 20k draws, **game** as the resampling unit):
gated **+4.4% [-1.8%, +10.5%]**, blind -4.9% [-16.7%, +3.7%], gap **+9.3% [-2.1%,
+23.7%]**, **P(gap>0) = 93.8%**.

**READ HONESTLY: +4.4% on n=76 over one week is NOT distinguishable from zero, and
neither is the gap at 95%.** The gates move the board ~9 points in the right direction
and 94% of draws agree on the sign. Encouraging; not an established edge; **not
promotable** until weeks 10+ confirm. The gate has also **never run on a live board** —
week 9 is a retrospective check on the week it was designed from.

**No market-exclusion change is warranted.** Both emitting markets are >= 0 gated.
Excluding `rush_yards` would have discarded a non-negative. Gate list unchanged; only
the builder's cited numbers were refreshed.

### 4. STRUCTURAL LIMIT — THE GATE CANNOT SERVE A LOTTERY TIER. THIS NEEDS A USER DECISION

**All 76 survivors priced -225 to -1800, median -590. ZERO plus-money legs.** A
three-leg card from median gated legs pays about **+60**.

`CUSHION_MIN >= 1.50` plus the main-line floor mathematically selects heavy favorites; a
plus-money rung sits above the projection by construction, which is what G3 exists to
reject. So:

- **SAFE/MODERATE can be gate-certified.** They pay roughly +60 to +250.
- **HIGH/LOTTERY cannot be certified by anything validated in this repository.** They
  come from the >+300 band (**-46.5%**, n=1018) and the >1.6x band (**-53.5%**).

A lottery ticket is not a worse version of a good bet; on this evidence it is a
different activity. **Open decision for the user: shrink the card set to what can be
certified, or keep HIGH/LOTTERY as explicitly labeled uncertified speculation with the
-46.5% attached.** Do not silently present them as deeper gated legs.

### 5. THE SHIPPED WEEK-9 TICKETS AND THE CHAT RECOMMENDATIONS, FINAL

Tickets, 1u each: SAFE 3/3 **WON** +178 · MODERATE 3/4 LOST · MEDIUM 3/4+void LOST ·
HIGH 3/5 LOST · LOTTERY 1/6 LOST. **Net -2.22u.**

**Roschon Johnson killed MODERATE, MEDIUM and HIGH** — three of the four losers — at a
**15.6% snap share**, finishing 0 receptions / 0 receiving yards. He is why G1, G4 and
the no-shared-legs rule exist; **the gate as written rejects all three of those legs.**
LOTTERY went 1/6 with five of six legs in the -46.5% band.

Chat legs, flat 1u over ten legs: **-0.46u**. **Both legs of the Monday ticket lost** —
Aaron Jones 16 yards vs >30.5, Jordan Addison 28 vs >39.5, offered together at +226 as
"~41% to cash." Neither was close. **The user declined on football grounds (low-total
defensive game) and was right — the third user fade in week 9 to beat a model
recommendation** (Roschon, the Egbuka rung, this card).

Note: **Addison's >19.5 rung was a gate survivor and won**, while the >39.5 rung
recommended in chat lost. Same player, same game. **The gate existed and was not used
for that pick.**

### 6. TWO LIVE DISTRIBUTION DEFECTS — STILL OPEN

Unchanged from V26, with one new datum: §2 here shows the **alt** probabilities are
better calibrated than `model_sd` is. `model_sd` remains the weaker object — the 0-10%
band hits 39.5% and the 70-100% band hits 49.3%, everything collapsing toward 46%, and
every `edge_pct` inherits it. High-end ceiling compression also stands: TOP-quartile
bias -12.5 (`rec_yards`) / -7.3 (`rush_yards`) where the market line ran +8.1.

### 7. IMMEDIATE NEXT ACTIONS, IN ORDER

1. **Friday 2026-11-13: run `capture_prospective_board_v2.py --week 10`** after
   designations publish, before Sunday kickoff. **The only deadline item.** For
   accumulating QUESTIONABLE weeks, not for unblocking D1. K4 (<20 designated) is the
   binding constraint, so accumulate weeks rather than expect one slate to settle it.
2. **The tier decision in §4.** User call, not a modeling one. Blocks any week-10 card.
3. **Verify the production changes on a live board.** Neither
   `OUT_DOUBTFUL_OPPORTUNITY_RETENTION = 0.0` nor the alpha rule nor the nine gates has
   run on a real slate. TNF WAS@NYG is the first genuine test.
4. **Resolve the MIN QB wk9 quarantine row** — Murray stayed Out, Wentz started.
5. Then: widen simulation dispersion to the measured residual spread per market and
   projection tercile, and model the bust rate explicitly. Re-audit every `edge_pct`
   consumer after.
6. Deferred: rebuild the clearance table per audit §5 — 0.03 weighted mean |error| on a
   true holdout with non-negative signed error per market. 2026 is now a usable third
   season for that split.

### 8. PROMOTED AT V27

**Nothing new to production.** V26's `OUT_DOUBTFUL_OPPORTUNITY_RETENTION = 0.0` and the
alpha-vacancy rule remain the only production changes in this lane. The nine-gate
builder is research tooling under `scripts/research/`, not a production path, and §3
explicitly does **not** promote it. `AGENTS.md` §3 holds throughout: `cushion` uses
`model_proj` **ordinally** against a rung, and no sportsbook variable constructs any
projection anywhere in this work.

### NON-NEGOTIABLE USER SCIENCE
**THE PLAYER SHOULD HAVE THE RULES, NOT THE POSITION HAVE THE PLAYER.** Named-individual
projections from identity, injury/state, real opportunity, teammate competition, defense,
role, script and market-specific football logic. Positional MAE/RMSE are QA only.
**ANOMALIES ARE QUESTIONS, NOT TRENDS.** Maintain promoted Bayesian/ML/State/MC/QB/RB
layers; do not scrap prior work.

**Averaged MAE is disallowed as the primary metric for any individual-player distribution
claim.** A model can have every mean right and every floor and ceiling wrong and score
identically. Use per-player quantile calibration, or — for a decision claim — side
accuracy against the line, which needs no distributional assumption.

### ⚠️ STILL BINDING — THE V16 CORRECTION
**Production does NOT price from a constant sigma or a normal CDF.** Path:
`full-slate.yml:573` → `run_pricing_with_full_roster_universe_v3.py` → v4/v5/v6 → `...v1.py`
→ `import scripts.run_pricing_v2 as pricing`; `run_pricing_v2.py:400,410` does
`p_over = np.mean(adjusted_outcomes > line)` and `model_sd = np.std(adjusted_outcomes)`.
`pricing_v2.SIGMA` is **never** referenced on the production path.
`tests/test_production_pricing_path_contract.py` pins this. **Before any claim about "what
production does", grep the workflow for the entry point it actually runs.**

### ⚠️ THE SAME ERROR HAPPENED AGAIN IN V18 — AND WAS CAUGHT
A measurement put production's top-1 within-RB share at **0.2226** (near-uniform) and
blamed duplicate `(player, market)` rows in `metrics_ready.csv` consuming the top-5 rushing
pool — Walker appears 10×, and `0.95/5 = 0.19`. **Retracted.** Production calls
`_full_roster_simulate` → `_build_full_universe`, which builds from
`player_form_consensus.csv`, *not* `metrics_ready`, and raises on duplicate
`(team, player_clean_key)`. Preserved universe: 396 rows, 28 teams, **0 duplicates**. The
bug does not exist in production.

**This is now three off-production-path claims in this project** (rate-scale guard,
constant-sigma, duplicate rows). The rule is not optional: **follow the import graph from
the workflow to what is actually called, before measuring anything.**

### RB RUSHING — THE FULL DIAGNOSIS (settled; do not re-derive)
Live 2026 rows, weeks 1/2/6/7, n=182 non-QB `rush_yards` with actuals:

| component | bias | MAE |
|---|---|---|
| `mc_proj` | **−16.15** | 25.38 |
| `ml_proj` | **+0.79** | 26.40 |
| `state_proj` | −3.68 | 26.66 |
| `model_proj` final | −8.46 | 24.39 |
| vegas | −2.43 | **22.78** |

Excess bias vs market on identical rows **−6.03 [−8.03, −4.03]** — significant and
selection-immune. Bias by within-team model rank is negative at **every** rank: rank 1
**−22.34 [−29.04, −15.64]**, rank 2 −5.21 [−10.62, +0.21], rank 3+ −5.13.

### ⚠️ ROOT CAUSE — and the FOURTH retraction, which overturned the lane's premise

**RETRACTED:** the parent diagnosis's *"team RB carry volume is right: 22.59 simulated vs
23.11 actual (bias −0.52)"*. Measured on the **exact production simulation universe**
(`week7_trace__data__football_simulation_universe.csv` — 396 rows, 28 teams, 0 duplicates,
the frame `_full_roster_simulate` passes to `canonical_simulate`) by **running the real
`simulate()`**: realized RB carries **19.77/team vs 23.15 actual, bias −3.38
[−6.36, −0.53]**. An independent analytic estimate agrees at 19.70. **Team rush mass is
LOW, not right.** The old 22.59 almost certainly came from the priced `metrics_ready`
population — the same off-path error as the other three retractions.

That false premise is why the lane chased *allocation* instead of *mass*, and why both
fixes were doomed from the start.

**The mechanism** (proposed by GPT audit 6477974816 AUDIT TARGET 2; confirmed here):
`rules_rush_share` is estimated per player **independently**, with no team-sum constraint.

| quantity | value |
|---|---|
| teams over the 0.95 cap | **28 / 28** |
| raw team rush-share sum | mean **1.2729**, max **1.6877** |
| cap multiplier applied to every modeled player | mean **0.7574**, min 0.5629 |
| RB carries pre-cap → post-cap | 26.60 → **19.77** (−6.83 = −28.03 yd/team) |

The two errors do not cancel: raw shares over-allocate by **+3.26** carries, the cap
over-corrects to **−3.38**. Yards lost to compression by rank — **rank 1 13.18, rank 2
7.88, rank 3+ 5.28** — against observed shortfalls of −22.34 / −5.21 / −5.13. A uniform
multiplicative haircut lowers **every** rank at once, which is exactly the pattern that
killed fix A (mass-preserving, so the missing mass was never inside the RB distribution)
and fix B (reweighting, so the defect is upstream of all three components).

**Do NOT just remove the cap** — pre-cap mass is over by +3.26, so that overshoots. The
defect is the missing reconciliation step. **This unifies with the Bayesian finding:**
`bayesian_v2._posterior_mean` shrinks each player toward a position prior independently,
which both **inflates total mass** and **flattens relative shape** (the ~9pp concentration
gap). One structural defect, two symptoms.

### FIX A — CLOSED NEGATIVE (K1 fired), BUT IT LOCATED THE CAUSE
γ fitted on raw trailing shares, nflverse 2013–2025, 6,296 team-weeks = **0.76**, not >1
(walk-forward sd 0.022 across twelve origins). Raw trailing share is *more* concentrated
than reality. **K1 → ship nothing.** γ=0.76 is mean reversion, which the promoted
empirical-Bayes baselines already apply.

**The real finding.** Production's RB rush-share vector, measured on the deduped universe
with `rules_rush_share`, controlled for back count:
**top-1 share 0.5173 vs football's 0.6277, gap −0.1103, 95% CI [−0.1453, −0.0741]**,
present in every back-count cell.

**Layer decomposition — the cause is the Bayesian layer:**

| stage | top-1 | target | gap |
|---|---|---|---|
| PlayerForm `rush_share_prior` | 0.6897 | 0.6658 | **+0.0239** |
| PlayerForm `rush_share` (blend) | 0.6288 | 0.6421 | **−0.0133** ← on target |
| post-Bayes | 0.5173 | 0.6277 | **−0.1103** |
| post-rules | 0.5173 | 0.6277 | −0.1103 ← untouched |

Paired on the same 25 teams: **−0.0895 [−0.1370, −0.0420]**, reduced in 21/25. Back-count
confound controlled (subset size changes for 9/25): restricted to the 16 teams where it did
**not** change → **−0.0933 [−0.1292, −0.0575]**, 15/16 teams. **`apply_bayesian_to_metrics`
over-shrinks RB rush shares by ~9pp; `apply_rules_to_metrics` passes it through
bit-identically on every team.** Week 8's archived PlayerForm is still near target
(0.6148 vs 0.6444), so this is a standing property of the layer.

γ=1.75 reproduces the target and is worth **+10.3 yd** to the top back of the −22.34 gap.
**But it is mass-preserving, so backs 2+ lose exactly that, while every rank is already
low — kill condition K4.** A transform is shipped **INERT**
(`RB_CARRY_CONCENTRATION_PROMOTED=False`, effective γ=1.0 identity, dual authority) as a
fallback only. 13 tests.

### FIX B — CLOSED NEGATIVE (K2 fired), AND IT INDICTS THE METHOD
`rush_yards`, fit wk1–2 (n=89), test wk6–7 (n=79), candidates frozen before scoring:

| candidate | weights | bias | MAE | side acc |
|---|---|---|---|---|
| C0 incumbent | 0.557/0.443/0 | −8.54 | **23.66** | 53.2% |
| C1 refit | **1.0/0/0** | −15.53 | 24.55 | 51.9% |
| C2 pure ML | 0/1/0 | **+0.26** | 24.34 | 53.2% |
| C3 equal (control) | ⅓ each | −6.42 | 23.91 | 53.2% |

**K2 fired:** the naive control C3 also passed G1–G3, so the finding is "the incumbent is
stale", not "ML is right", and nothing is promoted. C2 has the right level and still fails
G1 (CI **[−4.12, +10.31]** straddles zero at n=79); **K3** fires too (bias flips wk6 +4.1 →
wk7 −3.3). **C1 indicts the method:** the *promoted* fitter, given recent data, chose
**100% MC** and made bias worse, because squared-error NNLS prefers the biased-but-sharper
component.

**Structural, code-level finding:** raw NNLS coefficients are **[1.5165, 0, 0], sum
1.5165** — the fitter *detected the level defect and asked for a 52% scale-up*;
`_normalise_nonnegative` divides it away and returns `(1.0, 0, 0)`. The blend is a strict
convex combination with no intercept, so **the ensemble cannot correct level drift except
by reweighting.** `rush_rec_yards` the same (sum 1.3357). 6 characterization tests pin it.

**Decision effect, wording corrected per GPT audit 6477974816:** fixing the level **changed
26 of 79 decisions and produced zero NET gain** — the flipped sides split **13/13**, so hits
stay **42/79 for C0, C2 and C3 alike** (verified row by row; a real coin-flip coincidence,
not a bug). C2 picks OVER on 58/79 while actuals went over on 37/79 — an unbiased mean is
not a calibrated side. **"Changes no decision" was false and is withdrawn.**

**Also withdrawn:** the stronger inference that every RB level fix must therefore be judged
primarily on side accuracy. That inverts the project's objective. PRISM optimizes
named-player projection quality (AGENTS.md first principle); side accuracy is a *downstream*
decision metric, reported but not definitional. At n=79 with a 13/13 split this is also an
underpowered test — it shows no evidence of betting gain from that substitution, not that a
more correct player projection is valueless.

### MC DISPERSION CALIBRATION — STILL SHIPPED BUT **INERT**
`k` remains `CANDIDATE_PENDING_PARITY_REPLAY`, NOT promoted. W7 exact-path parity failed
closed because the W7 fixture lacks the ML/State prediction columns. **This is now
unblocked going forward:** the `eb109a6a` archival fix is working and wk08 replay inputs
are landing (`data/market_track_record/replay_inputs/2026_wk08/`, including
`model_ml_diagnostics`, `model_state_diagnostics`, `player_form_consensus`).

**The wk08 manifest reports `"complete": false`, and that is CORRECT — do not "fix" it.**
`required_missing` is `['data/metrics_ready.csv', 'data/model_rule_simulation_inputs.csv']`.
Both are downstream of sportsbook props, and the wk08 runs were **no-live-odds dry runs**
(no `boards/2026_wk08.csv`, no `raw_props/2026_wk08.csv`, no `game_odds/2026_wk08.csv`).
Full Slate uploads `data/**` and `outputs/**`, so those files would have been captured had
they existed; they do not exist because there were no props to price. The preservation
script is failing closed and recording the reason, which is the intended behaviour.
**A replay input set therefore cannot be complete on a no-odds run by construction — the
set completes on the next LIVE-ODDS run.** That is the event to wait for before attempting
the parity replay.

Promotion still requires baseline parity established **before** any treatment result is
read, and `k` must not be retuned on W7.

### SHIPPED TO PRODUCTION (cumulative, this lane)
`PERCENTY_COLS` rate-scale fix (`9aa6058c`) — the only behaviour-changing merge. MC
dispersion transform (`b34f85aa`) — wired, **inert**. RB carry concentration — wired,
**inert**. Replay input preservation (`eb109a6a`) — working.

### STILL TRUE FROM PRIOR CHECKPOINTS — DO NOT REDO
Three opponent-side nulls. Walk-forward per-tail quantiles in all three markets: floor
slope 1.7–2.4× slower than ceiling; centrality→floor `r=+0.620`; per-player volatility
weakly persistent (`r≈0.25`). Live 2026 backtest: production probabilities severely
overconfident (Brier 0.2800 vs market 0.2491). Weeks 3–5 2026 boards are **permanently
lost** to a 7-day retention setting (AGENTS.md §5a).

### MUST-PASS ACCEPTANCE FOR ANY MERGE IN THIS LANE
1. **Preregister and freeze before scoring** — form, eligibility, held-back data, decision
   rule, kill conditions. Commit the preregistration before any result exists.
2. Include a **naive control** in every candidate set. K2 exists because C3 caught a result
   that would otherwise have read as a win.
3. **Averaged MAE is never the primary read.** For a decision claim, show side accuracy.
4. Anomaly gate per AGENTS.md: `DNP_INACTIVE` / `PARTIAL_GAME` / `UNRESOLVED_CONTEXT`
   state-labelled before any observation becomes evidence.
5. Sportsbook separation: no market variable at any stage upstream. Betting remains
   **CLOSED**; no paid OddsAPI fetch without express user authorization.
6. GPT independent audit, post ruling to #535. **The user must expressly authorize any
   merge.**

### REMAINING AUTHORIZED RESEARCH QUEUE
**Active, in this order (agreed through #535 comments 6478960995 and 6479098849):**
**(1) roster/availability EXISTENCE integrity** — Ourlads is the sole existence authority
(458 → 458 → 458 → 396 → 396; the availability layer never adds a row), so build a union
existence universe plus a game-day refresh. **(1a) a deterministic identity defect inside (1):**
`canonicalize_player_name_safe` **does not strip name suffixes**, so nflverse
`Mike Washington Jr.` (`mikewashingtonjr`) never joins the universe's `Mike Washington`
(`mikewashington`). 29 of 489 players with 2026 wk1–7 carries/targets (5.9%) keep a suffix
in their clean key; 4 realised mismatches at W7. **This silently breaks grading joins too**,
so treat it as a data-integrity fix, not cosmetics.
**(2) team rush-budget reconciliation**, which must not run before (1).
**(3) WR/TE existence-ceiling diagnostic — DONE**, see
`docs/research/PRISM_EXISTENCE_CEILING_IS_SYSTEM_WIDE_V1.md`: the ceiling is **system-wide**
(absent share 4.16% of targets vs 7.41% of carries) but **materiality is RB-concentrated** —
the worst rushing absence was a 15-carry lead back, the worst receiving absence 5 targets.
14/17 absent receivers were `ACT` in an earlier week, so the dominant class is deterministic
provider omission, not roster uncertainty. **Do not retune target shares from it.**
**(4) Bayesian evidence-dependent shrinkage**, which the auditor rules should be evaluated
only once the roster/budget contract is coherent.

**Standing capability gap — NOT a parallel lane, do not open it in competition with the
above:** we use **no NextGen Stats at all**. One reference repo-wide, Migration 75,
explicitly `production_actionable = False`; zero NGS on the production path.
`load_nextgen_stats` **works for 2026 weeks 1–7** (rushing/passing 253 rows each, receiving
617) and carries `expected_rush_yards`, `rush_yards_over_expected`, `efficiency`,
**`percent_attempts_gte_eight_defenders`**, `avg_separation`, `avg_yac_above_expectation`.
`PRISM_STRANDED_SIGNAL_DISPERSION_PREREGISTRATION_V1.md` wrongly recorded the 2026 NGS feed
as unavailable — it conflated NGS with **participation**, which genuinely does fail for
2026; that doc now carries a correction banner. **Target-week NGS rows are OUTCOMES**, so
any use must be strictly-prior-weeks-only and walk-forward. Open questions put to the
auditor (#535 comment 6479259425): is NGS publication timing pregame-safe for a Thursday
run, and does M75's QB-efficiency null already close RB rushing efficiency and per-player
box counts? Separately, `nfl_data_py` is a **fallback only** (5 files, every reference an
`except` branch) behind `nflreadpy` (55 files), and is what pins `pandas<2.0`.

Queued, not approved:
whether the same over-shrink affects WR/TE target shares; the `_normalise_nonnegative`
scale-discard as a deliberate ensemble change; preserving decision-gate evidence so weeks
6+ become bet-gradable; completing the wk08 replay input set so parity becomes possible;
shrinking the overstated matchup multipliers (contraindicated for TE, excess bias vs vegas
−0.14); alignment (LWR/RWR/SWR) restoration; Inside-five TD; TE-R5P. **No paid OddsAPI, no
sportsbook upstream, no W7 outcome fitting, no weakening of a truth/integrity gate.**

### COMMUNICATION / DURABILITY
Claude communication: GitHub Issue #535. Do not confuse executor local pytest counts with
independently CI-verified tests. Do not claim the user approved a merge. Update this ONE
root TOP after material milestones and archive displaced checkpoints under `docs/handoffs/`.
Per AGENTS.md §5a, results worth citing go in git, never only in an expiring artifact.

**Session context the next chat should not have to rediscover:** the user has corrected the
executor repeatedly and been right each time; the most consequential correction was that
MAE is the wrong metric for an individual-player claim. Two and a half days have produced
**one** behaviour-changing shipped fix, three retracted off-path claims caught before they
shipped, and a verified map of where leverage is **not** — including the corrected finding that the tested RB level substitution **changed 26 of 79 decisions but produced zero net side-accuracy gain (13/13 on flipped decisions)**. The user is time-pressured
with Week 8 imminent and does not want another lane change. The active lead is now **roster/existence integrity first**, then **coverage-aware team rush-budget reconciliation**, then **Bayesian evidence-dependent shrinkage**. Do not skip that order.


### V20 NEXT-CHAT RESTART PROTOCOL — DO THIS, DO NOT RECONSTRUCT FROM MEMORY
1. Read `AGENTS.md`.
2. Read this V20 root handoff completely.
3. Fetch Issue #535 comments newer than **6479259425** before doing anything else.
4. Re-fetch current `main` SHA and audit any new Claude commit before accepting its claims.
5. Continue **AUDIT TARGET 6**: establish whether the candidate existence/roster source is truly pregame-safe, and separately measure WR/TE existence loss. Do not design/promote a reconciliation until roster authority is leakage-safe.
6. Preserve the order: **roster/existence integrity → coverage-aware rush-budget reconciliation → Bayesian evidence-dependent shrinkage**.
7. Do not reopen betting, fetch paid odds, retune W7, promote MC dispersion, remove the 0.95 cap, or normalize incomplete rosters to 1.0 without explicit user authorization and preregistered evidence.
8. Every material result goes into git and #535; update this ONE root TOP and archive the displaced version.

### USER COMMUNICATION CONTRACT
The user wants plain-English translation of Claude's technical updates and an independent answer to: **Did this actually improve PRISM's named-player projections, or did we only learn something?** Always distinguish:
- **research/discovery** — what we learned;
- **production change** — what code actually changed live behavior;
- **validated improvement** — what survived honest production-path evidence.
Do not call a discovery an improvement. Do not agree blindly. Do not optimize positional MAE. Keep the named player in his actual football situation as the scientific unit.
