# NFL Data Inventory and New-Information Frontier

**Status:** canonical documentation audit; no predictive experiment, no production-science change, no purchase, no film charting, no CV implementation.

**Audit base:** `91afb3a51cda316467cd8431fa5243a28195c1b0` (main checkpoint verified before this audit).

**Purpose:** answer one question before any new NFL data project is funded or built:

> Do we already have this information, can we derive it, have we already tested it, or is it genuinely new enough to justify acquiring/building?

This document is deliberately isolated from active scientific work. It does not modify or supersede any frozen WR/RB/QB/TE plan, production authority, Issue #535 direction, or sportsbook boundary.

---

## 1. Executive answer

The repository is **not data-poor**. It already has a broad conventional NFL data stack: schedules, roster/identity, weekly player stats, rich play-by-play, injuries, weather/stadium context, depth roles, team tendencies, historical participation/coverage shells, NGS receiving aggregates, PFR defender aggregates, FTN tactical/decision/error fields, current WR-CB context, and position-specific opportunity/entitlement models.

The important frontier is therefore **not “more stats.”** It is **richer relational grain** that describes *who was responsible for whom, where everybody was through the play, and what happened at the interaction point*.

The highest-value genuinely new families are:

1. **Route-level receiver/defender geometry + true responsibility** — route, alignment, release, primary/secondary coverage matchup, separation by route phase, leverage, help/bracket, and open-not-targeted state. This is materially richer than current NGS average separation/cushion or scraped `primary_cb`/`shadow_flag` context.
2. **True blocker-rusher assignments + engagement timing/geometry** — exact OL/TE/RB blocker-to-rusher responsibility, pressure timing, and win/loss mechanics. M85 already proved this is conceptually useful and source-blocked rather than nonexistent.
3. **RB contact + run geometry** — first-contact location/time, yards before/after contact, missed/broken tackles, free-defender/contact count, pursuit/closing geometry, and ideally point-of-attack/run lane. This family is not present in the repo under the audited aliases and is distinct from current RB calibration research.
4. **Run-blocking responsibility / point-of-attack structure** — offensive blocker assignments and defensive fit relationships on designed runs. Harder than simple contact labels, but cross-position useful for RB, OL and defensive-front context.

The public NFL Big Data Bowl is extremely important, but **not because it gives us a production history**. It gives bounded competition slices containing precisely the kinds of labels needed to develop and validate a proprietary tracking/charting system: frame-by-frame x/y/speed/acceleration/orientation, routes, motion, blocker-rusher IDs, coverage responsibility, missed tackles, and player/ball trajectories. It should be treated as **training / validation / ground truth**, not a league-wide live source.

### Bottom line

- **Do not build a PFF clone of ordinary aggregates.** Most of that information already exists here, has already been tested, or is derivable.
- **Do not rebuild NGS average separation/cushion/YACOE.** The repo already has and tested those families.
- **Do not infer exact coverage or blocking responsibility from nearest-player geometry and call it truth.** The repo's own source audits explicitly reject that shortcut.
- If proprietary data is pursued, build it at the **route/frame/assignment/contact** layer where the repo is truly missing information.
- Economically, the best long-run strategy is likely **HYBRID**: use licensed data if a sufficiently complete multi-season/live contract is affordable; otherwise use Big Data Bowl labels as ground truth, automate geometry extraction, and reserve human review for identity and semantic responsibility.

---

## 2. Status vocabulary

| Status | Meaning |
|---|---|
| `HAVE_RAW` | Raw source already enters the repo at the needed or nearly-needed grain. |
| `DERIVABLE` | Required feature can be constructed from data already present without acquiring a new source. |
| `HAVE_AGGREGATE_ONLY` | Repo has a coarser summary, but not the richer relationship/frame/assignment information. |
| `TESTED_PROMOTED` | Family has been scientifically tested and is part of an authorized production/research authority. |
| `TESTED_CLOSED` | Family was tested and should not be re-run absent materially new information. |
| `SOURCE_BLOCKED` | Concept is scientifically legitimate but the needed historical/live source contract is absent. |
| `GENUINELY_MISSING` | No equivalent raw/derivable family was found in the audited repo; acquiring/building could add new information. |
| `LOW_VALUE_DUPLICATE` | Building a new version would mostly duplicate existing information or a scientifically closed family. |

These labels apply to **information families**, not every individual column in nflverse or every historical research branch.

---

## 3. Canonical inventory — what the repo already has

### 3.1 Core game, player and environment data

| Family | Status | Source / grain | Historical / live contract | Exact repo lineage | Temporal notes | Recommendation |
|---|---|---|---|---|---|---|
| NFL schedule, opponent, kickoff, venue/game state | `HAVE_RAW` | nflverse/nflreadpy; game/team-week | Multi-season historical; current schedule usable in-season | `scripts/utils/build_team_week_map_v2.py`, `scripts/make_team_form.py`, `scripts/backtest/historical_inputs.py` | Pregame-safe when schedule snapshot is used | Use existing |
| Weekly roster + player identity | `HAVE_RAW` | nflreadpy weekly rosters; player-week | Multi-season weekly history | `scripts/player_identity_roster_history_v1.py`, `scripts/backtest/historical_inputs.py` | Use target-week/prior snapshots only | Use existing |
| Historical/current depth roles | `HAVE_RAW` / `TESTED_PROMOTED` | nflreadpy depth charts + Ourlads; player-week/current | Historical weekly source where available; Ourlads current | `scripts/backtest/historical_inputs.py`, `scripts/providers/ourlads_depth.py` | Current scraped roles are not a full historical assignment database | Use existing; do not rebuild generic depth rank |
| Weekly player box/advanced stats | `HAVE_RAW` | nflreadpy `load_player_stats(..., summary_level="week")`; player-week | Multi-season; in-season provider dependent | `scripts/player_stats_loader_v2.py`, historical player-log builders | Strict-prior rolling features required | Use existing |
| Rich play-by-play game situation | `HAVE_RAW` | nflverse PBP; play | Multi-season | `scripts/utils/pbp.py` plus many backtests | Target-game outcome/PBP forbidden for pregame feature construction | Use existing |
| Down/distance/clock/score/game state | `HAVE_RAW` | nflverse PBP; play | Multi-season | `scripts/utils/pbp.py` | Historical summaries must be lagged | Use existing |
| Passing outcome / attempts / completions / INT | `HAVE_RAW` | nflverse PBP; play | Multi-season | e.g. `scripts/backtest/audit_qb_extreme_ypa_mechanisms.py` | Same-game realized values are post-kickoff | Use existing |
| Air yards + yards after catch | `HAVE_RAW` | nflverse PBP; play | Multi-season | `audit_qb_extreme_ypa_mechanisms.py` | Historical only for pregame tendencies | Use existing |
| EPA + CPOE | `HAVE_RAW` | nflverse PBP; play | Multi-season | same | Historical only | Use existing |
| Pressure / QB hit / sacks | `HAVE_RAW` / `HAVE_AGGREGATE_ONLY` | nflverse PBP plus FTN/PFR; play/player-week | Multi-season aggregate context | M70/M75/M81 families | Not true blocker-rusher responsibility | Use existing aggregates; acquire richer assignment only |
| Shotgun / no-huddle / play action | `HAVE_RAW` | nflverse PBP; play | Multi-season | `audit_qb_extreme_ypa_mechanisms.py` | Historical tendencies only pregame | Do not rebuild |
| Injuries / practice / body part / designation | `HAVE_RAW` | nflverse injuries; player-week | Multi-season weekly | `scripts/backtest/build_historical_injuries.py` | Builder explicitly preserves target-week report and avoids silent prior-week substitution | Use existing |
| Official/current availability/inactives | `HAVE_RAW` / `TESTED_CLOSED` for incremental QB signal | NFL/ESPN/nflverse/current production builders | Current/in-season plus audited historical paths | M78/M82 lineage; production availability seams | Valuable operationally; prior QB research did not justify treating it as new model science | Keep operational source; do not relabel as novel feature family |
| Stadium/roof/geography | `HAVE_RAW` | authoritative stadium map | Current/static | `scripts/utils/stadium_locations.py`, `scripts/build/build_weather_week.py` | Pregame-safe | Use existing |
| Weather: temperature/wind/precip | `HAVE_RAW` | NWS hourly forecast; game | Current pre-kickoff | `scripts/build/build_weather_week.py` | Forecast is available pre-kickoff; historical replay requires historically appropriate source/snapshot if used scientifically | Use existing |
| Sportsbook props/game odds | `HAVE_RAW` but **downstream boundary** | The Odds API; offer/game | Live when explicitly enabled | production pricing/odds scripts | Must not leak upstream into football generation unless a separately authorized design says otherwise | Keep downstream; not a football-data build target |

### 3.2 Team context, defense and participation

| Family | Status | Source / grain | What exists | Research disposition | Recommendation |
|---|---|---|---|---|---|
| Offensive tendencies / pace | `HAVE_RAW` | Sharp; team-season/current | offensive tendencies, neutral pace and related tables | Multiple M-series uses; same-data repackaging heavily explored | Use existing |
| Defensive tendencies | `HAVE_RAW` | Sharp; team-season/current | defensive tendencies | Static/richer defense families extensively tested | Do not build duplicate team aggregates |
| Coverage scheme team rates | `HAVE_RAW` / `DERIVABLE` | Sharp + nflverse participation; team-week | man/zone and shell context | Coverage-v2 team rates near-neutral in historical feature ablation; M75/M82 also close aggregate matchup lane | Use for context only; do not treat as new science |
| OL/DL aggregate tables | `HAVE_RAW` / `HAVE_AGGREGATE_ONLY` | Sharp/public advanced tables; team/player aggregate | line performance summaries | M85 says these are not true assignments | Keep; do not confuse with blocker-rusher data |
| Defenders in box | `HAVE_RAW` | nflverse participation; play | `defenders_in_box` | Used to derive avg/light/heavy box rates | Use existing |
| Man/zone type | `HAVE_RAW` | nflverse participation; play | `defense_man_zone_type` | Existing historical defense enrichment | Use existing |
| Coverage shell/type | `HAVE_RAW` | nflverse participation; play | `defense_coverage_type` | Derives Cover 0/1/2/3/4/6/9/2-man rates | Use existing |
| Box/shell team-week summaries | `DERIVABLE` | participation; team-week | avg box, light/heavy box, man/zone, shell rates | M83/M75 context already examined | Do not build another generic shell/box dataset |
| Defensive adaptive comparable-opponent plan | `TESTED_CLOSED` | FTN response + participation; team-week | blitz/pass-rusher response + man/zone/box | M83 `NO_DEFENSIVE_ADAPTATION_MECHANISM` | Do not reopen with new k/distance/clustering/model only; require direct new plan information |

Historical participation is useful, but repo source audits note the public participation release contract is not a reliable live/in-season replacement for proprietary tracking in recent seasons. That matters when deciding whether a feature can move from backtest to pre-kickoff production.

### 3.3 Receiver / coverage information already present

| Family | Status | Source / grain | Exact fields / examples | Research / production usage | Recommendation |
|---|---|---|---|---|---|
| NGS receiving aggregates | `HAVE_RAW` / `TESTED_CLOSED` as incremental WR/QB residual family | NFL NGS via nflreadpy; player-week | targets, receptions, avg separation, avg cushion, avg intended air yards/air distance, YAC above expectation, expected YAC, intended-air-yards share | M75; WR R9-R11; no qualifying breakthrough | Do not rebuild average separation/cushion/YACOE |
| Derived NGS concentration/top weapon | `DERIVABLE` / `TESTED_CLOSED` | player-week -> team-week | target-weighted sep/cushion/aDOT/YACOE, top1/top2 shares, top1 sep/aDOT/YACOE | M75/M82 | Do not revisit with same aggregates |
| PFR defender coverage aggregates | `HAVE_RAW` / `TESTED_CLOSED` as incremental family | defender-week | targets, comps, cmp%, yards, YPT, passer rating, aDOT, YAC, defender identity/position | M75 | Use for descriptive context; not a substitute for responsibility |
| Current WR alignment/matchup context | `HAVE_AGGREGATE_ONLY` | Rotowire/Sharp/RotoBaller-derived current player context | `slot_pct`, `wide_pct`, `man_rate`, `zone_rate`, `primary_cb`, `shadow_flag`; `wr_cb_exposure.csv` | production/context plumbing exists; historical stable responsibility contract does not | Useful current enrichment, not proprietary ground truth |
| Exact receiver-defender responsibility | `SOURCE_BLOCKED` | ideal grain: route/frame | primary/secondary defender assignment, zone responsibility/exchanges | M84 `HOLD_SOURCE_BLOCKED_NEW_INFORMATION` | **Acquire/build if pursuing new data** |
| Route-level full trajectories | `GENUINELY_MISSING` at extensible history/live scale | route/frame | x/y through release-stem-break-target/arrival | BDB slices prove feasibility; no full repo contract | **High priority build/acquire** |
| Route classification / route participation | `HAVE_AGGREGATE_ONLY` + `GENUINELY_MISSING` at full historical route grain | participation/BDB slices/current context | route participation pieces exist; BDB 2025 exposes `wasRunningRoute`, `routeRan` | no full multi-season/live route database in repo | Build only as part of richer route geometry system, not as isolated stat clone |
| Open-not-targeted state | `GENUINELY_MISSING` | route/frame semantic label | receiver separation/leverage/open window even when not targeted | not represented by NGS target-centric weekly aggregate | **High-value derived label if trajectories/responsibility are built** |
| Leverage / help / bracket / double semantics | `GENUINELY_MISSING` | frame/route | inside/outside/top leverage, help defender, bracket state, exchanges | no exact family found; generic shells are not equivalent | Build after assignment/trajectory reliability exists |
| TE-specific defender matchup artifact | `GENUINELY_MISSING` | current/route | no separate TE analogue found for WR-CB exposure | TE R3/R4 specifically call for new individual matchup information | Prefer shared route/responsibility infrastructure over TE-specific scrape |

### 3.4 FTN tactical / decision / error fields

M81 already created a frozen dictionary and tested these families. They are not frontier candidates.

| Family | Status | Fields | Disposition |
|---|---|---|---|
| Tactical call structure | `TESTED_CLOSED` | `is_motion`, `is_screen_pass`, `is_rpo` | M81: failed; worsened required metrics |
| Pressure response | `TESTED_CLOSED` | `n_blitzers`, `is_qb_out_of_pocket`, `is_throw_away`, `is_qb_fault_sack` | M81 failed |
| Throw decision quality | `TESTED_CLOSED` | `is_interception_worthy`, `read_thrown`, `is_catchable_ball` | M81 failed |
| Receiver error attribution | `TESTED_CLOSED` | `is_drop` | M81 failed |

M81 also tested strict-prior trailing-8 / trailing-3 / trend transforms and frozen interactions. Do not rescue these families by changing model class, feature recombination, or subset after the fact.

### 3.5 Opportunity / entitlement information already solved or heavily researched

| Family | Status | Authority / lineage | What it means for new-data work |
|---|---|---|---|
| Role/opportunity identity | `TESTED_PROMOTED` | M82 promoted foundation | Do not rebuild “who is WR1/RB1/TE1” as a proprietary project |
| Team receiving target pool / WR hierarchy | `TESTED_PROMOTED` | M82 foundation; WR M38/R15 | New WR data should target efficiency/translation/route relationships, not another generic target pool |
| WR individual target entitlement | `TESTED_PROMOTED` | `WR_R15_PRODUCTION_MODEL_V1` | Opportunity comparatively healthy; yard efficiency/ceiling remains weaker |
| TE target entitlement / participation | `TESTED_PROMOTED` | TE R2->R5->`TE_R5P_PRODUCTION_MODEL_V1` | Simple snap/participation rebuild is low value; matchup/route efficiency is open |
| RB rushing allocation / finite room | `TESTED_PROMOTED` foundation + research constraints | STACK/M91-96/P3 lineages | Generic depth-order/workload remaps are not frontier |
| RB Week-1 receiving tail | `TESTED_PROMOTED` | R22, using R19 assets | Tail-shape authority exists for Week 1 |
| RB Week-1 receptions/opportunity | `TESTED_PROMOTED` | R26 | Do not rebuild generic RB target/reception allocation as new data |
| RB receiving-yard historical efficiency transforms | `TESTED_CLOSED` | R23-R27D | YPR/YPT/YAC/raw target-shape/xYAC/YACOE mean-correction lane exhausted |
| RB rushing tail overlay / team-rush slicing | `TESTED_CLOSED` | STACK6/M95T stop rules | Do not create another retrospective tail overlay from same variables |
| RB prior projection difficulty -> MC width | research-qualified, **not production** | RB-PD2 yard-difficulty width | This is calibration from prior errors, not contact/run mechanics; contact geometry remains distinct new information |

---

## 4. M82 anti-retest ledger — the hard boundary

M82 is the repo's most important “do not accidentally rediscover our own work” artifact. It reconciles **27 research families** and concludes there is no same-data partial integration candidate.

### Promoted foundations

- role/opportunity identity;
- 57/43 pass-rush foundation;
- gentle pressure foundation;
- rushing allocation pool;
- receiving target pool / WR hierarchy.

### Full-stack tested / signal-screen closed families

Do not repackage these without a materially different information source:

- pass-rate / game-script repackaging;
- old market-context family;
- richer static defense;
- raw range decompression;
- attempt trust stack;
- extreme-error classifier;
- directional attempts surprise;
- possession/dropback generative state;
- model combinations;
- offensive intent/opening playcaller;
- QB volatility;
- aggregate explosive-weapon matchup;
- opportunity oracle;
- limited dropback context;
- M75 NGS/PFR personnel-tracking matchup aggregates;
- personnel discontinuity;
- inactives as incremental predictive family;
- FTN M80-M81 tactical/decision/error families.

### M82 source-frontier families

M82 originally left four “new information required” doors:

1. route × coverage-shell interaction with trustworthy route history/live contract;
2. true blocker × true rusher assignment;
3. top-weapon true responsibility matchup;
4. defensive adaptive gameplan.

M83 subsequently closed the comparable-opponent adaptive-gameplan mechanism scientifically. M84 and M85 confirmed that true receiver-defender and blocker-rusher assignments are **source blocked**, not conceptually invalid.

---

## 5. Position-specific failure-mode map

### QB

Already rich in standard play context, pressure, NGS/PFR aggregates, FTN tactical/decision fields, personnel/inactive audits and synthesis work. The legitimate missing information is chiefly **relationship-level protection and coverage structure**:

- true blocker-rusher responsibility and pressure path/timing;
- route × coverage responsibility at frame level;
- receiver availability/open windows beyond target-only aggregates.

Do not buy/build another QB box-score/pressure aggregate package expecting it to be new.

### WR

Current research says receiving opportunity is healthier than receiving-yard efficiency/translation/right-tail behavior. Aggregate NGS separation/cushion/aDOT/YACOE and persistent player explosive/YAC/air-yard traits have already failed to solve the problem. The next legitimate information family is **route-level relationship geometry**, not another average-efficiency transform.

### TE

Target entitlement/participation is solved and shipped through R5P. Team-pool-only corrections failed when they lacked individual differentiation. TE needs the same route/responsibility/matchup infrastructure as WR, with smaller sample sizes and therefore stricter anti-overfit discipline.

### RB

The repo has heavily researched rushing opportunity/allocation and conventional receiving efficiency. Current RB-PD2 work concerns player-specific projection uncertainty, not physical run mechanics. The most clearly distinct new football information is **contact/run geometry** and potentially **run-blocking responsibility / point of attack**.

---

## 6. Big Data Bowl audit — what public tracking can and cannot do for us

### Central conclusion

Big Data Bowl data prove that the desired proprietary concepts are measurable. They do **not** supply a complete multi-season + current weekly contract. Use them as:

- development/training data;
- algorithm validation;
- CV/trajectory ground truth;
- semantic-label ground truth where PFF assignments are included;
- benchmark slices for QA.

Do **not** treat a competition slice as representative league-wide historical availability or a live 2026 feed.

### Competition inventory

| Competition | Public slice / theme | Important data | Best use here | Direct production-history substitute? |
|---|---|---|---|---|
| 2020 | Running-back handoff yardage; 2017-2018 rushing examples | Player state at handoff: X/Y, speed, acceleration, distance, orientation, direction, IDs, rusher ID, formation/personnel, defenders in box, game situation | Run-snapshot geometry research / model-development reference | **No** — handoff snapshot, not full extensible tracking history |
| 2021 | Pass coverage; 2018 regular-season passing plays | Frame tracking for eligible non-linemen, player/ball coordinates, events; offensive route information | Receiver/defender tracking and route-ground-truth development | **No** — 2018 competition slice |
| 2022 | Special teams; 2018-2020 | Full tracking plus PFF special-teams scouting: gunners/vises/punt rushers, tacklers/missed tacklers, kick timing/type/direction | General tracking/CV validation; special-teams interaction labels | **No** — special-teams-only and cleaned competition data |
| 2023 | Pass protection; Weeks 1-8 of 2021 | Snap-to-release tracking; PFF responsibilities; blocker/rusher relationship concepts | **Primary blocker-rusher ground truth** | **No** — bounded 2021 slice |
| 2024 | Tackling; Weeks 1-9 of 2022 | All 22 + football tracking, event tags, PFF tackle/assist/forced fumble/`pff_missedTackle` | **Primary RB/contact/tackle geometry ground truth** | **No** — bounded 2022 slice |
| 2025 | Pre-snap tendencies; first 9 weeks of 2022 | Full-play tracking from pre- to post-snap; motion/shift, `wasRunningRoute`, `routeRan`, blocker IDs, pressure attribution, PFF coverage assignment, primary/secondary coverage matchup IDs | **Best public semantic-label slice for route/responsibility/blocking + motion** | **No** — bounded competition slice |
| 2026 | Predict movement while ball is in air; training data from 2023-2024, leaderboard evaluation against 2025 Weeks 14-18 | Before-/after-throw x/y trajectories, speed, acceleration, orientation/direction, player side/role, target/pass context, ball landing x/y | Trajectory/interaction modeling and validation; receiver/defender movement ground truth | **No** — competition/evaluation package, not generalized live feed |

### Particularly useful 2025 public fields

The 2025 competition is strategically important because it proves the exact relationship labels we want can exist in a structured table:

- `wasRunningRoute`
- `routeRan`
- `inMotionAtBallSnap`
- `shiftSinceLineset`
- `motionSinceLineset`
- `blockedPlayerNFLId1`
- `blockedPlayerNFLId2`
- `blockedPlayerNFLId3`
- `pressureAllowedAsBlocker`
- `timeToPressureAllowedAsBlocker`
- `pff_defensiveCoverageAssignment`
- `pff_primaryDefensiveCoverageMatchupNflId`
- `pff_secondaryDefensiveCoverageMatchupNflId`

Tracking rows include player/ball IDs, frame/time, team, play direction, `x`, `y`, `s`, `a`, distance, orientation, motion direction and event tags.

### Particularly useful 2024 public fields/events

- tackle / assist / forced-fumble labels;
- PFF missed-tackle indicator;
- full-player and football coordinates;
- event tags including contact/tackle moments.

That makes BDB 2024 a practical truth set for evaluating first-contact detection, tackler attribution, closing angle and yards-after-contact extraction.

### External source references

- NFL Big Data Bowl overview: https://operations.nfl.com/gameday/analytics/big-data-bowl/
- 2020: https://www.kaggle.com/competitions/nfl-big-data-bowl-2020
- 2021: https://www.kaggle.com/competitions/nfl-big-data-bowl-2021
- 2022: https://www.kaggle.com/competitions/nfl-big-data-bowl-2022
- 2023: https://www.kaggle.com/competitions/nfl-big-data-bowl-2023
- 2024: https://www.kaggle.com/competitions/nfl-big-data-bowl-2024
- 2025: https://www.kaggle.com/competitions/nfl-big-data-bowl-2025
- 2026 analytics: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics
- 2026 prediction: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction

---

## 7. Explicit DO NOT BUILD list

Unless the proposal adds materially richer grain, do **not** spend proprietary-data effort recreating:

1. schedules/opponents/kickoff metadata;
2. weekly rosters/player identity;
3. generic depth-chart rank;
4. weekly player box stats;
5. play-by-play down/distance/clock/score;
6. basic pass/rush/dropback indicators;
7. air yards and YAC;
8. EPA/CPOE;
9. generic QB hit/pressure/sack counts;
10. shotgun/no-huddle/play-action flags;
11. generic man/zone/shell rates;
12. defenders-in-box/light-box/heavy-box rates;
13. average NGS separation;
14. average cushion;
15. average intended air yards/aDOT;
16. expected YAC/YACOE weekly aggregates;
17. PFR defender target/completion/YPT/rating aggregates;
18. motion/screen/RPO aggregate history merely to recreate M81;
19. blitz-count/out-of-pocket/throwaway aggregates merely to recreate M81;
20. drops/catchable/interception-worthy aggregate history merely to recreate M81;
21. injuries/practice designations;
22. stadium/weather feed;
23. generic team OL/DL/tendency/pace tables;
24. current WR `primary_cb` / `shadow_flag` scrape as if it were exact responsibility;
25. generic WR/TE/RB target entitlement or depth-order workload ranking;
26. another historical YPR/YPT/YAC/xYAC/YACOE transform for RB receiving-yard mean;
27. another same-data STACK6/M95T-style RB rushing tail overlay;
28. another static-defense / aggregate-explosive-player matchup layer;
29. a “nearest defender = assignment” database;
30. Big Data Bowl competition rows copied into the repo and described as complete NFL history.

---

## 8. Genuinely missing / worth acquiring or building

### Priority 1 — route-level receiver geometry + true coverage responsibility

**Status:** `SOURCE_BLOCKED` for exact responsibility; `GENUINELY_MISSING` for extensible route/frame geometry history.

**Why it is new:** current repo data are weekly/player aggregates or current matchup summaries. They do not answer, route by route:

- who released from where;
- what route was run;
- who was responsible at each route phase;
- whether defender was press/off and with what leverage;
- whether help/bracket existed;
- separation at release, stem, break, throw and arrival;
- whether a receiver created an open window but was not targeted;
- whether a defender exchanged responsibility in zone/banjo concepts.

**Failure-mode relevance:** directly aligned with the remaining WR/TE receiving-efficiency/translation problem and useful to QB decision/open-window research.

**Multi-position value:** very high — WR, TE, QB, DB/team defense.

**Temporal integrity:** target-game route outcome is post-kickoff and must never be used directly in a pregame forecast. Build strictly-prior player/team/coach tendencies from completed games.

**Preferred economic mode:** `BUY` if a complete historical+current exact-assignment contract is affordable; otherwise `HYBRID`.

### Priority 2 — true blocker-rusher responsibility + protection geometry

**Status:** `SOURCE_BLOCKED`.

Need:

- blocker -> primary/secondary/tertiary rusher assignment;
- engagement start/release;
- pass-set depth/angle where derivable;
- rusher alignment/path;
- double-team/chip/help state;
- time to pressure / pocket displacement;
- pressure responsibility rather than aggregate pressure count.

**Why it is new:** M85 explicitly demonstrates that existing nflverse/Sharp/PFR pressure and line tables are not true assignments.

**Multi-position value:** QB, OL, TE/RB pass protection, pass rush/team defense.

**Preferred economic mode:** `BUY` or `HYBRID`. Pure build is possible but semantic assignment QA is expensive.

### Priority 3 — RB first-contact + tackle/run geometry

**Status:** `GENUINELY_MISSING` at extensible history/live scale.

No implemented repo family was found under the audited obvious aliases for yards-before-contact, yards-after-contact, missed/broken tackles, `pff_missedTackle`, `rushLocationType`, or run-concept/point-of-attack labels.

Recommended core schema:

- play/player IDs;
- ball carrier;
- handoff/snap/contact/tackle timestamps;
- first-contact x/y;
- line-of-scrimmage x;
- yards before first contact;
- yards after first contact;
- tackler/missed-tackler IDs;
- number of defenders in contact window;
- nearest free defender at handoff and at first contact;
- closing speed / pursuit angle;
- ball-carrier speed/acceleration at contact;
- blocker proximity / unblocked-defender indicator;
- contact outcome / broken-contact continuation.

**Why it is new:** current RB science centers on workload/allocation, conventional efficiency and distribution calibration. These are physical play-interaction features, not another historical box-score transform.

**BDB validation:** 2024 tackling data provide coordinates, event tags and PFF missed-tackle labels for a bounded 2022 slice.

**Preferred mode:** `HYBRID` — much of contact detection can be automated from tracking/video, with human QA for ambiguous multi-player collisions.

### Priority 4 — run concept / point-of-attack / run-blocking responsibility

**Status:** `GENUINELY_MISSING` / potentially `SOURCE_BLOCKED` depending on commercial source.

Desired labels:

- designed run concept (inside/outside zone, duo, power, counter, draw, sweep, etc. where reliably chartable);
- intended gap / point of attack;
- OL/TE/FB blocking responsibility;
- defensive front/alignment and fit responsibility;
- penetration / displacement by assigned relationship.

This is harder than contact detection because concept and assignment are semantic coaching labels. It should not be the first proprietary build unless a provider makes it economical.

### Priority 5 — ball trajectory / throw-window geometry

**Status:** `HAVE_AGGREGATE_ONLY` / `GENUINELY_MISSING` for complete frame/trajectory history.

2D player/ball geometry can add receiver/defender/QB interaction detail beyond CPOE and target result. Exact 3D ball flight is materially harder and should not be assumed obtainable from standard All-22. Treat 2D landing/arrival and timing as the realistic first target.

---

## 9. Ranking of the new-information frontier

This rank is about **information novelty and build/acquisition value**, not a prediction that any feature will improve the model.

| Rank | Family | Novelty vs repo | Failure-mode relevance | History/live potential | Automation feasibility | Human QA burden | BDB validation | Multi-position value | Preferred mode |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | Route geometry + true coverage responsibility | Very high | Very high (WR/TE efficiency, QB windows) | Commercial: potentially high; self-build depends on video | Medium for tracks; low-medium for semantics | High for identity/responsibility | Excellent (2021/2025/2026 slices) | Very high | **BUY/HYBRID** |
| 2 | Blocker-rusher assignment + protection geometry | Very high | High for QB/protection | Commercial potentially high | Medium | High for assignment semantics | Excellent (2023/2025) | Very high | **BUY/HYBRID** |
| 3 | RB first-contact/tackle geometry | High | High for RB efficiency/ceiling decomposition | Historical film build possible; live expensive | Medium-high for event geometry after tracks | Medium | Excellent (2024) | High | **HYBRID** |
| 4 | Run concept + point-of-attack + run-block assignment | Very high | Potentially high | Hard self-build; commercial charting preferable | Low-medium | Very high | Partial | High | **BUY/HYBRID** |
| 5 | 2D ball/throw-window trajectory | High | Medium-high | Tracking license > video reconstruction | Medium | Medium | Strong (2021/2026) | High | **BUY/HYBRID** |
| 6 | Isolated route-name/alignment database without responsibility geometry | Medium | Medium | Buildable | High | Medium | Strong | Medium-high | **BUILD only as subproduct** |
| 7 | New generic player/team advanced aggregates | Low | Low after repo closures | Easy | High | Low | N/A | Low | **SKIP** |

---

## 10. Computer-vision / video feasibility architecture

### Required pipeline

`All-22/video -> field calibration/homography -> player detection -> multi-object tracking -> team classification -> player identity -> ball/event timing -> x/y trajectories -> route/contact segmentation -> responsibility semantics -> separation/leverage/help/bracket -> human QA -> structured temporally versioned database`

### 10.1 Video source choice

**All-22 is strongly preferred** for receiver/defender route and coverage work because it keeps the secondary and route development in frame. Broadcast video routinely cuts/zooms away from off-ball receivers and safeties and therefore cannot support a reliable full-route responsibility database.

Broadcast can still be useful for selected RB contact/tackle events where the ball carrier is centered, but it will be inconsistent for full formation/secondary geometry.

Before any bulk video extraction, conduct a separate rights/licensing/terms review. This document does not make a legal conclusion about permitted use.

### 10.2 Field calibration / homography

**Automation:** feasible.

Use sidelines, yard lines, hash marks and known field dimensions to map pixels to field x/y. The hard cases are camera pan/zoom, partial field visibility, cut transitions, lens distortion and frames with limited line landmarks.

**Human review:** low-to-medium if automated confidence flags are good.

**Validation:** compare recovered x/y to BDB tracking slices on matching plays where video is available/licensed.

### 10.3 Player detection and multi-object tracking

**Automation:** feasible with modern detection + MOT.

Hard cases:

- line-of-scrimmage occlusion;
- piles/contact;
- receivers crossing;
- sideline personnel;
- camera cuts;
- identical uniforms/nearby players.

Track continuity is essential; a single identity swap can corrupt route/responsibility features downstream.

### 10.4 Team assignment and player identity

Team classification from uniform/field side is easier than exact player identity.

**Exact identity is one of the hardest operational problems** because All-22 jersey numbers can be small, blurred or hidden. Jersey OCR, roster constraints, position/alignment priors and cross-play track linkage can reduce work, but human verification should be expected for ambiguous plays.

### 10.5 Ball and event synchronization

The football is small and frequently obscured. Exact ball tracking from video alone is difficult.

Realistic first objective:

- snap frame;
- handoff frame;
- throw/release frame;
- catch/arrival frame;
- first-contact frame;
- tackle/end frame.

These event times plus player x/y are enough to derive many valuable features even without full 3D ball trajectory.

### 10.6 Route segmentation/classification

Once player tracks are credible, route path features are relatively automatable:

- release angle;
- depth;
- break point;
- curvature;
- speed changes;
- final direction/location.

BDB route labels can supervise/validate a route classifier. Hard cases include option routes, scramble drills, route conversions against leverage, picks/rubs and busted plays.

### 10.7 Coverage responsibility

This is the hardest semantic layer and should **not** be replaced by nearest-defender logic.

Man, match-zone, banjo/switch, brackets, route distribution and post-snap exchanges can make the nearest defender different from the defender with primary responsibility.

Recommended system:

1. geometry-based candidate responsibility probabilities;
2. coverage shell/assignment context;
3. primary/secondary responsibility output with confidence;
4. mandatory human review below a confidence threshold;
5. validation against BDB PFF primary/secondary matchup labels.

### 10.8 Leverage, help and bracket

These become much more feasible *after* trajectories and responsibility are reliable.

Potential derived fields:

- inside/outside/top leverage at release/break/throw;
- defender cushion by route phase;
- safety/help distance and angle;
- bracket probability/state;
- catch-point defender advantage;
- open-window area and duration.

They are geometry-derived but still depend on correct semantic assignment.

### 10.9 RB contact extraction

Compared with coverage responsibility, first-contact geometry is more automatable:

- identify ball carrier;
- detect handoff/possession;
- calculate defender distances/closing vectors;
- detect collision/contact candidates;
- identify first contact and tackle/end frames;
- compute pre/post-contact distance.

Multi-defender collisions, arm tackles, glancing contact and simultaneous contact require human QA. BDB 2024 is the best public truth slice for validation.

---

## 11. BUY vs BUILD vs HYBRID vs SKIP

### BUY

Use when a vendor can provide:

- exact route/frame/assignment/contact concepts;
- broad multi-season history;
- stable player IDs;
- current-season updates before the next game;
- documented field definitions/versioning;
- license terms compatible with internal modeling;
- enough raw or auditable detail to avoid opaque score-only dependence.

Best BUY candidates: true coverage responsibility; blocker-rusher assignment; run-concept/blocking charting.

### BUILD

Best for deterministic derivatives once raw tracks exist:

- separation by route phase;
- leverage geometry;
- open-window duration;
- pursuit/closing angle;
- yards before/after detected contact;
- motion/path descriptors;
- route-path embeddings.

Do not pay a premium merely for transformations we can reproduce from credible raw/assignment data.

### HYBRID — preferred long-run proprietary strategy

1. obtain lawful video and/or licensed/raw tracking where practical;
2. use public Big Data Bowl slices as development and validation truth;
3. automate field calibration, detection, tracking, geometry and high-confidence event detection;
4. use human reviewers for player identity and semantic responsibilities;
5. store both raw tracks and labels, not only final metrics;
6. derive strictly-prior player/team/coach tendencies for pregame models;
7. measure inter-rater/algorithm agreement and version every labeling rule.

This concentrates manual labor where machines are weakest instead of paying humans to chart information the repo already gets elsewhere.

### SKIP

Skip proprietary build of:

- generic box stats;
- generic efficiency rates;
- generic shells/man-zone/box;
- NGS-style average separation/cushion alone;
- public pressure counts;
- weather/injury/depth-chart aggregation;
- model-repackaging features already closed by M82/M81/M75/R-series.

---

## 12. Temporal integrity rules for any new proprietary database

A rich postgame dataset can still be useful to a pregame model, but only through **strictly prior** history.

Every record should carry:

- `game_id`, `play_id`, season/week/date;
- source/video/tracking version;
- label version;
- charted-at / available-at timestamp if operationally relevant;
- player IDs with identity confidence;
- model/algorithm version for machine-generated labels;
- human-review status/confidence.

For target game G:

- no route, coverage, contact, blocker-rusher or tracking observation from G may enter G's pregame feature set;
- no future-game data may enter historical rolling summaries;
- within-week features must respect actual availability time, not merely season/week labels;
- competition labels released long after a season can be used for retrospective research, but they do not prove that the same source was deployable in-season;
- a historical source without a current update contract is not automatically production-deployable.

---

## 13. Anti-retest registry by position / research family

### QB

Closed or absorbed unless materially new source information arrives:

- static/richer defense;
- market/game-script repackaging;
- attempt trust/range/decompression;
- extreme-error classifier/directional surprise;
- possession/dropback generative variants;
- model combinations;
- offensive intent/opening playcaller;
- QB volatility;
- aggregate explosive-weapon matchup;
- opportunity oracle;
- limited dropback context;
- M75 NGS/PFR aggregate tracking matchup;
- personnel discontinuity/inactives;
- M80/M81 FTN tactical/decision/error families;
- M83 comparable-opponent defensive adaptation.

### WR

Do not repeat:

- M72 aggregate explosive weapon x defense;
- M75 average separation/cushion/aDOT/YACOE/secondary-quality interactions;
- R7 persistent explosive/YAC/air-yard traits;
- R9-R11 NGS residual attempts;
- R3 combined residual-persistence calibration family;
- C1 target-mass adjustment;
- C3 broad joint combinations;
- ND3 vacated/dynamic entitlement;
- blind M38/R15 retuning;
- fabricated player-level CB responsibility from on-field/nearest-defender evidence.

### RB

Do not repeat:

- STACK6 team-rush-context slicing;
- M95T detached retrospective rushing-tail overlay;
- depth-order/remap as workload authority;
- R23-R27D conventional receiving mean transforms using historical YPR/YPT/YAC/target shape/xYAC/YACOE;
- point-mean rescue of the already-studied ceiling problem without new causal information.

RB-PD2's yard-difficulty width result is a separate research-qualified calibration lane and does not disposition contact/run geometry.

### TE

Do not repeat:

- team-TE-pool-only residual correction without individual differentiation;
- generic participation as a blanket correction;
- another target-entitlement model when R5P already governs entitlement.

Open legitimate TE information is route/matchup/efficiency data richer than current participation.

---

## 14. Recommended acquisition/build sequence

This is a data-engineering sequence, **not authorization to run model experiments**.

### Phase A — vendor/source contract audit before building anything

For each candidate provider, ask for a sample dictionary and answer:

- exact grain;
- seasons and missing-game rate;
- whether every route/play is included;
- historical revision behavior;
- player-ID system;
- current-season latency;
- whether pre-snap/postgame timestamps are documented;
- whether primary/secondary coverage responsibilities are actual charted assignments or inferred nearest defenders;
- whether blocker-rusher relationships are true assignments;
- whether raw coordinates/tracks are included;
- license/modeling rights.

If a source delivers Priority 1-3 with broad history and live updates at tolerable cost, buying is likely cheaper and faster than charting from scratch.

### Phase B — public ground-truth proof of pipeline

Without running a predictive NFL-prop experiment, use Big Data Bowl only to validate extraction quality:

- recover x/y against NGS coordinates;
- route classification against BDB route labels;
- responsibility against BDB primary/secondary PFF matchup IDs;
- blocker-rusher assignment against BDB fields;
- contact/missed-tackle events against BDB 2024 labels.

The acceptance metric at this phase is **data fidelity**, not projection MAE or betting performance.

### Phase C — limited human-QA pilot

Chart a small, legally usable historical sample only after a source/license decision. Measure:

- track continuity error;
- identity accuracy;
- route-label agreement;
- assignment agreement;
- contact-event timing error;
- human minutes per play;
- ambiguous/unresolvable rate.

Do not scale until the semantic error rate and cost are known.

### Phase D — only then decide scale

If route/assignment/contact labels are reliable and economically feasible, backfill historical seasons and construct strictly-prior features. A separate frozen scientific plan would be required before testing any candidate in the predictive stack.

---

## 15. What would change this ranking?

The ranking should be revisited if any of the following becomes true:

- a provider exposes a stable, machine-readable multi-season + current NGS responsibility feed;
- PFF or another vendor offers exact route/coverage/blocking/contact data at a cost below internal annotation;
- the NFL releases broader tracking data beyond competition slices;
- All-22 licensing/availability makes automated historical extraction impractical;
- a future canonical repo handoff demonstrates that one of these missing families was already acquired/tested after this audit base SHA.

Until then, the most defensible frontier is **relationships and geometry**, not another layer of conventional advanced statistics.

---

## 16. Exact repo lineage reviewed for this audit

Primary governance / continuity:

- `AGENTS.md`
- `CURRENT_NFL_RESEARCH_HANDOFF.md`
- `NFL_MASTER_CONTINUITY_RECORD.md`
- current WR receiving-yards handoff/evidence documents
- current RB-PD2 plan/run lineage

Core research/source authorities:

- `docs/migrations/M82_RESULT.md`
- `scripts/backtest/build_m82_integration_ledger.py`
- `docs/migrations/M83_RESULT.md`
- `docs/migrations/M84_RESULT.md`
- `docs/migrations/M84_TOP_WEAPON_ESCAPE_HATCH_SOURCE_AUDIT.md`
- `scripts/backtest/audit_top_weapon_escape_hatch_sources.py`
- `docs/migrations/M85_RESULT.md`
- `docs/migrations/M85_TRUE_BLOCKER_RUSHER_SOURCE_AUDIT.md`
- `scripts/backtest/audit_true_blocker_rusher_sources.py`
- `docs/migrations/M81_FROZEN_FEATURE_DICTIONARY.md`
- `docs/migrations/M81_RESULT.md`
- `scripts/backtest/audit_qb_personnel_tracking_matchup.py`
- `scripts/backtest/run_qb_personnel_tracking_matchup.py`
- `scripts/backtest/enrich_historical_defense.py`
- `scripts/backtest/audit_participation_source.py`
- `scripts/backtest/audit_wr_cb_source.py`
- `docs/research/overnight/WR_GAP_FINDINGS.md`
- `docs/research/overnight/RB_POST_WEEK1_GAP_FINDINGS.md`
- `docs/research/RB_PD2_YARD_DIFFICULTY_MC_WIDTH_V1_PLAN.md`
- `docs/research/overnight/TE_GAP_FINDINGS.md`

Source/build plumbing:

- `scripts/utils/pbp.py`
- `scripts/player_stats_loader_v2.py`
- `scripts/utils/build_team_week_map_v2.py`
- `scripts/backtest/historical_inputs.py`
- `scripts/player_identity_roster_history_v1.py`
- `scripts/backtest/build_historical_injuries.py`
- `scripts/build/build_weather_week.py`
- `scripts/providers/ourlads_depth.py`
- `scripts/providers/sharpfootball_pull.py`
- `scripts/build/build_cb_coverage_player.py`
- `scripts/build/build_wr_cb_exposure.py`
- `scripts/build/build_coverage_v2.py`

This is a family-level inventory, not a claim that every column in every source table or every experimental branch was individually re-read. Where the repo has a canonical terminal synthesis/ledger, this audit uses that synthesis rather than reopening every superseded branch.

---

## 17. Canonical decision rule going forward

Before anyone proposes buying, scraping or charting a new NFL statistic, classify it in this order:

1. **Do we already have the raw field?** -> use it.
2. **Can we derive it from existing raw fields at the same informational grain?** -> derive it.
3. **Did we already test this family and close it?** -> do not retest it with cosmetic transforms.
4. **Do we only have an aggregate while the proposal adds exact relationship/frame/assignment information?** -> that may be genuinely new.
5. **Is the only public example a Big Data Bowl competition slice?** -> use it for ground truth/validation, not as a live-history claim.
6. **Can the resulting information exist before kickoff through strictly-prior history?** -> if no, it cannot be a direct pregame feature.
7. **Would buying the exact semantic labels be cheaper than reconstructing them from video?** -> compare BUY vs HYBRID before building.
8. **If building, can algorithmic fidelity be validated independently before any predictive experiment?** -> require yes.

The strategic goal is not to own the most football columns. It is to own the **small number of relationship-level variables that the existing data stack cannot already tell us**.
