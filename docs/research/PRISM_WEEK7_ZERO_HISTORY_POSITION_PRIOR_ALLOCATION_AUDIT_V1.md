# PRISM Week-7 Zero-History Position-Prior Allocation Audit V1

**Date:** 2026-10-26  
**Lane:** player-first architecture / outcome-blind production forensic  
**Scientific status:** **ARCHITECTURE DEFECT CONFIRMED / NO PREDICTIVE PROMOTION**  
**Outcome use:** **0**  
**Sportsbook reacquisition:** **0**  
**New OddsAPI calls:** **0**  
**Production changes:** **0**

## 1. Why this audit exists

The Walker-first environment decomposition exposed a separate player-first question:

> Can a current production player with no player-specific history receive a large opportunity share purely from the position prior, and then compete with named players inside the finite team opportunity room?

Under PRISM's permanent rule:

> THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.

this must be answered at slate scale rather than treated as one Walker anomaly.

This audit uses only the already-paid Week-7 pregame artifact.

Canonical authority:
- run `39849371821` — SUCCESS
- exact head `4dc178b59da8708dbea978b8e54e753b1e2f75c5`
- artifact `12456298985`
- digest `sha256:d1a1fb48e5bb920c5291f35b7457b5870801b689751deb2d742f98c555603a85`

No target-game outcomes are used.

## 2. Exact production mechanism

The generic Bayesian player baseline in `scripts/modeling/bayesian_v2.py` uses:

1. a position-group population mean;
2. prior-season player evidence when available;
3. current-season player evidence when available.

For a player with:
- `prior_games = 0`; and
- `current_games = 0`;

the posterior contains only the position-group population prior.

The artifact labels this state explicitly:

`bayes_evidence_state = position_prior_only`

That posterior then becomes the player's `bayes_rush_share` / `bayes_tgt_share`.

The canonical rules layer generally passes those shares through unless an opponent/injury rule changes them.

The joint simulator then treats those player rows as actual competitors for finite team opportunity:
- target shares are conserved to a 0.95 modeled-player cap;
- rushing uses the top five player rushing shares, then caps/scales their modeled probability mass to 0.95.

Therefore a position prior is not merely a shrinkage fallback for a player mean. In the current generic opportunity path it can become **player-level entitlement mass**.

## 3. Week-7 population

The full paid Week-7 football universe contained:
- **396** skill-player rows;
- **28** playing teams in the target slate.

Zero-player-history definition for this audit:

`prior_games == 0 AND current_games == 0`

Population:
- **34** zero-history players;
- across **21 / 28** playing teams.

Zero-current-season-history definition:

`current_games == 0`

Population:
- **69** players;
- across **25 / 28** playing teams.

The second population includes players with prior-season evidence but no 2026 game evidence. It is reported separately and must not be conflated with true zero-history rows.

## 4. Position-prior values inherited by zero-history players

Representative exact artifact values:

### RB / FB position-prior-only rows

Examples:
- Ben VanSumeren, KC FB1
- Dean Connors, LAR RB3
- Max Bredeson, MIN FB1
- Riley Nowakowski, PIT FB1
- Audric Estim, CLE RB3

All five have:
- prior games = **0**
- current games = **0**
- `bayes_evidence_state = position_prior_only`
- `bayes_rush_share = 0.359568`
- `bayes_tgt_share = 0.079174`

The position prior is identical because no player evidence exists.

### QB position-prior-only rows

Representative zero-history backups:
- Carson Beck
- Haynes King
- Sam Howell
- Sam Ehlinger
- Behren Morton
- Cole Payton
- Ty Simpson
- Will Howard

Each receives:
- `bayes_rush_share = 0.111763`

despite zero prior/current player games.

### WR position-prior-only rows

Representative rows:
- Tank Dell, HOU WR6
- Hollywood Brown, PHI WR7
- CJ Daniels, LAR WR6
- Camden Brown, DAL WR6
- Simi Fehoko, ARI WR6
- Michael Sturdivant, GB WR5
- Deandre Carter, DET WR5
- Montorie Foster, SEA WR5

Each receives:
- `bayes_tgt_share = 0.165842`

before finite-team conservation / WR specialist redistribution.

### TE position-prior-only rows

Representative rows:
- Cameron Latu, NE TE3
- Drew Ogletree, IND TE3
- Seydou Traore, MIA TE3
- Carsen Ryan, CLE TE3

Each receives:
- `bayes_tgt_share = 0.129391`

before generic opponent rules and TE room redistribution.

These are not inferred values; they are exact as-run production artifact fields.

## 5. Rushing-room materiality

The simulator selects each team's top five `rules_rush_share` rows, then scales their total modeled probability mass to at most 0.95.

Across the 28-team Week-7 slate:

### Zero-history players

- zero-history rows entering a team's top-five rush allocator: **13**
- teams affected: **11 / 28**
- combined final rushing allocation probability: **1.824446**
- analytic expected carries assigned from current rules volume: **45.829**
- share of total modeled top-five carry expectation: **6.87%**

Representative zero-history carry expectations:

| Team | Player | Role | Final rush probability | Analytic expected carries |
|---|---|---|---:|---:|
| LAR | Dean Connors | RB3 | 0.244277 | 6.474 |
| MIN | Max Bredeson | FB1 | 0.252108 | 6.109 |
| KC | Ben VanSumeren | FB1 | 0.228210 | 5.587 |
| PIT | Riley Nowakowski | FB1 | 0.202400 | 5.213 |
| CLE | Audric Estim | RB3 | 0.216676 | 5.111 |
| CAR | Haynes King | QB3 | 0.103840 | 2.740 |
| DAL | Sam Howell | QB2 | 0.089532 | 2.366 |
| PHI | Cole Payton | QB2 after availability rerank | 0.094430 | 2.337 |

These are analytic expectations from the production input probabilities and team rush-volume means, not realized game outcomes and not a replay of the sampled Monte Carlo arrays.

### Zero-current-season-history players

Players with `current_games = 0` occupy an even larger share:

- top-five rushing rows with zero current-season games: **30**
- teams affected: **21 / 28**
- combined final rushing allocation probability: **4.014335**
- analytic expected carries: **101.156**
- share of modeled top-five carry expectation: **15.17%**

This does not mean every zero-current player is invalid. It establishes that the current allocation path can transmit stale/prior/position evidence into material target-week opportunity without a separate active-role conditional allocation gate.

## 6. Target-room materiality

The explicit target-entitlement layer conserves modeled player target probability to **0.95 per team**.

Across 28 playing teams:
- total modeled player target entitlement = **26.6**

### Zero-history players

Final target entitlement assigned to zero-history players:
- **1.264686**
- **4.75%** of all modeled player target mass.

Zero-history players with:
- final target entitlement >= 0.05: **17**
- final target entitlement >= 0.10: **1**

Examples:

| Team | Player | Role | Final target entitlement |
|---|---|---|---:|
| HOU | Tank Dell | WR6 | 0.101660 |
| NE | Cameron Latu | TE3 | 0.096442 |
| PHI | Hollywood Brown | WR7 | 0.087182 |
| IND | Drew Ogletree | TE3 | 0.083646 |
| LAR | CJ Daniels | WR6 | 0.077610 |
| MIA | Seydou Traore | TE3 | 0.070615 |
| DAL | Camden Brown | WR6 | 0.069596 |
| PIT | Riley Nowakowski | FB1 | 0.066288 |
| ARI | Simi Fehoko | WR6 | 0.065070 |
| CLE | Carsen Ryan | TE3 | 0.063888 |
| KC | Ben VanSumeren | FB1 | 0.057154 |

### Zero-current-season-history players

Final target entitlement assigned to players with `current_games = 0`:
- **2.134477**
- **8.02%** of all modeled player target mass.

Again, some of these players may legitimately deserve opportunity. The defect is not "zero current games means force zero." The defect is that position/stale priors can become finite player entitlement without an independently validated target-week player-role state deciding whether that evidence should be active.

## 7. FB -> RB role semantics amplify the issue

The current availability reranker contains:

`prefix = "RB" if grp in {"RB","FB"} else grp`

So eligible FBs are reranked inside the RB-style role namespace.

Examples in the paid artifact:
- Ben VanSumeren: raw depth `FB1` -> `role_after_availability = RB1`
- Max Bredeson: raw depth `FB1` -> `RB1`
- Riley Nowakowski: raw depth `FB1` -> `RB1`

All three also have zero player history and inherit the RB position-group rush prior.

This combination is particularly dangerous:
1. no player evidence;
2. RB-family population prior;
3. RB1-style reconciled role label;
4. finite top-five rushing allocation.

This audit does not claim every fullback should be removed. It identifies a deterministic semantic path that can make the **position prior look like named-player RB1 entitlement**.

## 8. Walker is a representative consequence, not the evidence source

For KC specifically:

- Walker Bayes rush share = **0.524947**
- Ben VanSumeren zero-history position prior = **0.359568**
- Justin Fields has **0 current games** and Bayes rush share = **0.220009**
- KC top-five raw rush-share sum = **1.496822**
- simulator scales that room to 0.95
- Walker effective final rushing probability = **0.333173**
- analytic expected carries = approximately **8.16**

The Walker game/result is not needed to establish the defect.

The defect is already visible before kickoff in:
- player evidence state;
- role semantics;
- raw room totals;
- finite-room normalization.

## 9. What this audit does NOT authorize

Do not respond by:
- setting all zero-history players to zero;
- deleting position priors;
- hard-coding depth-chart thresholds from Week 7;
- using named-player exceptions;
- fitting an RB1/RB2/RB3 coefficient from the Week-7 outcome;
- bypassing the already-qualified Stage-1/Stage-2 participation/allocation research.

Position priors remain valid as priors.

The unresolved question is how much **target-week named-player authority** they should have when player-specific participation/role evidence is absent, stale, or contradictory.

## 10. Relationship to the corrected Player Identity Card

The blocked Identity Card V1 audit already identified the need to:

- create the target-week player universe from legal pregame roster authority;
- preserve active zero-opportunity weeks rather than dropping them;
- separate historical identity from current-team state;
- emit explicit sparse/no-history state;
- bridge identities stably.

This population audit adds one concrete requirement:

> The descriptive card must make `position_prior_only`, `prior_only`, and `prior+current` evidence states explicit at player-week grain, and must expose whether production opportunity is coming from player evidence or population fallback.

That is descriptive state, not a new projection coefficient.

## 11. Relationship to Stage-2

Historical conditional allocation is already qualified in:
- RB rushing;
- RB receiving;
- TE receiving.

It is not currently wired to production.

The frozen Week-7 Stage-2 source-readiness re-probe remains the correct next source question.

This audit increases the importance of that bridge but does not pre-judge its Week-7 effect.

## 12. Required future trace

A production-quality player-first opportunity trace should expose, per target-week player:

1. target-week roster/availability authority;
2. role/depth state;
3. player evidence state:
   - position-prior-only;
   - prior-only;
   - prior+current;
4. raw player opportunity statistic;
5. Bayesian/posterior opportunity statistic;
6. any generic matchup opportunity multiplier;
7. active-role / participation authority;
8. conditional allocation authority if qualified;
9. room membership;
10. top-N inclusion, where applicable;
11. room normalization factor;
12. final finite allocation probability;
13. analytic expected team opportunity;
14. source / fallback reason.

That trace would make a Walker/VanSumeren-style conflict visible before the final projection rather than requiring post-hoc code archaeology.

## 13. Disposition

Disposition:

`PRISM_WEEK7_POSITION_PRIOR_PLAYER_ENTITLEMENT_ARCHITECTURE_DEFECT_CONFIRMED__NO_PREDICTIVE_PROMOTION`

Confirmed:
- zero-history position priors materially enter player-level opportunity allocation;
- this occurs across many teams, not one named-player case;
- rushing and receiving rooms are both affected;
- FB-to-RB role semantics can amplify the problem;
- current production lacks a sufficiently explicit target-week player authority between population prior and finite opportunity allocation.

Not claimed:
- the correct replacement formula is not known;
- no outcome was used;
- no promoted science is reopened;
- no production change is authorized.
