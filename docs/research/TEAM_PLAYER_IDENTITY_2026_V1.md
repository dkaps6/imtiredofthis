# Team identities, player roles, and how Vegas prices them: 2026 weeks 1-10

**Date:** 2026-11-20.

**Script:** `scripts/research/team_player_identity_2026_v1.py`.

**Rows:** `data/research/identity/`.

**Prior-work check:** no earlier study profiled team identities, tested whether team, player or
opponent misses against the line repeat, or decomposed how FanDuel builds player lines. Related
but different:
- `WR_ROOM_REGIME_INSTABILITY_AUDIT_V1` (room targets-per-play model);
- `PRISM_SITUATIONAL_EDGE_PREREGISTRATION_V1` (vacancy and alpha-tier hypotheses);
- `REVERSE_ENGINEER_2026_W1_10_V1` §2 (volume and script).

## 1. Teams have real identities, but not in volume

Odd-week vs even-week correlation across the 32 teams:

| real identity (repeats) | r | mostly noise (game flow) | r |
|---|---|---|---|
| offensive efficiency (EPA/play) | **0.62** | plays per game | 0.22 |
| pass/run lean (neutral PROE) | **0.59** | rushes / dropbacks per game | 0.27 / 0.28 |
| team quality (point margin) | **0.54** | defense EPA allowed | 0.24 |
| pass rate when trailing | 0.41 | pass rate when leading | -0.27 |
| | | **beating the spread** | **0.08** |
| | | **scoring vs Vegas implied points** | **-0.01** |

- **Who a team is repeats:** how efficient it is, whether it leans pass or run, and how good it
  is.
- **How many plays it gets does not.** That depends on game flow.
- **Beating Vegas does not repeat.** JAX covered by 8.8 points a game through week 10, and that
  says nothing about the next game.

**Identities, `team_profiles_2026_v1.csv`:**
- **Pass-first:** CIN (PROE +13.8; Chase 26% of targets), LA (+7.1; Nacua 36%), KC, CAR, HOU.
- **Run-first:** TB (-12.2), MIA (-10.1), LAC (-9.3), SEA (-7.0), ATL (-5.9), NYG, BUF, JAX.
- **Best teams:**
  - JAX +13.9/game, run-first;
  - BUF, CIN, SF, KC.
- **Worst offense:** LAC, EPA -0.16.
- **Worst teams:** LAC, ARI, MIA, PHI and TEN, all below -8 per game.

## 2. Players have real roles inside those identities

- Shares repeat strongly. The lesson of `REVERSE_ENGINEER_2026_W1_10_V1.md` §5 is that role
  repeats and efficiency does not.
- **One-receiver offenses:**
  - SEA: JSN 37%, the only receiver at 15% or more;
  - LA: Nacua 36%;
  - PHI: D. Smith 33%;
  - DAL: Lamb 32%;
  - MIN: Jefferson 31%;
  - DET: St. Brown 30%.
- **Bellcow backs:**
  - Gibbs 75% of carries, Walker 73%, Hubbard 73%, Pollard 69%;
  - Henry, Bijan, Pierce and J. Williams 66%.
- **Committees:**
  - NE Stevenson 39%, NO Estime 40%;
  - SF McCaffrey 44% and DEN Dobbins 44% (DEN a committee; SF a role split).
- **Good players on bad teams:**
  - ATL: Bijan Robinson, Drake London;
  - NYJ: Garrett Wilson, Breece Hall;
  - IND: Jonathan Taylor;
  - ARI: Trey McBride;
  - PHI (2-7): DeVonta Smith;
  - LV: Bowers, Jeanty;
  - TEN: Pollard.
  - Their roles are as large as stars' on good teams. The team context changes their **game
    script**, not their **share**.

## 3. How Vegas builds a player line (FanDuel, weeks 6-10)

| market | share of the line explained by the player's season average alone | weight on the last 3 games | line as % of season average | per point of team implied total |
|---|---|---|---|---|
| rec_yards | **84.5%** | 0.04 | **84%** | ~0 |
| rush_yards | **85.4%** | 0.04 | **90%** | +0.27 yds |
| receptions | **81.7%** | 0.11 | 95% | ~0 |
| pass_yards | 65.8% (78.8% with implied total) | 0.04 | 98% | **+3.0 yds** |

**A FanDuel player line is, to a first approximation, his season average shaded down 5-16%.**
- QB passing lines also move with the team's implied points.
- **Recent games get almost no extra weight.**
- The shade below the average matches the skew in §5 of `WHAT_WE_KNOW_2026_W11.md`: the median
  game sits below the mean.

## 4. Do Vegas's misses repeat? No. They reverse.

Odd-week vs even-week correlation of (actual - line):

| unit | all markets | rec_yards | rush_yards | receptions |
|---|---|---|---|---|
| **player** (252) | **-0.14** | **-0.27** | -0.20 | **-0.28** |
| team (32) | -0.46 | 0.02 | -0.14 | -0.31 |
| opponent defense (32) | 0.09 | 0.06 | 0.12 | 0.26 |

- **A player who beat his line in one set of weeks tends to fall short in the other.** Backing
  "the guy who keeps hitting" is backing the reversal.
- That is the quantitative reason hit history alone does not work. The operator's instinct that
  hit history is not enough is borne out here: at player level it is slightly backwards.
- Team-level misses do not repeat.
- Opponent defenses vs receptions show the only positive value (r 0.26, n=32). Weak; noted
  only.

## 5. Mispricing by team quality x star (weeks 6-10, blind board; a lead, not a finding)

- **Star receivers (top-25% rec_yards line) on good teams (prior point diff > +4):** OVER
  65.3%, 5/5 weeks, n=49.
- On bad teams: OVER 61.5%, n=26.
- **But middle teams' star receivers went UNDER 64.1%** (n=103).

This is non-monotonic and the cells are small, from a scan of many cells. It is consistent with
`PRISM_ALPHA_COMPRESSION_BOARD_CHECK_V1.md` §4(b) (the market set alpha lines about 10% low in
2026), which that study warns is a season regime that may not repeat. **Track it; do not bet a
rule on it.**

## 6. What this means

1. **Identities and roles are real, and Vegas already prices them.** The line is the season
   average with a shade. Knowing that SEA funnels everything to JSN, or that DET's backfield is
   Gibbs, is **not** an edge by itself.
2. **The edges are where Vegas's simple rule breaks:**
   - **Game script it under-adjusts:** big favorites run, so underdog RBs go under (the
     tracker rule).
   - **The skew it shades for but not fully:** rushing unders +5.2% blind; overs lost 7/7 weeks.
   - **Role changes it barely weights.** Recent games get 0.04-0.11 weight in the line, while
     RB carry share deserves about 0.7 on the last 3 games
     (`RB_CARRY_SHARE_RECENCY_RESULTS_V1.md`). Untested as a betting rule.
3. **Do not chase hit streaks or team ATS streaks.** Both reverse.
