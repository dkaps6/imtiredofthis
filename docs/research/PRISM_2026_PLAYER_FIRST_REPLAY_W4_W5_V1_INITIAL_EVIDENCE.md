# PRISM 2026 Player-First Replay — Weeks 4-5 V1 — Initial Evidence

Date: 2026-10-17

Status: **INITIAL DIAGNOSTIC EVIDENCE / PLAYER-FIRST REPLAY ACTIVE / NO MODEL CHANGE**

Frozen protocol:
`docs/research/PRISM_2026_PLAYER_FIRST_REPLAY_W4_W5_V1_PROTOCOL.md`

Protocol freeze:
`8d5d5cf45f98849a202cced5d7f4e9187a7ee32d`.

This document records the first replay finding from already-preserved Week-4 /
Week-5 authorities. It does not fit a model or promote a mechanism.

## 1. First question

Before inventing another position-level adjustment, ask:

> Are the same named players reappearing among PRISM's largest misses?

If yes, that is evidence that individual-player state / reliability deserves
direct forensic attention rather than treating every miss as an independent
position-average error.

## 2. Week-4 repeat-player evidence

Parent:
`failure_mode_top100_standardized_misses.csv` from canonical run
`37485600879` / artifact `11423535484`.

Within the global Weeks 1-4 top-100 standardized-miss set:
- Week-4 rows: **30**;
- Week-4 unique players: **22**;
- Week-4 rows belonging to a player already present in a Weeks 1-3 extreme miss:
  **9 / 30 = 30.0%**;
- Week-4 unique repeat identities: **7 / 22 = 31.8%**.

Repeat identities:
- Ashton Jeanty;
- CeeDee Lamb;
- Jaylin Noel;
- Kenneth Walker III;
- Kyren Williams;
- Ollie Gordon II;
- Tyquan Thornton.

This is descriptive selected-tail evidence, not a population rate.

### Concrete repeated high-side misses

Kenneth Walker III:
- Week 1 rush yards: model **27.10**, actual **173**;
- Week 1 rush+rec: model **39.71**, actual **191**;
- Week 2 rush+rec: model **45.68**, actual **178**;
- Week 4 rush yards: model **62.26**, actual **177**;
- Week 4 rush+rec: model **80.38**, actual **179**.

CeeDee Lamb:
- Week 2 receiving yards: model **54.41**, actual **153**;
- Week 4 receiving yards: model **60.34**, actual **189**;
- Week 4 receptions: model **4.09**, actual **17**.

Tyquan Thornton:
- Week 2 receiving yards: model **18.92**, actual **86**;
- Week 4 receiving yards: model **23.18**, actual **111**.

Jaylin Noel:
- Week 1 receiving yards: model **13.40**, actual **53**;
- Week 4 receiving yards: model **17.53**, actual **85**.

Ollie Gordon II:
- Week 3 rushing yards: model **13.62**, actual **41**;
- Week 4 rushing yards: model **28.46**, actual **100**.

These cases immediately create a PRISM question:

> What strictly-prior player identity / role / teammate / allocation state was
> available before the later game, and why did the projection still remain far
> below the player's realized opportunity/production?

Do not answer that by simply adding yards after the fact.

## 3. Week-5 recurrence against Weeks 1-4

Week-5 parent:
- casebook run `38397361891`;
- artifact `11812403368`;
- top 40 largest absolute final-model player-market misses;
- **36 unique players**.

Weeks 1-4 parent:
- canonical top-100 standardized-miss atlas from `37485600879`.

Overlap:
- **11 / 36 Week-5 unique failure identities = 30.6%** had already appeared
  among the Weeks 1-4 top-100 extreme misses;
- those repeat identities generated **13 / 40 = 32.5%** of the Week-5 top-40
  failure rows.

Repeat identities:
- Aaron Jones;
- Braelon Allen;
- Brock Bowers;
- CeeDee Lamb;
- Deshaun Watson;
- Jahmyr Gibbs;
- Jayden Daniels;
- Jaylen Warren;
- Jordan Love;
- Malik Nabers;
- Ollie Gordon II.

Again: selected-tail descriptive evidence only.

## 4. Repeat identities skew toward opportunity involvement

Among the **13 Week-5 failure rows** belonging to prior extreme-miss identities:
- opportunity-dominant: **6**;
- mixed opportunity + efficiency: **4**;
- efficiency-dominant: **3**.

Therefore opportunity is involved in:
- **10 / 13 = 76.9%** of repeat-player Week-5 failure rows.

Among the **27 Week-5 failure rows** whose players were not in the prior
Weeks 1-4 top-100 extreme set:
- opportunity-dominant: **8**;
- mixed: **3**;
- efficiency-dominant: **16**.

Opportunity is involved in:
- **11 / 27 = 40.7%** of those nonrepeat rows.

This does **not** prove a causal or predictive difference because the sample is
conditioned on extreme misses.

It does identify an immediate player-first forensic priority:

> repeated known-player misses appear especially likely to contain unresolved
> opportunity / role-allocation information.

## 5. High-priority repeat opportunity cases

### Aaron Jones

Earlier:
- Week 2 rush yards: model **39.40**, actual **105**;
- Week 2 rush+rec: model **35.75**, actual **105**.

Week 5:
- rush yards model **41.77**, actual **113**;
- predicted rushing opportunities **10.91**;
- actual rushing opportunities **25**;
- predicted efficiency **4.17**;
- actual efficiency **4.52**;
- label: **OPPORTUNITY_DOMINANT**.

The Week-5 miss is overwhelmingly about how much rushing work Aaron Jones
received, not an exotic YPC event.

### Braelon Allen

Week 4:
- rush yards model **21.39**, actual **59**.

Week 5:
- rush yards model **36.78**, actual **116**;
- predicted rushing opportunities **6.92**;
- actual **24**;
- predicted efficiency **4.05**;
- actual **4.83**;
- label: **OPPORTUNITY_DOMINANT**.

### Jahmyr Gibbs

Week 1:
- rush yards model **59.47**, actual **156**;
- rush+rec model **80.50**, actual **186**.

Week 5:
- rush yards model **52.84**, actual **151**;
- predicted rushing opportunities **12.94**;
- actual **25**;
- predicted efficiency **4.65**;
- actual **6.04**;
- rush-yard label: **OPPORTUNITY_DOMINANT**;
- rush+rec label: **MIXED_OPPORTUNITY_EFFICIENCY**.

### Malik Nabers

Week 4:
- receiving yards model **44.37**, actual **112**.

Week 5:
- receiving yards model **52.71**, actual **121**;
- predicted targets **6.63**;
- actual targets **20**;
- predicted YPT **7.93**;
- actual YPT **6.05**;
- label: **OPPORTUNITY_DOMINANT**.

The player produced more yards despite *lower* efficiency than projected because
the target allocation miss was enormous.

### Ollie Gordon II

Earlier:
- Week 3 rush yards model **13.62**, actual **41**;
- Week 4 rush yards model **28.46**, actual **100**.

Week 5:
- rush yards model **33.47**, actual **88**;
- predicted rushing opportunities **5.76**;
- actual **18**;
- predicted efficiency **3.96**;
- actual **4.89**;
- label: **OPPORTUNITY_DOMINANT**.

This is now a three-week repeated named-player forensic priority.

## 6. Repeat identity does not imply a one-way mean correction

The replay also immediately warns against simply boosting repeat-miss players.

### CeeDee Lamb

Earlier:
- Week 2 receiving: **54.41 -> 153**;
- Week 4 receiving: **60.34 -> 189**;
- Week 4 receptions: **4.09 -> 17**.

Week 5:
- receiving yards: **81.04 -> 9**;
- predicted targets **8.26 -> actual 5**;
- predicted YPT **9.24 -> actual 1.80**;
- label: **MIXED_OPPORTUNITY_EFFICIENCY**.

The same identity can move from repeated high-side misses to a huge low-side
miss. That is a reliability/state problem, not permission to apply a permanent
upward correction.

### Deshaun Watson

Earlier extreme identity:
- Week 1 rushing-yards high-side miss.

Week 5:
- pass yards **238.23 -> 155**;
- predicted attempts **24.77 -> actual 33**;
- predicted YPA **7.28 -> actual 4.70**;
- label: **MIXED_OPPORTUNITY_EFFICIENCY**.

Again, the player identity recurs, but the exact failure mechanism changes.

## 7. Repeat efficiency / fragility cases

Not every repeat identity is an allocation problem.

Jordan Love:
- Week 1 pass yards **193.73 -> 387**;
- Week 5 pass yards **232.09 -> 367**;
- Week-5 predicted attempts **27.52 -> actual 31**;
- predicted YPA **7.32 -> actual 11.84**;
- Week-5 label: **EFFICIENCY_DOMINANT**.

Jayden Daniels:
- Week 2 rush yards **22.74 -> 69**;
- Week 5 rush yards **21.14 -> 103**;
- Week-5 predicted rush opportunities **4.43 -> actual 9**;
- predicted rushing efficiency **4.83 -> actual 11.44**;
- Week-5 label: **EFFICIENCY_DOMINANT**.

Jaylen Warren:
- Week 3 rush yards **49.83 -> 127**;
- Week 3 rush+rec **67.00 -> 176**;
- Week 5 rush yards **50.88 -> 101**;
- Week-5 opportunities **11.53 -> 15**;
- efficiency **4.70 -> 6.73**;
- Week-5 label: **EFFICIENCY_DOMINANT**.

These players belong in the same-player reliability / efficiency-history audit,
not a generic role boost.

## 8. Immediate interpretation

The first replay pass already supports the user's governing PRISM concern:

**individual player history is not merely decorative context.**

A material share of the later extreme-miss population consists of players PRISM
had already missed extremely earlier in the season.

The next question is not:
- how do we reduce RB MAE?;
- how do we reduce WR MAE?;
- which position needs +X yards?

It is:

> For each repeat identity, what did PRISM know before kickoff about this
> player's historical playbook, current role, teammate competition, opportunity
> allocation, opportunity subtype, opponent interaction, efficiency, and
> reliability—and which of those facts failed to reach the final projection?

## 9. Exact next forensic queue

First queue — repeated opportunity-involved identities:
1. Aaron Jones;
2. Braelon Allen;
3. Jahmyr Gibbs;
4. Malik Nabers;
5. Ollie Gordon II;
6. Brock Bowers.

Second queue — repeated efficiency/reliability identities:
1. Jordan Love;
2. Jayden Daniels;
3. Jaylen Warren.

State/reliability reversal queue:
1. CeeDee Lamb;
2. Deshaun Watson.

For each player, build a strictly-prior pregame state sheet for the later target
week before proposing any new mechanism.

## 10. Integrity

This initial evidence pass:
- paid OddsAPI calls: **0**;
- sportsbook refetches: **0**;
- Week-6 outcomes used: **0**;
- new coefficients fit: **0**;
- new thresholds fit: **0**;
- production changes: **0**;
- projection means changed: **0**.

Disposition:

`PRISM_PLAYER_FIRST_REPLAY_ACTIVE__REPEAT_PLAYER_FAILURE_SIGNAL_FOUND`.

This is a diagnostic prioritization result, not a production promotion.
