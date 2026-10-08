# PLAYER OUTPUT COMPONENT DECOMPOSITION V1 — RESULT

Date: 2026-10-08  
Branch: `research-player-output-component-decomposition-v1`

## Frozen disposition

`OPPORTUNITY_DOMINANT_ACROSS_ALL_PRIMARY_MARKETS__INDIVIDUAL_ROLE_SHARE_ALLOCATION_NEXT`

Diagnostic only. No production promotion.

## Evidence authority

Canonical latest exact-head workflow:
- run: `37777319154`
- head: `cb1e511adcb58e40fc41d0949306083a304e83dd`
- conclusion: SUCCESS
- tests: SUCCESS
- decomposition: SUCCESS
- certification: SUCCESS
- strict repository audit: SUCCESS

Artifact:
- id: `11550446909`
- name: `player-output-component-decomposition-v1-37777319154`
- digest: `sha256:7acfc453258a49f3aadf0d0ea386b13827eb715f1dfb055b96505f733320d204`
- retention expiry: 2027-01-06

Earlier successful scientific authority `37771878672` is superseded for
continuity by the later exact-head rerun above. The scientific disposition is
unchanged; the later run aligns the final target-identity regression tests with
the GSIS-roster resolver.

Frozen parents:
- all-player W1-W4 point replay run `37683439543`
- replay-matched baseline opportunity rows from run `37687979574`

No sportsbook/player-prop fields were used upstream.
No paid OddsAPI call was made.
No model parameter was fit.
No automatic promotion occurred.

## Population and integrity

Primary paired rows: **4,017**  
Oracle-scoreable rows: **3,991**  
Rows with unavailable model-effective efficiency: **21**

Grading-source exclusions:
- rows: **5**
- unique player-weeks: **2**
- reason: `GRADING_SOURCE_CONFLICT_UNRESOLVED_PBP_TARGET`

Those player-weeks were retained in row-level evidence but excluded from every
oracle. No target count was fabricated.

Target grading:
- completed-game PBP target rows: **2,056**
- verified zero-receiving-use fallback rows: **1,360**
- resolved PBP-vs-frozen target-count discrepancies: **0**

Algebraic integrity:
- max baseline reconstruction gap:
  `2.842170943040401e-14`
- max full-actual identity gap:
  `5.684341886080802e-14`

Both are inside the frozen `1e-10` tolerance.

## Primary result

The opportunity oracle improved MAE more than the efficiency oracle in **all
eight** predeclared position/market cells.

| Position | Market | Baseline MAE | Opportunity oracle MAE | Opp. MAE removed | Efficiency-subset baseline MAE | Efficiency oracle MAE | Eff. MAE removed | Disposition |
|---|---|---:|---:|---:|---:|---:|---:|---|
| QB | pass_yards | 73.995 | 56.328 | **23.9%** | 61.274 | 53.469 | 12.7% | OPPORTUNITY_DOMINANT |
| RB | rush_yards | 19.846 | 10.893 | **45.1%** | 21.273 | 19.732 | 7.2% | OPPORTUNITY_DOMINANT |
| RB | rec_yards | 10.466 | 5.747 | **45.1%** | 11.529 | 8.357 | 27.5% | OPPORTUNITY_DOMINANT |
| RB | receptions | 1.180 | 0.505 | **57.2%** | 1.175 | 1.203 | **-2.4%** | OPPORTUNITY_DOMINANT |
| WR | rec_yards | 22.275 | 12.739 | **42.8%** | 23.404 | 16.800 | 28.2% | OPPORTUNITY_DOMINANT |
| WR | receptions | 1.511 | 0.689 | **54.4%** | 1.479 | 1.308 | 11.6% | OPPORTUNITY_DOMINANT |
| TE | rec_yards | 15.615 | 7.026 | **55.0%** | 16.238 | 12.423 | 23.5% | OPPORTUNITY_DOMINANT |
| TE | receptions | 1.427 | 0.453 | **68.3%** | 1.347 | 1.238 | 8.1% | OPPORTUNITY_DOMINANT |

The efficiency percentages use the frozen efficiency-eligible subset, while the
opportunity percentages use the full identity-valid market population, exactly
as specified in the frozen contract. They are oracle diagnostics, not causal
effect estimates.

## Error-transmission evidence

Final-output error is strongly associated with individual opportunity error:

| Position | Market | Corr(opportunity error, output error) | Corr(efficiency error, output error) |
|---|---|---:|---:|
| QB | pass_yards | **0.814** | 0.491 |
| RB | rush_yards | **0.828** | 0.401 |
| RB | rec_yards | **0.789** | 0.619 |
| RB | receptions | **0.923** | 0.330 |
| WR | rec_yards | **0.749** | 0.553 |
| WR | receptions | **0.871** | 0.383 |
| TE | rec_yards | **0.788** | 0.515 |
| TE | receptions | **0.907** | 0.336 |

The opportunity-error correlation is larger in every predeclared cell.

## Interpretation

The current individual-player stack is not mainly failing because it lacks one
more generic YPT/YPC/YPA/catch-rate adjustment.

The dominant remaining problem is **how much opportunity the named player
receives**:

- QB: pass-attempt / active-QB workload state
- RB: carries and targets
- WR: targets
- TE: targets

This is consistent with the all-player replay's earlier workload-compression
finding: low-workload players are systematically projected too high and
high-workload players too low.

The receiving-count markets are especially decisive. Perfect conversion does
little for RB/WR/TE receptions compared with correcting target workload.

## Protected conclusion

Do **not** respond to this result by opening another generic efficiency,
coverage, matchup, or game-script multiplier.

Per the frozen next-step rule, the next model-development work must remain
inside **individual pregame role/share/opportunity allocation**.

That means answering player-specific questions such as:
- Is this player actually expected to participate?
- What role does he occupy in this exact offense this week?
- How much of the team's pass/rush opportunity should belong to him?
- Has his role changed strictly before kickoff?
- How should teammate availability redistribute opportunity?
- Is he a low-, middle-, or high-workload player whose current allocator is
  compressing him toward the room average?

Opponent matchup and per-opportunity efficiency remain downstream layers; this
result does not say they are irrelevant. It says they are **not the dominant
current bottleneck**.

## No production action

`automatic_promotion = false`

This result authorizes the next bounded individual role/share allocation
research lane only. It does not authorize a point-projection correction,
threshold, multiplier, or live production change.
