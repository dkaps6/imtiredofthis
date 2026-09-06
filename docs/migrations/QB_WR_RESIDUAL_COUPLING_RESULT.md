# QB-WR Residual Coupling — Result

## Canonical run
- Run: `34065446410`
- Tested SHA: `35b53145ec541fdf6ebac42e9e44432df49c085e`
- Artifact: `9998790011`
- Artifact SHA256: `f43480d4e70929dd234ab449d9a2ee29bbe728a18ef434af5dd18442e515a6e3`
- Aligned QB team-weeks: 884 (2024-2025)
- Sportsbook inputs: none
- Production changed: no

## Frozen all-WR residual result
- Pearson: **0.728638**
- Spearman: **0.701332**
- Same-sign rate: **0.656109**
- Q4-minus-Q1 QB residual gap: **125.972 yards**
- Pearson 2024: **0.720650**
- Pearson 2025: **0.737857**
- Spearman 2024: **0.695397**
- Spearman 2025: **0.707500**

All frozen strong-coupling gates passed.

## Pregame-selected WR subsets
- Top 2 WRs by pregame M38 projection: Pearson 0.612281, Spearman 0.565455, same-sign 0.639140, Q4-Q1 gap 102.626 yards.
- Top 1 WR by pregame M38 projection: Pearson 0.450222, Spearman 0.384745, same-sign 0.608597, Q4-Q1 gap 73.795 yards.

## Official disposition
**`STRONG_QB_WR_ERROR_COUPLING`**

This is a historical error-structure finding, not a deployable pregame feature. QB passing yards and receiver yards are mechanically related, so some coupling is expected; the result shows that M89 QB residuals and M38 WR residuals share substantial error structure across both seasons. It does not rescue M72 or M75 and does not authorize realized WR outcomes upstream of QB.

A future QB bridge still requires information available pregame. A natural next structural question is whether shared pregame team pass-volume / attempt uncertainty explains the common errors, especially given the separate QB attempt-vs-YPA and WR reception-vs-YPR mechanism decompositions.
