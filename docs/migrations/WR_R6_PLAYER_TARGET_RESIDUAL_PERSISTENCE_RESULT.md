# WR-R6 Player Target Residual Persistence — Result

## Canonical evidence
- Branch: `research-wr-r6-player-target-residual-persistence`
- Workflow run: `34070137942`
- Job: `101585837406`
- Artifact: `10000180409`
- Artifact digest: `sha256:ea5ca807937f5e6de85868683d126aa31adc5f67023b8b278a1b577dc8096d3b`
- Frozen plan commit: `56ed34856b18ea6b522bc0372d84629c2df82acd`
- Evaluator commit: `3cec6d731547797003129c277e98316c6d967748`
- Launch commit: `208ec54f565e8fbf2aab292be0e4acb015b3e667`

## Integrity
- Exact WR-R5 casebook rows: 2,130
- Sportsbook inputs used: no
- Model fitting used: no
- Production changed: no
- Lagged signals use only earlier player-games; current/future outcomes are excluded.

## Frozen primary signal
`PRIOR4_WR_SHARE_RESID`: prior up-to-4 (minimum 3) player-game mean of within-WR target-share residual, where each historical residual is:

`actual WR target share - projected WR target share`

Primary evaluation was restricted to rows where the player's current M38 WR role matched the immediately previous game's M38 role.

## Result
Official disposition: **`NO_ACTIONABLE_WR_PLAYER_TARGET_PERSISTENCE_2025`**.

The signal was descriptively meaningful but failed exactly one preregistered gate. Per the frozen protocol, the gate is not lowered and the result is not promoted.

### Primary stable-role metrics
- Signal coverage: **0.708920** — PASS (>= 0.65)
- Stable-role rows with signal: **1,256** — PASS (>= 700)
- Pearson vs current allocation residual: **0.201464**
- Spearman vs current allocation residual: **0.177823** — PASS (>= 0.10)
- Q4-Q1 allocation-residual gap: **0.068629** — PASS (>= 0.025)
- Q4-Q1 raw-target-residual gap: **1.03731 targets** — PASS (>= 0.75)
- Same-sign rate: **0.546178** — **FAIL** (required >= 0.58)
- W2-18 Spearman: **0.177823** — PASS (> 0)
- W13-18 Spearman: **0.230652** — PASS (> 0)
- WR1 Spearman: **0.08306**
- WR2 Spearman: **0.04285**
- WR3 Spearman: **0.15363**
- Positive WR1/WR2/WR3 slices: **3 of 3** — PASS (required >=2)

### Secondary / contrast
- Prior-4 raw-target residual stable-role Spearman: **0.078408**
- Role-change contrast: N **254**, Spearman **0.176121**

## Interpretation
There is real descriptive evidence that the model's within-WR target allocation residual has player-specific persistence, especially later in the season. However, the preregistered same-sign reliability threshold did not clear. That means WR-R6 does **not** authorize a player-specific target correction, a threshold change, a nearby-window retry, or a production modification.

This result still sharpens the research map: player identity contains some stable target-entitlement information, but a player's recent residual alone is not reliable enough to become the forecasting mechanism. The next WR work must add genuinely new pregame football information or a newly justified mechanism-conditioned interaction, then pass a separately frozen full-stack test.
