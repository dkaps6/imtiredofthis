# Player Situational Target Earning V1 — Source / Nonredundancy Result

**STATUS: COMPLETE / PLAYER_SITUATIONAL_TARGET_EARNING_SOURCE_READY**

Frozen plan:
`docs/research/PLAYER_SITUATIONAL_TARGET_EARNING_V1_PLAN.md`

Authority:
- branch `research-player-situational-target-earning-v1`
- run `37635210417` — SUCCESS
- source SHA `9685cdf45e665f824cd7e7153705e270a0a1448a`
- artifact `11489645374`
- digest `sha256:6c83ba1973cc7c5961e8f277402cb50a143db1c73ec6305c6d56a5c652ad6da5`

Final disposition:

`PLAYER_SITUATIONAL_TARGET_EARNING_SOURCE_READY`

## Live Week-5 boundary

Only 2026 Weeks 1-4 were read.

- scheduled teams: 30
- Week-5 roster source: Week 5
- rostered WR/TEs: 464
- WR/TEs with at least one 2026 target: 247
  - WR: 156
  - TE: 91
- stable identity coverage: 100%
- receiver-ID coverage on 2026 target events: 100%
- context denominator coverage: 100% for every frozen context
- player-share coverage: 100%
- Week-5 outcomes read: 0
- sportsbook inputs: 0
- candidate models fit: 0
- production changed: false

## Nonredundancy vs ordinary overall target share

### Early down — REDUNDANT
- Spearman vs overall share: 0.9714
- delta SD: 0.0194
- only 4/247 players differ by >=5 percentage points

Do not carry early-down share forward as a distinct V1 state.

### Third down — MATERIALLY NONREDUNDANT
- Spearman vs overall: 0.8731
- delta SD: 0.0535
- 68/247 players differ by >=5 points

### Red zone — MATERIALLY NONREDUNDANT
- Spearman vs overall: 0.7337
- delta SD: 0.0754
- 102/247 players differ by >=5 points

### Two minute — MATERIALLY NONREDUNDANT
- Spearman vs overall: 0.7959
- delta SD: 0.0708
- 88/247 players differ by >=5 points

Three of four predeclared situational contexts contain information not reducible to ordinary target share.

## Interpretation

The player-level opportunity residual found independently in WR and TE has a plausible free pregame state family available for scientific testing:

- third-down target earning;
- red-zone target earning;
- two-minute target earning.

These are individual-player role signals inside a team context. They are materially different from:
- position priors;
- snap share;
- generic receiver-room targets-per-play;
- overall target share alone.

This source result does not establish predictive value.

The next authorized step is a separately frozen historical residual diagnostic against the exact promoted WR and TE opportunity authorities.

No production change is authorized.
