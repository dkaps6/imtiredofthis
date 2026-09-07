# WR-R7 YPR-Mechanism-Conditioned Efficiency — Result

## Canonical evidence
- Branch: `research-wr-r7-ypr-mechanism-conditioned-efficiency`
- Run: `34070862759`
- Job: `101587816590`
- Head SHA: `3880e422ce15f12e472d46d4dc295df8394dbe8d`
- Artifact: `10000400972`
- Artifact digest: `sha256:8919294419926254e8d022ff7c7276a8af13c5146d24a8e39796c9e5547cddd8`
- Frozen plan commit: `2a77a8e4208627e200a9782e0123eabbb6d967a6`
- Evaluator commit: `180d2abff0067d5aeb133aa508057c79c125bbf9`

## Integrity
- Full exact WR-R5/ND6 rows: **2,130**
- Qualifying WR-R5 profiles: **133**
- YPR-dominant qualifying players: **16**
- YPR-conditioned player-games: **200**
- Sportsbook inputs used: no
- Model fitting used: no
- Production changed: no

## Official disposition
**`NO_ACTIONABLE_WR_YPR_CONDITIONED_EFFICIENCY_SIGNAL`**

Zero of the eight frozen signals passed every candidate gate. The family gate therefore failed. No threshold, history window, interaction, or population definition is changed after the result.

## Player-side signals
- `PLAYER_EXP20_PER_TARGET_PRIOR8`: Spearman **-0.179723**, YPR-component Q4-Q1 **-1.4775 yd**, total receiving-yard residual gap **-1.1654 yd**, under-25 enrichment **1.1494x**, positive player-Spearman rate **0.0625**.
- `PLAYER_EXP40_PER_TARGET_PRIOR8`: Spearman **-0.082030**, YPR-component gap **-1.2851 yd**, total residual gap **+2.0702 yd**, tail enrichment **1.0627x**.
- `PLAYER_YAC_PER_RECEPTION_PRIOR8`: Spearman **-0.122597**, YPR-component gap **-6.7088 yd**, total residual gap **-2.2531 yd**.
- `PLAYER_AIR_PER_TARGET_PRIOR8`: Spearman **-0.091237**, YPR-component gap **-6.7190 yd**, total residual gap **-4.9179 yd**.

The player-side results are not merely below gate; most point weakly in the opposite direction. Persistent player explosive/YAC/air-yard traits therefore do not explain the game-to-game YPR miss mechanism for this subgroup.

## Defense-side signals
- `DEF_EXP20_PER_ATT_ALLOWED_PRIOR8`: Spearman **-0.031835**, YPR gap **-1.4634 yd**, total residual gap **-9.0692 yd**.
- `DEF_EXP40_PER_ATT_ALLOWED_PRIOR8`: Spearman **0.025958**, YPR gap **-3.0465 yd**.
- `DEF_AIR_PER_ATT_ALLOWED_PRIOR8`: Spearman **-0.063084**, YPR gap **-3.5570 yd**, tail enrichment **1.1613x**.

### Strongest descriptive signal: defensive YAC allowance
`DEF_YAC_PER_COMPLETION_ALLOWED_PRIOR8` was the only signal with a substantial mechanism-consistent YPR gap:
- N **200**, coverage **1.00**
- Spearman **0.079565** — FAIL vs >=0.10
- Q4-Q1 YPR-component gap **+4.8993 yd** — PASS
- Q4-Q1 total receiving-yard residual gap **+1.3584 yd** — FAIL vs >=5.0
- under-25 tail enrichment **0.9032x** — FAIL vs >=1.20
- W2-18 YPR-component gap **+5.7047 yd** — PASS
- W13-18 YPR-component gap **+5.5829 yd** — PASS
- 16 players with >=8 valid rows; positive player-Spearman rate **0.625** — PASS

This remains a failed candidate. It is recorded as descriptive evidence only and cannot be rescued by lowering the correlation, residual-gap, or tail gates.

## Scientific interpretation
WR-R5 correctly identified a real YPR-dominant subgroup, but WR-R7 shows that the obvious persistent efficiency proxies are not enough to forecast its game-level YPR misses. The next materially new YPR lane should therefore use different pregame information—route/coverage/tracking/QB-delivery or similarly richer matchup mechanics—rather than recycling static player explosiveness, YAC, or air-yard history.
