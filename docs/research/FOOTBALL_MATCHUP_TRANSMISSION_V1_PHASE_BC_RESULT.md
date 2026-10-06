# Football Matchup Transmission V1 — Phase B/C Result

Status: **COMPLETE — THREE SEPARATE INTEGRATION CANDIDATES AUTHORIZED FOR FROZEN HISTORICAL SCORING — NO PRODUCTION CHANGE**

Branch: `research-football-matchup-transmission-v1`

Canonical run:
- run `37514137803`
- artifact `11436786668`
- digest `sha256:ccc0a3962f5505bdb0465fb8be8ff0f158f3781064cbffabc875361ed99db8ef`
- head `a8cf0731675228e2eea8c25411dae5f8b2f187fc`

Frozen methods:
`docs/research/FOOTBALL_MATCHUP_TRANSMISSION_V1_PHASE_BC_METHODS.md`

## Contract integrity

The successful run certified:
- 2024 + 2025 only;
- Weeks 2-18 only;
- sportsbook inputs = 0;
- candidate models fit = 0;
- Phase A rerun = false;
- production changed = false;
- 38,258 skill-position residual rows;
- 894 frozen M89/M90 QB control rows;
- 1,024 target team-week rows;
- 62 predeclared Phase B/C specs tested.

Inference required both:
- game-cluster 95% bootstrap support;
- player-cluster 95% bootstrap support.

A feature replicated only if expected direction cleared both clustering gates in BOTH seasons.

## Replicated results

Four Phase-B specs replicated. Three have exact live semantics and are not closed prior families, so they may advance only to separately frozen integration-candidate scoring.

### 1. RB rushing — opponent pass-rate-faced

Candidate source:
- cohort: `RB_RUSH`
- feature: `def_pass_rate_faced`
- family: game environment
- orientation: lower opponent pass-rate-faced = more favorable rushing environment
- parity: `EXACT_LIVE_SEMANTICS`

2024:
- rows 1,313
- games 256
- players 141
- Spearman rho 0.08161
- game-cluster 95% CI [0.02210, 0.13811]
- player-cluster 95% CI [0.02880, 0.13396]

2025:
- rows 1,313
- games 256
- players 145
- Spearman rho 0.07175
- game-cluster 95% CI [0.01981, 0.12449]
- player-cluster 95% CI [0.02161, 0.12030]

Disposition:
`INTEGRATION_CANDIDATE_ELIGIBLE`

This does NOT reopen M95A/M95B. The direct `def_rush_epa` and usage×run-defense interaction lanes did not replicate. The surviving signal is specifically the opponent tendency / pass-rush environment seam.

### 2. WR receiving yards — offensive true PROE

Candidate source:
- cohort: `WR_REC`
- feature: `off_true_proe`
- family: game environment
- orientation: higher offensive true PROE = more favorable receiving environment
- parity: `EXACT_LIVE_SEMANTICS`

2024:
- rows 1,998
- games 256
- players 224
- Spearman rho 0.05181
- game-cluster 95% CI [0.00629, 0.09846]
- player-cluster 95% CI [0.00414, 0.09817]

2025:
- rows 1,999
- games 256
- players 222
- Spearman rho 0.04654
- game-cluster 95% CI [0.00481, 0.08885]
- player-cluster 95% CI [0.00081, 0.09014]

Disposition:
`INTEGRATION_CANDIDATE_ELIGIBLE`

This points directly at the already-confirmed architecture defect where canonical rules always populate `rules_pass_rate = 0.57`, shadowing the PROE fallback in simulation.

### 3. TE receiving yards — defensive pass-success allowed

Candidate source:
- cohort: `TE_REC`
- feature: `def_pass_success_allowed`
- family: receiving defense
- orientation: higher opponent pass-success allowed = more favorable TE receiving environment
- parity: `EXACT_LIVE_SEMANTICS`

2024:
- rows 1,023
- games 256
- players 118
- Spearman rho 0.06281
- game-cluster 95% CI [0.00148, 0.12304]
- player-cluster 95% CI [0.01105, 0.11241]

2025:
- rows 1,049
- games 256
- players 125
- Spearman rho 0.10435
- game-cluster 95% CI [0.04886, 0.15872]
- player-cluster 95% CI [0.03699, 0.17026]

Disposition:
`INTEGRATION_CANDIDATE_ELIGIBLE`

### 4. TE position YPT allowed — diagnostic replication only

`def_te_ypt_allowed` replicated in Phase B, but historical source parity to the current live Sharp position-YPT field is not proven.

Disposition:
`REPLICATED_DIAGNOSTIC_SOURCE_PARITY_BLOCKED`

It cannot authorize production integration from this audit.

## Phase C

No predeclared usage × matchup interaction replicated across both 2024 and 2025.

Therefore:
- no bellcow carveout;
- no top-N usage filter;
- no role threshold;
- no target-game usage;
- no M95A/M95B rescue.

## Source-blocked / parity-unproven families

No candidate is authorized from:
- Sharp `dl_stuff_rate`;
- Sharp `dl_ybc_per_rush`;
- outside YPT;
- slot YPT;
- middle-open rate;
- historical public box-rate proxies;
- historical public man/zone proxies;
- historical public position-YPT proxies.

## Next authorized action

The three exact-semantic Phase-B signals above must be frozen as independent integration candidates BEFORE any implementation is scored.

Diagnostic replication does not authorize a production change.
