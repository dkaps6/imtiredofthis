# Football Matchup Transmission V1 — Integration Candidates Freeze

Status: **FROZEN BEFORE CANDIDATE SCORING — RESEARCH ONLY — NO PRODUCTION CHANGE**

Parent diagnostic:
- Phase B/C run `37514137803`
- artifact `11436786668`
- result: `docs/research/FOOTBALL_MATCHUP_TRANSMISSION_V1_PHASE_BC_RESULT.md`

The three candidates below are independent. They are not a coefficient search, grid, ensemble, or head-to-head model-selection exercise. Each is scored against the unchanged canonical football baseline and may pass or close on its own frozen gate.

No candidate may be modified after seeing its score. A failed candidate closes in this V1 form.

## Shared historical scoring contract

Seasons:
- 2024 Weeks 2-18
- 2025 Weeks 2-18

Forbidden:
- 2026 outcomes
- sportsbook / prop lines / odds
- OddsAPI
- target-game usage
- coefficient fitting
- threshold search
- top-N selection
- post-hoc role carveouts

Baseline:
- reproduce the exact football-only historical projection authority used by the right-tail diagnostic;
- frozen MC / ML / State components and frozen ensemble weights;
- candidate changes only the declared football seam;
- ML and State components remain unchanged;
- same rows, same seeds, same target-week pregame universes.

Candidate scoring:
- primary target = absolute-error improvement: `abs(base - actual) - abs(candidate - actual)`;
- support floor = 200 rows, 50 games, 25 players;
- paired game-cluster and player-cluster bootstrap, 5,000 reps, seed `20261006`;
- each candidate must improve primary-market MAE in BOTH 2024 and 2025;
- game-cluster and player-cluster 95% lower bounds for primary absolute-error improvement must be > 0 in BOTH seasons;
- primary RMSE must be non-worse in BOTH seasons;
- candidate also fails if any predeclared collateral skill-position cohort shows statistically supported harm in either season under BOTH game and player clustering (upper 95% CI for AE improvement < 0);
- no minimum yard threshold is fitted.

A historical PASS only authorizes a separate live/parity forward-shadow contract. It does not authorize immediate production promotion.

---

## Candidate FMT-INT-RB-DEF-PASS-RATE-FACED-V1

Diagnostic source:
`RB_RUSH / def_pass_rate_faced`

### Football mechanism

Current generic game script:
`pass_share = 0.57`

Candidate:
`pass_share = clip(opponent_pass_rate_faced, 0.35, 0.75)`

where `opponent_pass_rate_faced` is the exact strict-prior M89/M90 semantic used by Phase B/C.

Why this seam:
- `pass_rate_faced` is already an absolute dropback/pass-environment rate;
- lower values mechanically create more team rushing opportunities;
- the candidate changes conserved team pass/rush volume, not RB YPC and not a player-specific yardage multiplier;
- no M95A/M95B run-defense boost is introduced.

Fallback:
- if exact pass-rate-faced is unavailable, retain baseline `0.57`.

Primary cohort:
- RB/FB/HB `rush_yards`.

---

## Candidate FMT-INT-WR-TRUE-PROE-V1

Diagnostic source:
`WR_REC / off_true_proe`

### Football mechanism

Current generic game script:
`pass_share = 0.57`

Candidate:
`pass_share = clip(0.57 + true_proe, 0.35, 0.75)`

where `true_proe` is the exact strict-prior promoted M89/M90 semantic.

Why this seam:
- PROE is a percentage-point pass-tendency delta;
- adding it to the existing neutral 0.57 anchor preserves units;
- this directly repairs the confirmed production behavior in which populated `rules_pass_rate` prevents simulation's PROE fallback from executing;
- no player-specific WR multiplier is introduced.

Fallback:
- if exact true PROE is unavailable, retain baseline `0.57`.

Primary cohort:
- WR `rec_yards`.

---

## Candidate FMT-INT-TE-DEF-PASS-SUCCESS-V1

Diagnostic source:
`TE_REC / def_pass_success_allowed`

### Football mechanism

Candidate modifies TE receiving efficiency only; it does not change target share.

For each target week, build the exact strict-prior opponent `def_pass_success_allowed` state and the cross-team pregame league mean of that same field.

Define:
`te_success_eff_mult = clamp(1 + (opp_def_pass_success_allowed - league_mean_def_pass_success_allowed), 0.50, 1.80)`

Candidate TE rule:
`rules_ypt_candidate = rules_ypt_baseline * te_success_eff_mult`

Why this seam:
- pass-success allowed is an efficiency variable;
- the transformation is dimension-preserving: a +0.10 success-rate difference produces a 1.10 multiplier;
- it uses no fitted slope;
- the clamp reuses the existing canonical matchup-multiplier safety bounds rather than introducing an outcome-searched range;
- target entitlement and catch-rate authority remain untouched.

Fallback:
- missing exact pass-success state -> multiplier 1.0.

Primary cohort:
- TE `rec_yards`.

---

## Predeclared collateral cohorts

Every candidate is also audited on:
- RB rush yards
- RB rush+receiving yards
- RB receiving yards
- WR receiving yards
- TE receiving yards

QB pass yards is not a promotion gate because M89/M90 remains the separate production authority and M56/M83 remain closed.

## Terminal dispositions

For each candidate independently:

PASS:
`HISTORICAL_INTEGRATION_PASS_FREEZE_FORWARD_SHADOW`

FAIL:
`HISTORICAL_INTEGRATION_FAIL_CLOSED`

No rescue tuning is authorized after scoring.
