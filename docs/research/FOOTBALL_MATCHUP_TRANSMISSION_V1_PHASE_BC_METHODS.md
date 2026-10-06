# Football Matchup Transmission V1 — Phase B/C Frozen Methods

Status: **FROZEN BEFORE PHASE B/C SCORING**

Parent plan: `docs/research/FOOTBALL_MATCHUP_TRANSMISSION_V1_PLAN.md`

Branch: `research-football-matchup-transmission-v1`

## 1. Purpose

Operationalize the already-authorized Phase B/C audit without changing the research question.

This is a diagnostic transmission audit. It does **not** fit a production model, choose coefficients, search thresholds, select top-N rows, carve out bellcows post hoc, use target-game usage, or use sportsbook data.

Phase A is already complete and must not be rerun.

## 2. Evaluation seasons and cutoffs

Score only:

- 2024 regular season Weeks 2-18
- 2025 regular season Weeks 2-18

Every matchup or opportunity feature must use evidence strictly before the target week.

Forbidden:

- 2026 outcomes
- target-week outcomes as features
- sportsbook/player-prop lines
- odds
- closing lines
- coefficient fitting

The Phase B/C runner must report `sportsbook_inputs_used = 0` and `candidate_models_fit = 0`.

## 3. Baseline residual authority

The skill-position residual baseline is the already-certified football-only 2024/2025 historical projection identity from Distribution Right-Tail Asymmetry V1:

- run `37493425352`
- artifact `11427113291`
- digest `sha256:5051111ebc20b79a334cfd1dc08ec8d8d52107e59acb023f7f5fefe4e5b771dd`

Use only identity, actual, and football projection-mean columns from that artifact.

Residual is:

`actual - existing_football_projection`

That makes Phase B conditional on the existing player/role projection rather than re-testing raw football outcomes.

QB pass yards is a **control only**. It must use the promoted M89/M90 football-only mean authority, not the generic skill-position conclusion. The control may diagnose residual association but can never reopen M56/M83 or authorize a new generic QB matchup candidate.

## 4. Source-parity classes

A descriptive historical signal is not automatically integration-eligible.

### A. Same-live-semantics / gate-eligible source families

These may cross the replication gate if all other requirements pass:

- M89/M90 corrected public-football context, reconstructed with the same strict-prior semantics:
  - `true_proe`
  - `neutral_pace_true`
  - `pass_rate_off`
  - `plays_est`
  - `def_pass_epa_allowed`
  - `def_pass_success_allowed`
  - `def_ypa_allowed`
  - `pass_rate_faced`
  - hit/sack pressure allowed/generated
- `def_rush_epa` reconstructed from nflverse PBP with the same defensive rushing-EPA definition used by `make_team_form.py`, using only current-season plays before the target week.

However, generic RB rushing role × run-defense promotion remains blocked by the M95A/M95B anti-retest ledger even if a descriptive replication is strong. Such a result may be reported, not promoted.

### B. Diagnostic-only because current live provider parity is not proven

These may be tested descriptively but cannot authorize integration from this audit:

- historical nflverse participation light/heavy box rates versus current live Sharp-preferred box rates;
- historical nflverse participation man/zone coverage versus current Coverage v2/Sharp semantics;
- historical official-stat position YPT allowed versus current Sharp position-YPT fields.

A replicated diagnostic signal here means “source acquisition/parity is worth solving,” not “promote the proxy.”

### C. Source-blocked for this audit

Do not substitute a look-alike statistic.

- Sharp `dl_ybc_per_rush` / YBC allowed
- Sharp `dl_stuff_rate`
- outside YPT allowed
- slot YPT allowed
- `middle_open_rate` when an exact historical same-semantic source is not available

These must appear in the source-readiness output as blocked rather than silently disappearing.

## 5. Frozen Phase B directions

All tested variables are oriented so **higher = more favorable for the target player market**. No sign may be flipped after outcomes are inspected.

### RB rush yards

- higher opponent defensive rush EPA allowed -> favorable
- higher offensive plays -> favorable
- faster neutral pace (lower seconds/play) -> favorable
- lower offense pass tendency / PROE -> favorable rushing volume
- lower defense pass-rate-faced -> favorable rushing environment
- higher light-box rate -> favorable, diagnostic-only
- lower heavy-box rate -> favorable, diagnostic-only

### WR receiving yards / TE receiving yards

- higher defensive pass EPA allowed -> favorable
- higher defensive YPA allowed -> favorable
- higher defensive pass success allowed -> favorable
- higher offense PROE/pass tendency -> favorable
- higher offensive plays -> favorable
- faster neutral pace -> favorable
- higher defense pass-rate-faced -> favorable
- lower pressure mismatch (defense generated minus offense allowed) -> favorable
- position YPT allowed -> favorable, diagnostic-only

TE zone exposure may be reported as diagnostic-only, with higher zone rate treated as favorable under the existing generic TE rule direction. No WR man/zone monotonic direction is promoted from this audit.

### RB receiving yards

Use the same pass-volume/pass-defense directions as other receiving markets. For pressure, the predeclared RB-checkdown direction is higher defensive pressure mismatch -> favorable receiving opportunity. This tests transmission residual only; it does not alter the existing rule.

### RB rush+receiving yards

Test the run-defense and pass-defense primitives separately plus pace/plays. Do not force a single PROE direction onto the combined market because pass/rush substitution makes that sign structurally ambiguous.

### QB pass yards control

Use the receiving/pass-volume directions above against the M89/M90 football-only residual. This is control evidence only; M56/M83 remain closed.

## 6. Feature scaling

For each target week, transform each matchup variable cross-sectionally at team/opponent grain using only the already-known pregame feature values for that week:

`weakness_z = oriented(feature - week_mean) / week_std`

This is continuous and leakage-safe. It is not a threshold search.

Rows with zero/undefined cross-sectional variance are unavailable for that feature/week.

## 7. Phase B statistic and support

For each predeclared market × feature × season cell:

- target: `actual - existing_football_projection`
- predictor: pregame `weakness_z`
- statistic: Spearman rank correlation
- minimum support: 200 player-game rows and 50 distinct games

Clustered uncertainty is required in **both** dimensions:

1. game-cluster bootstrap
2. player-cluster bootstrap

Use 5,000 bootstrap replicates with fixed seed `20261006`.

A season cell has directional clustered support only when:

- point Spearman rho > 0
- game-cluster 95% CI lower bound > 0
- player-cluster 95% CI lower bound > 0

A Phase B feature replicates only when the above holds in **both 2024 and 2025**.

No coefficient is fit.

## 8. Phase C opportunity construction

No target-game usage is allowed.

Use production-style strict-prior PlayerForm opportunity blending from official weekly history:

- prior-season share as the prior
- current-season games strictly before target week
- current-game weight = `current_games / (current_games + 4)`
- target share for receiving
- rush share for rushing

Where routes are unavailable from the authoritative historical weekly source, route-rate interaction is explicitly unavailable rather than invented.

For each eligible Phase C feature, build:

`interaction = strict_prior_opportunity * matchup_weakness_z`

No threshold, top-N, bellcow split, or outcome-selected role subset is allowed.

Predeclared interactions:

- RB rush share × run-defense weakness
- WR target share × WR receiving-defense weakness
- TE target share × TE receiving-defense / zone weakness
- RB target share × RB receiving-defense weakness

The same two-season + two-cluster replication rule applies.

## 9. Anti-retest gates

Even a replicated statistical signal is not automatically a candidate.

- M95A/M95B: generic RB rushing role × defensive rushing vulnerability / offense-defense matchup is already a closed or mixed family. Report replication, but do not create “bad run defense => boost RB X%.”
- M56/M83: generic QB matchup families remain closed. QB is control only.
- Rush Pool remains closed.
- Opportunity Authority remains closed.
- Bayes retuning remains closed.
- retired WR coverage penalty remains closed.
- sportsbook-conditioned football inputs remain prohibited.

Receiving position-YPT/coverage mechanisms are not ruled out by M95A/M95B, but they cannot cross the integration gate until same-live-source semantics are proven.

## 10. Candidate gate

A result may be labeled `INTEGRATION_CANDIDATE_ELIGIBLE` only if all are true:

1. expected sign in both 2024 and 2025;
2. game-cluster support in both seasons;
3. player-cluster support in both seasons;
4. residual target proves incrementality to the existing football projection;
5. same live semantics are proven;
6. the mechanism is not a closed prior family;
7. no sportsbook input was used;
8. no coefficient was fit.

If any mechanism clears this gate, freeze a **separate integration-candidate contract before scoring any implementation**.

Diagnostic PASS is not production promotion.
