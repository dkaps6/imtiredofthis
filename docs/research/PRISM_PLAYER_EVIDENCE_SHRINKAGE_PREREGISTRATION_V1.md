# PRISM — player-evidence shrinkage: PREREGISTRATION (V1)

**Frozen before any candidate was scored.** Nothing in this document was written after
seeing a result.

Target: `scripts/modeling/bayesian_v2._posterior_mean`, the promoted empirical-Bayes layer.
Root-cause evidence: `PRISM_RB_RUSH_SHARE_MASS_BUDGET_V1.md` addendum 3.

## 0. The defect, already measured

```python
weights = [GROUP_STRENGTH[metric]]                              # 3.0 — CONSTANT, every player
values  = [group_mean]                                          # a MARGINAL position prior
if prior_games > 0:
    weights.append(min(prior_games, PRIOR_PLAYER_CAP[metric]))   # own history, CAPPED at 6
if current_games > 0:
    weights.append(current_games)
posterior = np.average(values, weights=weights)
```

Two independent defects:

**(a) The prior is MARGINAL, not ALLOCATIVE.** `DEFAULTS["rush_share"]["RB"] = 0.32` is a
correct statement about *a* featured back and incoherent as a team allocation. Applied to
3.32 modeled backs per team the priors alone imply a team rush-share sum of **1.5414**
(28/28 teams over 1.0; RB block alone 1.063). A coherent budget needs ≈**0.226** at that
depth.

**(b) The player's own weight is a COUNT, capped.** `min(prior_games, 6)` credits a
17-game veteran with 6 games, so the generic position prior is **33.2% mean / 25.0% median
/ 20.0% floor** of a `rush_share` projection and **35.0% / 25.0% / 20.0%** of `tgt_share`.
**No player, however established, can get it below 20%**, and the only thing that varies
between players is how many games they played — never how *informative* those games are.

## 1. Hypothesis

**H1.** Correcting both — an **allocative** shrinkage target and an **evidence-precision**
player weight — improves *within-team relative share* accuracy for named players.

**H0.** Neither correction improves within-team share accuracy, or any gain is explained by
simply applying less shrinkage uniformly.

**Why within-team RELATIVE share is the right target, and why this does not wait on the
roster or budget work:** `_allocate_counts` applies a **uniform multiplicative** haircut
(mean 0.7574), which **preserves relative shares**. A shape improvement is therefore
measurable straight through the known-broken team budget. The budget and roster-existence
defects remain real and separately sequenced; they do not gate this.

## 2. Candidate set — frozen, no additions after scoring

| id | shrinkage target | player weight | free params |
|---|---|---|---|
| **C0** | marginal (incumbent) | `min(games, cap)` | 0 (baseline) |
| **C1** | **allocative** (depth-aware) | `min(games, cap)` | prior table only |
| **C2** | marginal | **evidence precision**, no cap | 0 beyond group variance |
| **C3** | **allocative** | **evidence precision**, no cap | both |
| **C4** | marginal | `min(games, cap)`, **`GROUP_STRENGTH` lowered to 1.0** | 1 |

**C4 is the decisive naive control and the reason this design can fail honestly.** It
applies *less shrinkage to everyone, uniformly* — no player-specificity at all. If C4
matches C3, then the finding is "there was simply too much shrinkage", **not** "shrinkage
must be player-specific", and the project's stated hypothesis is **not** supported even
though the numbers improve. K2 below enforces that.

## 3. Functional forms — frozen

**Allocative prior (C1, C3).** The shrinkage target for the rank-*k* player among *K*
modeled players at his position group on his team, so that one team's priors sum to a valid
budget by construction. Fitted on **2013–2025 only**, ranked by a **pregame** trailing-share
proxy (never by the outcome), walk-forward: the table for season *S* uses `[2013, S−1]`.
Already measured for the RB top slot: **0.7058** at 2 backs, **0.6277** at 3, **0.5625** at 4.

**Evidence-precision weight (C2, C3).**

```
w_player = n / (1 + n * var_player / var_group)
```

where `n` is the player's game count, `var_player` the variance of his own per-game
observations of the metric, `var_group` the position-group variance. As `var_player → 0`
the weight → `n` (full credit for a stable role); as it grows the weight shrinks. **No
cap.** This is standard empirical Bayes — shrinkage ∝ noise/(noise+signal), not ∝ 1/n — and
it is player-specific **by construction**: two players with 17 games each get different
weights if one's role is stable and the other's is erratic.

`var_group` is estimated on strictly prior seasons. Variance estimates are regularized by
the group variance for players with `n < 3`; that floor is a numerical guard, not a fit.

## 4. Data and train/test separation — frozen

The historical `player_form_consensus` frames production used do not exist on disk, so
candidates are run through the **real `_posterior_mean`** on a **historical consensus proxy**
built from nflverse weekly stats: per `(season, week, team)`, each player's prior-season
share and current-season-to-date share, exactly the inputs the production signature takes.

**Disclosed limitation:** the proxy is not production's actual PlayerForm output. Absolute
levels from this study therefore do **not** transfer. **The candidate comparison remains
valid**, because every candidate sees identical inputs — only the shrinkage target and
weight differ. No absolute claim about production's accuracy will be made from it.

**Walk-forward, rolling origin:** everything used to project season *S* week *w* comes from
`[2013, S−1]` plus `[week 1, w−1]` of season *S*. 2026 supplies no fitting input of any kind
and is reserved as the prospective check.

## 5. Metrics — frozen

**Primary 1 — within-team relative share accuracy.** Per team-week, normalize the modeled
share vector within the position group and compare to the actual normalized share vector.
Reported as a **distribution across players**, plus the **top-1 share gap** against the
football target. Immune to the uniform haircut.

**Primary 2 — heterogeneity by player evidence, which is the actual hypothesis.** Does the
improvement differ by the player's own evidence quality (game count × stability)? Stratified
by evidence quartile. **If the gain is flat across strata, the result is C4's, not C3's**,
and H1 is not supported however good the headline looks.

**Guardrails, reported and able to block:** team share-sum coherence (should move 1.27
toward 1.0); aggregate MAE; **committee-back harm** — backs in genuinely split
backfields must not be degraded to buy lead-back accuracy.

**Averaged MAE is NOT a primary metric** and no claim will rest on it, per the standing
instruction that a position average cannot settle an individual-player question.

**Uncertainty:** paired bootstrap (4,000 draws, seed 42) on per-team-week differences versus
C0, since all candidates are scored on identical rows.

## 6. Promotion gates — frozen

- **G1** C3 beats C0 on Primary 1 out of sample, bootstrap CI excluding zero;
- **G2** the gain is **concentrated in high-evidence players** (Primary 2) — the
  player-specificity claim must be visible, not assumed;
- **G3** **C4 does not match C3.** If uniform less-shrinkage explains the gain, promote
  nothing specific;
- **G4** team share-sum coherence improves or is unchanged;
- **G5** committee backs not harmed beyond their preregistered interval;
- **G6** aggregate MAE not materially worse;
- **G7** no sportsbook variable at any stage (AGENTS.md §3).

## 7. Kill conditions — frozen

- **K1** C3 does not beat C0 on Primary 1 → bank as a null; the incumbent stands.
- **K2** C4 matches C3 → the finding is "too much shrinkage", **not** player-specificity.
  Report it as such, promote nothing specific, and say plainly that H1 failed.
- **K3** the gain is flat across evidence strata → no evidence of player-specificity; same
  disposition as K2.
- **K4** team share-sum coherence worsens → the allocative prior is mis-specified.
- **K5** committee backs materially harmed → rejected regardless of headline gain.
- **K6** required inputs unavailable → fail closed; do not substitute a hand-chosen weight.

## 8. Scope

Ships **INERT** behind dual authority if it passes, exactly as the MC dispersion and RB
concentration transforms did. `data/model_ensemble_weights.csv`, the ensemble, the
simulator, the 0.95 cap and the roster path are all **untouched** by this study. No
sportsbook variable enters any prior, weight, fit or metric. Betting remains **CLOSED**. No
paid odds fetch is authorized or performed.
