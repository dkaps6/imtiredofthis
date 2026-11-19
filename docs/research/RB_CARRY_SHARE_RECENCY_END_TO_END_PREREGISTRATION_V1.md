# RB carry-share recency, end to end through the production carry chain: PREREGISTRATION V1

**Date:** 2026-11-19, frozen before any arm runs. **Research only. No production change.**

**Parent:** `RB_CARRY_SHARE_RECENCY_RESULTS_V1.md` (PASS at the PlayerForm-share level, a = 0.85).

## Why a second test is required: a validity gap in the parent

The parent scored PlayerForm's blended `rush_share`, `(1-w)·prior + w·current` with
w = g/(g+4). **The simulator does not consume that column.**

`bayesian_v2.build_bayesian_baseline` rebuilds each player's share from the **split**
`rush_share_prior` and `rush_share_current`:

    posterior = weighted mean of [position group mean (weight 3.0),
                                  rush_share_prior (weight min(prior_games, 6)),
                                  rush_share_current (weight current_games)]

`simulation_v2` reads `rules_rush_share`, which equals `bayes_rush_share`. It then keeps the
team's **top 5** shares (`_top_n_shares`), clips each to 0.95, and, **if the sum exceeds 0.95,
scales all of them by 0.95/sum** (`_allocate_counts`). The expected carries are team rush
attempts × the final share. `_sharpen_rb_rush_shares` is INERT (neutral gamma) and is omitted.

So the parent proved the input improves, not that it survives Bayes, top-5 and the cap. This
test checks whether it does.

**Prior-work check.**
- The carry-concentration, mass-budget and role-shrinkage studies measured this chain's level
  and shape defects.
- None changed the **current-season input's window**.
- The two coupled fixes designed in `PRISM_RB_RUSH_SHARE_MASS_BUDGET_V1.md` Addendum 3
  (allocative priors, evidence-precision weights) are **not** tested here. They need their
  own preregistration.

## Design

**Seasons:**
- 2025 holdout: priors from 2024.
- 2026 weeks 1-10 holdout: priors from 2025.
- Weeks 4+ only, so most players have at least 3 current games.
- `a = 0.85` is **frozen from the parent**. Nothing is fitted here.

**Universe, per team-week (an approximation of Ourlads plus availability):**
- `load_rosters_weekly` status ACT;
- QB, RB, WR, TE;
- **minus** players with an official Out or Doubtful report that week.

Disclosed: production's universe comes from the Ourlads depth chart. This approximation
matches its shape (about 2-3 QBs and 3-4 RBs per team), not its exact membership.

**Inputs per player, all strictly prior, joined on gsis IDs:**
- `rush_share_prior`: his previous-season carries / his team's carries in games he played,
  with `prior_games`.
- `rush_share_current`: season-to-date carries / team carries before this week, with
  `current_games`.
- Group mean: the `prior_games`-weighted mean of `rush_share_prior` within the position
  group, computed across that week's whole universe (as in `_weighted_group_mean`).

**Arms:**
- **C0:** the production formula above.
- **C1:** identical, except that when `current_games ≥ 3`, `rush_share_current` becomes
  `0.15·season_to_date + 0.85·last3`.

**Chain:** posterior, clipped to [0,1] → top 5 per team → clip 0.95 → scale to 0.95 if the sum
exceeds 0.95 → × the **actual** team rush attempts. Actual attempts are used so that only the
share differs between arms.

**Scored rows:** RBs in the universe who took an offensive snap.

## Metrics

- **Primary:** MAE of projected RB carries vs actual carries, C0 minus C1, with a 95%
  bootstrap CI clustered by team-game, for each holdout.
- **Reported, not gating:**
  - lead back: the team's top RB by C0 projection, MAE and bias;
  - shape: within-team normalized share MAE over the team's RBs (immune to the uniform cap);
  - team RB mass, mean projected vs actual.

## Kill and pass

- **K1:** the primary CI includes 0 in either holdout → **the input gain does not survive the
  production chain**. Recorded as a null for this chain as it stands.
- **K2:** lead-back MAE is worse in either holdout beyond its CI → null.
- **PASS:** positive, with the CI excluding 0 in both holdouts and no K2. A pass still makes it
  a candidate only. Promotion needs a production replay with the real universe and a check of
  its interaction with the INERT concentration transform.

## Prediction (scored after)

C1 still wins, but by less than at the input level. Bayes gives the current-season input only
`current_games` of weight, against 3 + up to 6 for the priors, so most of the recency signal
is diluted.

**Named risk:** the dilution is strong enough that the gain falls inside the noise, which
would be K1.
