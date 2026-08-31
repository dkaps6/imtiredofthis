# M95N — Conditional Player-Game Micro-Regime Audit Results

## Authoritative run

- Workflow: `M95N RB Micro-Regime Audit`
- Run: **`33435092627`**
- Job: **`99629424342`**
- Tested SHA: **`13f86f95e548a4675d2030340b7a9e2caf6e5172`**
- Branch: `research-rb-m95n-micro-regime-audit`
- Artifact: `migration-95n-rb-micro-regime-audit`
- Artifact ID: **`9774088423`**
- Artifact SHA256: **`b5044b3b55f0a2ec2fc9090e2f0e9580c77ca1b097dcb4c715581b12b0aa74b4`**
- Execution: success
- New model fit: `0`
- Feature search: `0`
- Coefficient search: `0`
- Sportsbook inputs: `0`
- Production change: `0`

## Question

M95M showed that M95K's stable-workhorse reranker is cross-season nonstationary. M95N tested whether the stable-workhorse population itself contains meaningful pregame micro-regimes: specifically, whether agreement between the frozen M95F current-context probability and historical feed/ceiling information is more stable than forcing one response function through games where those two evidence sources disagree.

## Fixed regime definition

For stable workhorses only, each target was split using pregame information only:

- `baseline_context_rank`: within-scope rank of frozen M95F probability;
- `feed_score`: fixed historical player/team feed-ceiling score;
- `aligned_high`: context high + history high;
- `context_only`: context high + history low;
- `history_only`: context low + history high;
- `aligned_low`: context low + history low.

No outcome was used to assign a regime.

## Main 20+ result — alignment is stable, disagreement flips

### 2023 Weeks 13-18

- aligned-high: **12 / 21 = 57.14%** actual 20+
- aligned-low: **3 / 18 = 16.67%**
- context-only: **7 / 16 = 43.75%**
- history-only: **2 / 18 = 11.11%**

### 2025 Weeks 13-18

- aligned-high: **13 / 30 = 43.33%** actual 20+
- aligned-low: **4 / 30 = 13.33%**
- context-only: **2 / 13 = 15.38%**
- history-only: **5 / 12 = 41.67%**

Two facts coexist:

1. **Agreement is directionally stable across seasons.** When both current context and historical feed were high, the 20+ event rate was much higher than when both were low in both 2023 and 2025.
2. **Disagreement is not stable.** In 2023, current context was the useful side of the disagreement (`43.75%` context-only versus `11.11%` history-only). In 2025 W13-18 the ordering reversed (`15.38%` context-only versus `41.67%` history-only).

The formal M95N disposition therefore set:

- `aligned_20plus_order_stable = 1`
- `discordant_20plus_preference_flips_by_season = 1`
- `micro_regime_dependence_supported = 1`
- interpretation: **`agreement_is_stable_signal_disagreement_requires_conditional_response`**

## Why M95K failed in 2023

The frozen M95K reranker behaved exactly like a universal historical-feed correction, which is dangerous when history and current context disagree.

In 2023 20+:

- `context_only` games had a high actual event rate (`43.75%`), but M95K reduced their probability by an average **`-0.03187`**;
- `history_only` games had a low actual event rate (`11.11%`), but M95K increased their probability by an average **`+0.03087`**.

That is the wrong direction for both discordant groups and explains a meaningful portion of the sealed M95L reversal.

In 2025 W13-18 the football relationship was different:

- `context_only` actual rate `15.38%` and M95K mean shift `-0.04223`;
- `history_only` actual rate `41.67%` and M95K mean shift `+0.18672`.

Thus M95K's historical-feed preference happened to be appropriate in that season but was inappropriate in 2023. The problem is not simply that feed history is useless; it is that the correct response when current context and historical feed conflict is not stationary under the current architecture.

## 25+ remains too sparse / unstable

Across the two same-window populations there were only **16 total 25+ events**. The aligned-high group was directionally higher than aligned-low in both seasons, but individual micro-regime AUCs and candidate changes are too unstable to justify a 25+ specialized architecture from M95N alone.

M95M's warning therefore remains: the current M95K 25+ conditional-ratio/mass-anchor architecture is not robust enough for promotion.

## Secondary discordant audit

M95N also split the two disagreement regimes by four precommitted pregame dimensions:

- projected volume;
- backfield concentration;
- opponent matchup/run-defense weakness;
- recent role momentum.

No single secondary dimension produced a clean, sufficiently sampled rule that resolved the 2023/2025 disagreement. Several cells were small and directional relationships themselves changed by season.

Therefore M95N does **not** justify hand-picking a specialized expert for individual games from the opened 2023 labels.

## Scientific interpretation

M95N supports the user's micro-level game-environment concern in a specific way:

- the model is already highly individualized in its inputs;
- however, a single shared stable-workhorse response function is too coarse when different evidence channels disagree;
- player-game archetypes have meaningful structure;
- the most repeatable structure found here is **agreement versus disagreement**, not a fully individualized player model;
- a separate model per player/game would still be statistically unsafe and likely overfit;
- the promising direction is a global backbone with precommitted conditional gating / mixture-of-experts behavior.

## Recommended next migration — M95O

**M95O — Agreement-Gated Stable-Workhorse 20+ Tail Candidate**

Precommit a conservative architecture before fitting:

- preserve M95F as the stable-workhorse baseline/backbone;
- use historical feed information only under a predeclared agreement condition between current pregame context and history;
- default discordant games back toward the global baseline rather than forcing the M95K rerank;
- focus first on **20+** because M95N gives much stronger evidence there;
- keep 25+ diagnostic/frozen until a stronger rare-event architecture exists;
- no sportsbook inputs;
- M94C central carries unchanged;
- no hand-picked player exceptions;
- any candidate derived from M95N must be evaluated on a new temporal protocol and cannot use opened 2023 W13-18 as pristine confirmation.

M95N is diagnostic evidence, not a production promotion.
