# M95H — RB Lead-Role Entitlement Results

## Authoritative run

- Workflow: `M95H RB Lead-Role Entitlement v2`
- Run: `33399681980`
- Job: `99512643135`
- Tested SHA: `61f509ea6ad99fc8827db3b4a88e48508063db29`
- Branch: `research-rb-m95h-lead-role-entitlement`
- Artifact: `migration-95h-rb-lead-role-entitlement-v2`
- Artifact ID: `9760755551`
- Artifact SHA256: `ec792214c84016d93600dfbef87583fdcd5ac7a3d627fef7f13a89556f95dee5`
- Artifact size: `155,724` bytes
- Execution conclusion: success
- Scientific disposition: `RETAIN_M95H_AS_DIAGNOSTIC_DO_NOT_PROMOTE`
- Production change: `0`
- M94C central carry mean preserved: `1`
- Sportsbook inputs: `0`

The first M95H workflow run (`33399534207`) failed before scientific evaluation because pandas 1.5 `Int64Index` does not expose `.ne()`. The v2 wrapper changed only that mechanical compatibility operation to `!=`; no feature family, model grid, development split, selection rule, validation gate, or football hypothesis changed.

## Scientific question

M95G showed that current-week availability and role information helps 20+ workload discrimination, but a generic vacated-role signal is true for several surviving RBs at once. M95H therefore tested the recipient-specific question:

> Which individual RB actually owns the upcoming backfield?

Pregame targets:

1. `P(player leads team RB carries)`
2. `P(player receives >=60% of team RB carries)`
3. `P(player receives >=70% of team RB carries)`

The targets are outcome labels only. No outcome information enters the pregame feature set.

## Protocol

- 2024 Weeks 5-12: temporal development fit.
- 2024 Weeks 13-18: architecture selection only.
- Architecture frozen after 2024 selection.
- Refit on eligible 2024 development rows.
- One untouched 2025 validation.
- M94C remains the central carry mean.
- No sportsbook data.
- No production code change.

Leakage-safe current-week sources were inherited from M95G v5:

- weekly rosters;
- injury reports and practice participation;
- 2024 week-tagged depth charts;
- 2025 depth snapshots mapped to the latest source date strictly before game day.

## Recipient-specific features

M95H replaced the broad vacancy idea with player-relative backfield features, including:

- current depth rank among available RBs;
- best available depth candidate;
- historical carry/share rank among currently available teammates;
- prior RB1 / RB2 identity and current availability;
- depth-chart promotion;
- competitor maximum and aggregate carry/share/target history;
- strong-competitor count and backfield scarcity;
- explicit successor-depth and successor-history candidate flags;
- vacancy interactions that apply to the candidate most likely to receive the role, rather than every surviving RB equally.

Candidate families were `history_only`, `entitlement_basic`, `entitlement_competition`, and `entitlement_interactions`.

## 2024 Weeks 13-18 architecture selection

### Lead-RB target

Selected: `entitlement_basic`, C=`0.03`

- AUC: `0.923337`
- Brier: `0.103228`
- log loss: `0.342016`
- team top-1 lead-RB accuracy: `83.05%`
- vacancy top-1 accuracy: `85.71%`
- development eligible: `1`

### >=60% RB carry-share target

Selected: `entitlement_competition`, C=`0.03`

- AUC: `0.921207`
- Brier: `0.100675`
- log loss: `0.318796`
- development eligible: `1`

### >=70% RB carry-share target

Selected: `entitlement_competition`, C=`0.03`

- AUC: `0.903591`
- Brier: `0.099288`
- log loss: `0.312236`
- development eligible: `1`

All three looked viable in the 2024 holdout and were frozen before 2025 was evaluated.

## Untouched 2025 validation

### 1. Exact lead-RB identity did not improve

Across 532 team-games with at least two modeled RBs:

- history-only top-1 accuracy: `82.8947%`
- M95H top-1 accuracy: `82.8947%`

Row-level lead-RB probability:

- history AUC: `0.922362`
- M95H AUC: `0.917416`
- history Brier: `0.099155`
- M95H Brier: `0.100566`
- history log loss: `0.335361`
- M95H log loss: `0.345037`

`lead_pass = 0`.

The exact successor problem therefore remains unsolved by the currently available roster/depth/injury features.

### Vacancy games expose the remaining difficulty

Among 45 vacancy team-games with at least two modeled RBs:

- history top-1 successor accuracy: `62.22%`
- M95H top-1 successor accuracy: `60.00%`

However, M95H assigned much more probability to the eventual true lead RB on average:

- history true-lead mean probability: `0.4854`
- M95H: `0.6088`

That extra confidence did not improve rank sufficiently. Vacancy row-level AUC declined from `0.77397` to `0.75979`, and Brier worsened from `0.20650` to `0.22369`.

This means the current data can detect that a role transition matters without consistently knowing which replacement receives the majority of the role.

### Incumbent-available and late-week lead identity are slightly better

Incumbent available, 455 team-games:

- history top-1: `86.37%`
- M95H: `86.81%`

Late weeks, 62 team-games:

- history top-1: `80.65%`
- M95H: `82.26%`

These are useful secondary signals but not sufficient to pass the lead-recipient gate.

## 2. >=60% carry share is mixed

All 2025 RB rows:

- history AUC: `0.907038`
- M95H AUC: `0.913600`
- history Brier: `0.101230`
- M95H Brier: `0.102402`
- history log loss: `0.333917`
- M95H log loss: `0.331363`

Ranking and log loss improved slightly, but the pre-specified Brier gate failed. `share60_pass = 0`.

Vacancy calibration was the main problem:

- actual >=60% share rate: `18.97%`
- history mean probability: `19.95%`
- M95H mean probability: `33.38%`

M95H became too optimistic when a backfield role opened.

By contrast, when the incumbent structure remained intact:

- AUC: `0.92789 -> 0.93818`
- Brier: `0.09090 -> 0.08764`

The role signal is more reliable in stable backfields than during sudden succession events.

## 3. >=70% carry share is the major M95H success

This target passed untouched 2025 validation.

All 2025 RB rows:

- actual >=70% share rate: `20.03%`
- history mean probability: `18.83%`
- M95H mean probability: `20.23%`
- AUC: **`0.903118 -> 0.919599`**
- Brier: **`0.096200 -> 0.090868`**
- log loss: **`0.298759 -> 0.280397`**

`share70_pass = 1`.

Incumbent-available:

- AUC: **`0.91386 -> 0.92756`**
- Brier: **`0.09614 -> 0.08974`**
- log loss: **`0.28722 -> 0.27202`**

Late weeks:

- AUC: **`0.91797 -> 0.92873`**
- Brier: **`0.08176 -> 0.07413`**
- log loss: **`0.24696 -> 0.22958`**

Vacancy games are especially revealing:

- actual >=70% rate: `7.76%`
- history mean probability: `10.47%`
- M95H mean probability: `18.33%`
- AUC: **`0.78401 -> 0.86501`**
- Brier: `0.07318 -> 0.08024`

So M95H sharply improves **ranking** of which vacancy candidates can dominate the backfield, but its vacancy probability level is too aggressive.

This is a strong, specific signal rather than a general entitlement-model win.

## Gates

- lead_pass: `0`
- share60_pass: `0`
- share70_pass: `1`
- incumbent_guard: `1`
- validation_pass: `0`

Formal disposition:

`RETAIN_M95H_AS_DIAGNOSTIC_DO_NOT_PROMOTE`

## Scientific conclusion

M95H does not justify promotion as a unified lead-role entitlement engine. It does, however, isolate a much stronger football signal than M95G's generic vacancy variable:

> Pregame role, depth, availability and competitor context meaningfully improve the probability ranking that a specific RB will command at least 70% of the RB rushing workload.

The signal is robust overall, in stable incumbent backfields, and in late-season games. In sudden vacancy situations it produces a large AUC improvement but overstates the absolute probability, which points to a calibration problem rather than absence of ranking information.

Exact successor identity remains materially harder than deep workload concentration.

## Recommended next migration — M95I

M95I should test **calibrated deep-concentration + workload-tail integration**.

Use only the validated part of M95H:

- `P(player receives >=70% of team RB carries)`

Do **not** carry forward the failed generic lead-RB probability as if it were validated.

Proposed architecture:

`M94C central/team rushing opportunity`

x `M95H deep-concentration entitlement probability`

x `M95F 20+/25+ workload-regime probability`

=> selective tail expansion for the backs who both own the backfield and are in a high-volume environment.

M95I should calibrate vacancy and incumbent-available populations separately because vacancy ranking is strong while its probability level is overconfident.

Required 2025 validation diagnostics:

- all RB / 0-5 / 6-10 / 11-14 / 15+ / 20+ / 25+ carry slices;
- carry MAE/bias/correlation;
- projected carry maximum and quantiles;
- count projected >=18 / >=20 / >=22 / >=25;
- 20+/25+ recall and precision;
- tail false positives;
- ordinary-game damage;
- stable-workhorse calibration;
- vacancy vs incumbent calibration;
- legacy all-player rushing guard;
- no sportsbook inputs;
- no production promotion unless the full pre-specified gate passes.
