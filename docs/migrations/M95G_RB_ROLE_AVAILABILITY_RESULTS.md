# M95G — RB Pregame Role Transition / Availability Results

## Authoritative run

- Workflow: `M95G RB Role Availability v5`
- Run: `33396339232`
- Job: `99501648190`
- Tested SHA: `39d163048d94f733596098e479334cbf7613f87f`
- Branch: `research-rb-m95g-role-availability`
- Artifact: `migration-95g-rb-role-availability-v5`
- Artifact ID: `9759476538`
- Artifact SHA256: `9ecf458c782686cc265a1f2c763f70c02fd2c77ab3f9e7a59321e13f2d78e08b`
- Artifact size: `1,995,391` bytes
- Execution conclusion: success
- Scientific disposition: `RETAIN_M95G_AS_DIAGNOSTIC_DO_NOT_PROMOTE`
- Production change: `0`
- M94C central carry mean preserved: `1`

## Scientific question

M95F showed that the high-workload regime is a real pregame signal but still over-triggered the highest-risk workhorse population. M95G tested whether strictly pregame weekly roster, injury-report, depth-chart, competitor-availability and role-transition information could improve the mapping from historical workload profile to the specific upcoming game's 20+/25+ carry state.

M95G did **not** change the M94C carry mean. It froze the M95F raw tail scorer and tested a second-stage role/availability calibration layer.

Protocol:

1. fit candidate role/availability calibrators on temporal 2024 OOF data through Week 12;
2. select architecture and operating threshold on 2024 Weeks 13-18 only;
3. freeze the selected architecture;
4. refit on eligible 2024 temporal OOF rows;
5. evaluate once on untouched 2025.

No sportsbook input was used.

## Data/source audit

Leakage-safe sources were successfully recovered for both seasons:

- 2024 weekly roster rows: `46,579`
- 2025 weekly roster rows: `46,849`
- 2024 raw depth-chart rows: `37,312`
- 2024 normalized week-tagged RB depth rows: `2,230`
- 2025 raw depth-chart rows: `554,215`
- 2025 normalized date-based depth rows assigned to the latest snapshot **strictly before game day**: `2,585`
- historical injury rows: `12,283`

RB validation-row roster presence:

- 2024 holdout: `97.286%`
- 2025 validation: `94.544%`

The injury-match rate is much lower (`15.45%` in 2024, `13.78%` in 2025) because the injury feed naturally contains only players who appeared on an injury report; it is not an overall source-completeness rate.

The nflverse depth source changed after 2024. The 2024 source is explicitly week-tagged; the 2025 source is date-based. M95G v5 maps each 2025 team-game to the latest depth-chart snapshot strictly before the scheduled game date and excludes same-day snapshots because trustworthy pregame timestamps are not available.

## Selected architecture

Both targets selected the same development architecture:

- candidate: `role_availability_interactions_lo_C0.05`
- logistic regularization C: `0.05`

Frozen operating thresholds:

- 20+ carries: `0.25`
- 25+ carries: `0.075`

### 2024 Weeks 13-18 holdout

20+ carries:

- M95F AUC: `0.879263`
- M95G AUC: `0.878239`
- M95F Brier: `0.065236`
- M95G Brier: `0.061233`
- selected operating point: precision `48.15%`, recall `57.78%`, 28 false positives, 1.20x as many flags as actual positives

25+ carries:

- M95F AUC: `0.911001`
- M95G AUC: `0.911255`
- M95F Brier: `0.031213`
- M95G Brier: `0.031353`
- selected operating point: precision `18.92%`, recall `41.18%`, 30 false positives, 2.18x as many flags as actual positives

The candidate therefore looked viable in the development holdout, with a strong 20+ calibration improvement and essentially unchanged 25+ ranking.

## Untouched 2025 validation

### 20+ carries — real incremental value

M95F:

- actual base rate: `0.070352`
- mean probability: `0.100862`
- AUC: `0.846474`
- Brier: `0.062636`
- log loss: `0.208788`

M95G:

- mean probability: `0.098644`
- AUC: **`0.859727`**
- Brier: **`0.061134`**
- log loss: **`0.201028`**

At the frozen `0.25` threshold:

- true positives: `55`
- false positives: `156`
- false negatives: `43`
- precision: `26.07%`
- recall: `56.12%`
- 211 flags vs 98 actual 20+ games

Thus the role/depth/availability layer added meaningful **20+ workload discrimination**. The AUC improvement is about +0.0133, and Brier/log-loss also improved. `target20_pass = 1`.

## 25+ carries — does not generalize

M95F:

- actual base rate: `0.017229`
- mean probability: `0.030643`
- AUC: `0.844321`
- Brier: `0.017985`
- log loss: `0.078526`

M95G:

- mean probability: `0.029268`
- AUC: **`0.826303`**
- Brier: `0.017957`
- log loss: `0.079765`

At the frozen `0.075` threshold:

- true positives: `14`
- false positives: `208`
- false negatives: `10`
- precision: `6.31%`
- recall: `58.33%`
- 222 flags vs 24 actual 25+ games

The probability mean/Brier are superficially close, but ranking degraded by about 0.018 AUC and the operating point became broader rather than narrower. `target25_pass = 0`.

## Stable-workhorse problem remains

2025 stable workhorses, 25+ carries:

- actual rate: `4.33%`
- M95F predicted rate: `11.12%`
- M95G predicted rate: `10.74%`

The calibration gap only shrank from `+0.06789` to `+0.06405`, about a 5.7% reduction. The pre-specified gate required at least a 20% reduction, so `stable_workhorse_improvement_pass = 0`.

For 20+ carries, full source-complete M95G actually became more optimistic on stable workhorses:

- actual: `20.87%`
- M95F: `29.53%`
- M95G: `32.46%`

So current-week depth/availability information improves global 20+ ranking without solving the high-risk workhorse calibration problem.

## Critical structural finding: a vacancy is not a successor

The most important M95G lesson is that the current player-level vacancy representation is too broad.

A feature such as `vacated_lead_role` means the prior lead RB is unavailable and the current row is not that prior lead RB. That condition is therefore true for **multiple remaining RBs on the team**, even though only one of them may actually inherit the workload.

2025 examples of these role-transition slices show why a generic vacancy cannot be treated as automatic bellcow eligibility:

- prior top-1 unavailable, 20+ actual rate: `2.59%`; M95G predicted `6.08%`
- vacated lead role, 20+ actual rate: `2.17%`; M95G predicted `5.31%`
- vacated lead role, 25+ actual rate: `1.09%`; M95G predicted `0.26%`

The sign and magnitude are inconsistent because the variable describes a **team opportunity vacancy**, not the identity of the recipient.

This explains why some sudden 25+ games remain difficult. A backup can have almost no trailing workload history yet become the true RB1 because of a very specific current-week depth/availability transition. Conversely, multiple active backups may all technically sit behind an unavailable incumbent, but only one is expected to inherit the lead role.

## Implementation history / non-scientific runs

Before v5, several runs exposed mechanical/source-contract issues. None changed the scientific model specification.

1. Initial M95G run failed before scientific results because `TEAM_KEYS = [season, week, team]` was unpacked in the wrong order while reconstructing prior-game leaders.
2. v2 fixed that mechanical key-order issue, then failed because the frozen M95F OOF artifact calls the raw score `raw_score`, while frozen holdout/validation traces use `raw_prob_20` / `raw_prob_25`.
3. v3 fixed those contracts and completed, but source audit showed zero usable 2025 depth rows because nflverse changed depth-chart schema after 2024.
4. v4 added date-based 2025 depth mapping but exposed a timezone-aware/naive date comparison issue.
5. v5 only normalized those source dates consistently and is the authoritative source-complete M95G result.

No candidate family, feature family, regularization grid, operating-threshold grid, selection rule, or 2025 validation gate was changed in response to those failed/superseded runs.

## Scientific conclusion

M95G provides **strong evidence that current-week role/depth/availability information contains incremental signal for identifying 20+ carry games**. It should be retained as a diagnostic component.

However, the same generic role/availability layer does not solve the rarer 25+ state. It loses 25+ ranking accuracy, does not sufficiently reduce stable-workhorse overprediction, and does not reliably identify the particular replacement RB who inherits a vacated workload.

Therefore M95G must not be promoted as one combined 20+/25+ production regime layer.

## Recommended next experiment: M95H recipient-specific lead-role entitlement

The next workload experiment should explicitly predict **which specific RB owns the upcoming backfield**, instead of attaching a team vacancy signal to every surviving RB.

Recommended pregame targets:

1. `P(player leads team RB carries)`
2. `P(player receives >=60% of team RB carries)`
3. `P(player receives >=70% of team RB carries)`

Recommended leakage-safe inputs:

- current depth rank among available RBs;
- weekly roster active/inactive status;
- all competing RB availability and injury status;
- prior-game and rolling carry share;
- recent snap share, where available;
- route/target/pass-block/third-down/two-minute usage if reliable;
- red-zone / inside-10 rushing role;
- prior top-1/top-2 identities and current availability;
- current player's depth movement/promotion;
- rookie/recent role-growth trend;
- newly signed/elevated replacement status;
- competitor scarcity weighted by competitors' prior usage, not merely competitor count;
- official historical game-day inactive data if an RB-safe version can be certified.

The desired architecture becomes:

`P(this player is the actual weekly lead-role recipient)`

× `P(team/game enters a high-rushing environment)`

× `P(20+/25+ workload | entitled)`

rather than:

`historical workhorse score + generic team vacancy`.

M94C should remain the central carry mean during this research.
