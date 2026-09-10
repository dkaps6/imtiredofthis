# QB Opportunity Chain Decomposition V1 — Result

## Disposition

`TEAM_PASS_OPPORTUNITY_PRIMARY_DIAGNOSTIC`

This is a diagnostic routing result only. It does **not** promote or alter any production model.

## Canonical lineage

- Branch: `research-qb-opportunity-chain-decomposition-v1`
- Frozen-plan commit: `78201e401bf714bb1ede6ae39537afdc118a1773`
- Evaluator commit: `123463811f3f873dfe3738e0bb694da44554412e`
- Tested/workflow head: `8916d1d4537c6d6f9470ee4d97a94fdde816eb6d`
- GitHub Actions run: `34523313743`
- Job: `103025958402`
- Artifact: `10170531084` (`qb-opportunity-chain-decomposition-v1`)
- Artifact digest: `sha256:75cd32198caf7d9cbf193d5769e5762acce53352f2b2626cd92a9b117e75dafd`
- M89 immutable source run: `33331073376`
- Shared QB/WR immutable source run: `34066549394`

## Integrity

All frozen integrity gates passed.

- M89 QB rows: `884/884`
- Shared WR target-mass rows: `440/440`
- Shared WR reception-mass rows: `884/884`
- sportsbook inputs used: `0`
- model fitting: `false`
- production changes: `false`
- predicted factor-product vs M89 expected attempts max abs gap: `1.7763568394002505e-14`
- M89 expected attempts vs frozen predicted attempts max abs gap: `3.552713678800501e-15`
- realized factor-product vs actual attempts max abs gap: `2.500001983207767e-09`
- three-factor Shapley attempt identity max abs gap: `2.5000055359214457e-09`
- opportunity-chain yards vs frozen attempt component max abs gap: `1.9964062403232674e-08`
- full final M89/M90 residual identity max abs gap: `1.6000002744931408e-07`

The original shared-pass-volume correlations were reproduced from the immutable source cohorts:

- 2025 QB attempt residual vs WR target-mass residual: Pearson `0.6973048915041207`; Spearman `0.670648420892482`; same-sign `0.7318181818181818`.
- 2024-2025 QB attempt residual vs WR reception-mass residual: Pearson `0.5238926874684943`; Spearman `0.4994040245907647`; same-sign `0.7092760180995475`.

## Frozen opportunity-chain result

The M89 attempt construction was decomposed as:

`team pass opportunity/dropbacks × attempt conversion × primary-QB share`.

### Pooled 2024-2025, all 884 QB-games

| Component | Mean abs attempt contribution | Share of absolute chain mass | Sign agreement with total attempt residual | Dominant-row rate | P90 abs contribution |
|---|---:|---:|---:|---:|---:|
| TEAM_PASS_OPPORTUNITY | 6.223448 | 68.2923% | 88.4615% | 76.2443% | 12.767065 |
| ATTEMPT_CONVERSION | 1.704205 | 18.7009% | 62.8959% | 14.8190% | 3.505030 |
| QB_SHARE | 1.185307 | 13.0068% | 39.0271% | 8.9367% | 3.687803 |

The primary component was stable by season:

- 2024 TEAM_PASS_OPPORTUNITY mean-absolute contribution: `6.359649`
- 2025 TEAM_PASS_OPPORTUNITY mean-absolute contribution: `6.086008`
- 2024 ATTEMPT_CONVERSION: `1.706203`
- 2025 ATTEMPT_CONVERSION: `1.702189`
- 2024 QB_SHARE: `1.244542`
- 2025 QB_SHARE: `1.125534`

### Existing ATTEMPTS_DOMINANT games

Among the previously frozen `ATTEMPTS_DOMINANT` cohort (`n=420`):

- TEAM_PASS_OPPORTUNITY mean-absolute contribution: `8.583312`
- absolute chain-mass share: `73.7574%`
- sign agreement with total attempt residual: `97.1429%`
- dominant-row rate: `85.9524%`
- p90 absolute contribution: `16.593540` attempts

### Large attempt misses

For absolute attempt misses of at least 8 (`n=313`):

- TEAM_PASS_OPPORTUNITY mean-absolute contribution: `10.805305`
- chain-mass share: `75.9464%`
- sign agreement: `99.3610%`
- dominant-row rate: `90.7348%`

For absolute attempt misses of at least 10 (`n=226`):

- TEAM_PASS_OPPORTUNITY mean-absolute contribution: `12.241868`
- chain-mass share: `77.0721%`
- sign agreement: `100.0000%`
- dominant-row rate: `92.0354%`

ATTEMPT_CONVERSION was dominant in `0.0%` of the 10+ miss cohort. QB_SHARE was dominant in `7.9646%`.

## Shared QB/receiver attribution

### 2025 WR target-mass cohort (`n=440`)

| QB opportunity component | Pearson vs WR target residual | Spearman | Same-sign | Signed component Q4-Q1 WR residual gap |
|---|---:|---:|---:|---:|
| TOTAL_QB_ATTEMPT_RESIDUAL | 0.697305 | 0.670648 | 73.1818% | 9.203058 |
| TEAM_PASS_OPPORTUNITY | 0.687155 | 0.657279 | 74.5455% | 9.440816 |
| ATTEMPT_CONVERSION | 0.291846 | 0.277449 | 59.0909% | 3.784374 |
| QB_SHARE | 0.003164 | 0.000117 | 37.0455% | 0.024245 |

TEAM_PASS_OPPORTUNITY therefore preserves almost all of the already-established QB-attempt/WR-target coupling.

### 2024-2025 WR reception-mass replication (`n=884`)

Pooled Spearman correlations:

- total QB attempt residual: `0.499404`
- TEAM_PASS_OPPORTUNITY: `0.500323`
- ATTEMPT_CONVERSION: `0.221143`
- QB_SHARE: `-0.008786`

TEAM_PASS_OPPORTUNITY replication is stable in both seasons:

- 2024 Spearman vs WR reception residual: `0.484618`
- 2025 Spearman vs WR reception residual: `0.519234`

## Frozen routing-gate decision

TEAM_PASS_OPPORTUNITY passed all four preregistered PRIMARY gates:

1. largest pooled mean-absolute attempt contribution: **PASS**
2. season-stability gate: **PASS**
3. 2025 WR-target absolute Spearman >= 0.30: **PASS** (`0.657279`)
4. WR-target absolute Spearman lead >= 0.10 over each other opportunity component: **PASS**

ATTEMPT_CONVERSION and QB_SHARE failed the PRIMARY routing gates.

Therefore the only authorized disposition is:

`TEAM_PASS_OPPORTUNITY_PRIMARY_DIAGNOSTIC`

## Scientific meaning

The remaining M89/M90 volume problem is not primarily a primary-QB-share problem and is not primarily an official-attempt conversion / sack-scramble problem. It is upstream: the model is missing the realized number of team pass opportunities/dropbacks.

This is also the same upstream state that largely drives the independent WR opportunity miss. The result strengthens the project architecture requirement that QB and receiver opportunity should share a team pass-opportunity state rather than behave as independent volume systems.

This does **not** establish that the team pass-opportunity miss is predictably correctable before kickoff. It establishes where the next source/predictability audit is allowed to focus.

## Immediate next authorized step

Perform a strict-prior, anti-reinvention source/predictability audit for **TEAM_PASS_OPPORTUNITY** only.

Do not:

- restart generic QB mean residual modeling;
- retest the QB-R1 player/context router;
- recycle M64/M65 possession/dropback/state-occupancy inputs as if new;
- recycle M67-M69 opening/playcaller/game-script families;
- use sportsbook/game-market information upstream;
- use target-game PBP outcomes as pregame features.

A subsequent predictive experiment may be frozen only if the source audit finds genuinely new, pre-kickoff football information or a demonstrably untested architecture seam specific to team pass opportunity.
