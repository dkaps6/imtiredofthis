# Shared Pass-State Coherence V1 — Read-Only Audit Result

Date: 2026-09-24

Status: **STRUCTURAL DIVERGENCE CONFIRMED**

This is a read-only architecture result. It is not a production qualification and it scored no target-game outcomes.

## Authority

- branch: `research-shared-pass-state-coherence-v1`
- audit head: `20b7fd1cd74c1438b249d8a3648963448f263dc7`
- workflow run: `36073919850`
- job: `107880970326`
- artifact: `10839371645`
- artifact digest: `sha256:d6dda84e5523dd4b353fcaab2a1e3a2cbecd2e33d71a1fc4c7e5934cb37b5a91`
- season/week: 2026 Week 3
- Monte Carlo draws: 5,000
- base seed: 42
- C2 seed: 5601
- sportsbook inputs used: **0**
- target-game outcomes used: **0**
- production changed: **false**
- C1 used: **false**
- C3 used: **false**

## Scope/integrity

- football players: **458**
- football teams: **32**
- games: **16**
- sportsbook rows used to define football universe: **0**
- C2 football QBs: **32**
- C2-selected QBs: **31**
- receiver player rows audited: **367**
- independently captured C2 shadow vs installed selected-C2 QB arrays max gap: **0.0**
- C2 pass/receiving identity max gap: **0.0**
- C2 shadow zero-reception + positive-yard rate max: **0.0**

The audit therefore measured the actual frozen C2 process, not an approximation.

## Decisive structural finding

Current production is materially split into two passing-game states.

For C2-selected teams:
1. the QB pass-yard distribution comes from the C2 completed-pass / receiving process;
2. WR/TE/RB receiving arrays remain the separate canonical production process;
3. those two receiver states are not close at the draw level.

### Team-level divergence

Across the 31 selected QB teams:

- median correlation: final C2 QB pass yards vs canonical modeled receiver total = **0.5575**
- median correlation: canonical modeled receiver total vs C2 shadow modeled receiver total = **0.5437**
- C2 QB vs C2 shadow total correlation = **1.0** by construction
- median absolute QB-minus-canonical-modeled-receiver mean gap = **11.55 yards**
- median C2 team scale = **0.9276**
- `|scale - 1| > 5%` in **61.3%** of selected teams
- `|scale - 1| > 10%` in **35.5%** of selected teams
- scale range = **0.8626 to 1.1966**
- median C2 residual share of QB passing mean = **5.07%**

Large current examples:
- MIA: QB mean exceeds canonical modeled receiver mean by **48.24 yd**, C2 scale **1.1966**
- JAX: canonical modeled receiver mean exceeds QB mean by **26.85 yd**, C2 scale **0.8691**
- TEN: gap **-23.80 yd**, scale **0.8626**
- CHI: gap **-23.18 yd**, scale **0.8800**
- CAR: gap **-22.92 yd**, scale **0.8699**

These are football-state mismatches, not sportsbook-line comparisons.

### Player-level divergence

Across 367 WR/TE/RB/FB player-team rows:

- median canonical-vs-C2-shadow receiver array correlation = **0.0399**
- mean correlation = **0.0474**
- median p90 absolute draw-level yard gap = **30.84 yd**
- mean absolute player mean gap = approximately **1.84 yd**
- canonical player median rate of **0 receptions but positive receiving yards** = **16.42% of draws**
- C2 shadow rate of 0 receptions but positive receiving yards = **0%**

By position:

| Position | Players | Median array corr | Mean abs mean gap | Median p90 draw gap |
|---|---:|---:|---:|---:|
| WR | 171 | 0.0444 | 2.14 yd | 38.66 yd |
| TE | 92 | 0.0527 | 1.82 yd | 31.63 yd |
| RB | 90 | 0.0256 | 1.37 yd | 18.77 yd |
| FB | 14 | 0.0227 | 1.27 yd | 19.13 yd |

The disagreement grows with receiver importance.

Entitlement quartiles:

| Quartile | Players | Median array corr | Mean abs mean gap | Median p90 draw gap |
|---|---:|---:|---:|---:|
| Q1 low | 92 | 0.0207 | 1.40 yd | 18.77 yd |
| Q2 | 92 | 0.0311 | 1.42 yd | 25.39 yd |
| Q3 | 91 | 0.0499 | 1.56 yd | 32.59 yd |
| Q4 high | 92 | 0.0898 | 2.97 yd | **49.53 yd** |

Examples among high-entitlement receivers:
- Jaxon Smith-Njigba: p90 canonical-vs-shadow draw gap **82.56 yd**
- Puka Nacua: **69.24 yd**
- CeeDee Lamb: **67.63 yd**
- Christian Watson: **66.91 yd**
- Nico Collins: **65.70 yd**
- Chris Olave: **64.92 yd**
- Tee Higgins: **64.57 yd**

These are not outcome errors. They quantify how differently the two current model states describe the same player's simulated game.

## Completed-pass semantic contradiction

The canonical receiver engine separately draws receptions and receiving yards:
- receptions are sampled from targets and catch probability;
- receiving yards are sampled around `targets * YPT`, independent of the sampled reception count.

Therefore the canonical joint state can contain **zero receptions and positive receiving yards** on the same draw.

The current audit confirms this occurs materially:
- median player rate: **16.42%**
- C2 completed-pass process: **0%**

This does not prove the C2 receiver distribution is more accurate. It proves the current production joint state is not physically coherent across receptions, receiving yards and selected-QB passing yards.

## Historical context

The old conservation integration was never scientifically rejected for receivers.

Run `34139757238`:
- QB rows: **884**
- QB CRPS: **40.3880 -> 39.1034** (+1.2846 yd improvement)
- 80% coverage: **58.82% -> 71.15%**
- QB mean unchanged
- receiver rows: **0**
- disposition: `MECHANICAL_OR_INTEGRITY_FAILURE`

Its receiver gates were never evaluated.

The older 2020-2025 conservation traces also showed substantial team rescaling:
- `|scale - 1| > 5%`: ~44.6% pooled
- `|scale - 1| > 10%`: ~16.9% pooled
- 2024: >10% in ~29.6%
- 2025: >10% in ~29.2%

The current 2026 Week-3 audit is even more structurally divergent:
- >5%: **61.3%**
- >10%: **35.5%**

## Disposition

`SHARED_PASS_STATE_COHERENCE_V1_STRUCTURAL_DIVERGENCE_CONFIRMED`

This result authorizes a separately frozen **one-pass-state integration candidate**.

It does **not** authorize production change.

C1/C3 remain closed. No QB mean retune, threshold search, player carveout, sportsbook-conditioned routing or outcome-driven rescue is authorized.
