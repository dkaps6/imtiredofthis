# Week-3 Specialist Non-Target MC Downstream Materiality V1 — Frozen Plan

**STATUS: FROZEN BEFORE SCORING. NO WEEK-3 OUTCOMES. NO PRODUCTION CHANGE.**

## Authority

Canonical repo: `dkaps6/imtiredofthis`

Parent authority:
- `SPECIALIST_NONTARGET_MC_PATH_DRIFT_CONFIRMED`
- parent run `36330399450`
- parent artifact `10935587149`

Frozen Week-3 paid source:
- Full Slate run `36293274478`
- source artifact `10923570170`
- digest `sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480`
- paid-run head `0982b62276303403e2ca58b16e6f4fc3e041f65d`

Production code identity required for the audited simulation/pricing modules is byte-identical between the paid-run head and current main.

No Week-3 outcomes may be loaded. No new OddsAPI request is allowed.

## Frozen question

The parent audit proved that an unrelated TE/WR specialist can alter protected finite-MC outputs even when the protected player's football entitlement is unchanged.

This audit asks:

> Is that path drift large enough to materially change downstream sportsbook probabilities, EV, preferred side, BET/PASS state, Best Snapshot wager identity, or board ranking — and is any observed movement larger than ordinary 25,000-draw Monte Carlo resampling noise under the same football inputs?

## Reconstruction contract

Use only the preserved paid artifact and unchanged production code.

Reconstruct the three exact entitlement stages:
1. M38 explicit baseline;
2. TE-R5P-only;
3. final WR-R15 state.

For each stage:
- run the exact 25,000-draw production simulator with production seed 42;
- apply the exact mean-neutral QB C2 selector to that stage before pass-yard pricing;
- use the preserved sportsbook lines/odds strictly downstream.

The reconstructed final WR-R15 state must reproduce the preserved paid board on supported markets before the materiality result is interpreted.

## Protected cohorts

Protection membership is exactly the parent V1 numeric contract:
- TE-stage protected iff TE entitlement delta <= 1e-12;
- WR-stage protected iff WR entitlement delta <= 1e-12.

No sportsbook field, outcome, position carveout, or postgame information may define protection.

## Supported priced markets

Primary priced audit:
- pass_yards
- rush_yards
- rec_yards
- receptions

Excluded from this V1:
- anytime_td: execution-capable but not dedicated-science certified;
- rush_rec_yards: production-active RB Rush+Receiving Conservation V2 has separate pathwise semantics and is not to be silently approximated here;
- rush_att: no Week-3 sportsbook rows in the paid board.

The exclusions do not weaken the parent MC-drift finding; they bound this downstream pricing audit to markets whose paid pricing path can be exactly reconstructed.

## Two frozen materiality surfaces

### A. SHAPE_ONLY_FIXED_FINAL_MEAN

For every stage, use that stage's exact simulated draw shape but align it to the preserved paid final football mean for the row.

This isolates finite-MC path/distribution-shape movement while removing mean-authority movement.

### B. FULL_DOWNSTREAM_PROPAGATION

Allow each stage's MC mean to flow through the exact frozen downstream mean authority:
- exact ensemble weights with fixed ML/State;
- Week-3 RB rush_yards remains generic calibrated ensemble;
- QB pass_yards runs exact M89/M90 synthesis using the stage-specific MC and ensemble values;
- receptions uses production Discrete Count Mean Alignment V1 after mean alignment;
- rec_yards/rush_yards remain continuous.

No coefficients are fit.

## Exact pricing decision metrics

For each specialist comparison (M38 -> TE-R5P and TE-R5P -> WR-R15), on protected sportsbook-priced rows report:

Side-row / quote level:
- absolute fair-probability delta: mean, median, p90, p95, p99, max;
- absolute expected-ROI delta: mean, median, p90, p95, p99, max;
- preferred-side flips at a concrete book+line;
- concrete quote HAS-EDGE/PASS flips where best side EV crosses zero.

Best Snapshot player-market level:
- BET/PASS flips;
- selected side flips;
- selected book/line/odds identity changes;
- absolute best-EV delta;
- Spearman rank correlation of best EV among common player-markets;
- top-10 and top-25 membership turnover.

Rank is descriptive only. No confidence threshold may be fitted.

## Ordinary finite-MC resampling benchmark

Use the exact final WR-R15 football state and exact downstream pricing path, but rerun the production simulator with these twelve preregistered alternative seeds:

`1042, 2042, 3042, 4042, 5042, 6042, 7042, 8042, 9042, 10042, 11042, 12042`

Each alternative seed is compared with production seed 42 on the same protected cohorts and same sportsbook board.

This benchmark changes no football input, coefficient, specialist entitlement, sportsbook line, or market availability. It measures ordinary 25,000-draw sampling instability of the current production architecture.

## Integrity gates

All must pass:

1. source artifact ID/name/digest/expiration match;
2. no Week-3 outcomes loaded;
3. no odds refetch;
4. target/TE/WR trace uniqueness and parent protection reconstruction pass;
5. reconstructed stage simulation means match the preserved parent stage-delta means to <= 1e-9;
6. final reconstructed production target means match paid `model_proj` to <= 1e-8 on supported rows;
7. final reconstructed fair probabilities match paid `fair_prob` to <= 1/25000 + 1e-12;
8. production code hashes for simulation/C2/pricing mean-authority modules match the paid head;
9. no production files are mutated outside research output;
10. all result rows are outcome-free.

Any failure => `SPECIALIST_MC_DOWNSTREAM_MATERIALITY_INTEGRITY_FAILURE`.

## Frozen disposition logic

After integrity passes:

### `SPECIALIST_MC_DOWNSTREAM_NOT_MATERIAL`
if both specialist comparisons have:
- zero protected Best Snapshot BET/PASS flips;
- zero protected selected-wager identity changes;
- zero protected concrete-quote preferred-side flips;
and their probability/EV movement summary is inside the ordinary-resampling envelope.

### `SPECIALIST_MC_MATERIAL_BUT_WITHIN_ORDINARY_RESAMPLING_NOISE`
if a specialist comparison causes at least one protected downstream decision/rank change, but every primary board-level movement metric is <= the maximum observed across the twelve ordinary-resampling comparisons.

Interpretation: the specialist exposes real decision instability, but RNG isolation alone is not justified because ordinary finite-MC resampling is at least as unstable.

### `SPECIALIST_MC_MATERIALITY_EXCEEDS_ORDINARY_RESAMPLING_NOISE`
if:
- at least one protected Best Snapshot BET/PASS or selected-wager identity change occurs; AND
- at least one primary board-level movement metric exceeds the maximum across all twelve ordinary-resampling comparisons.

Only this disposition can authorize designing a separate mechanical RNG-isolation candidate. It still does not authorize implementation in this audit.

## No-rescue / no-tuning contract

This audit may not:
- use Week-3 outcomes;
- change any football entitlement;
- change seeds in production;
- increase production iterations;
- split RNG streams;
- add common-random-number routing;
- cache/splice arrays;
- change TE-R5P / WR-R15 / M38;
- change QB C2 or M89/M90;
- alter RB Rush+Receiving V2;
- fit a confidence threshold;
- change sportsbook selection rules.

If ordinary resampling itself produces material BET/PASS/rank instability, that is a separate systems finding. It may motivate a separately frozen convergence/decision-stability study, not an immediate repair.

## Required result artifacts

- `result.json`
- `specialist_summary.csv`
- `specialist_detail.csv`
- `resampling_summary.csv`
- `integrity.json`
- `top_material_moves.csv`

Production changed = false.

## Pre-scoring implementation-fidelity amendment — QB C2 replay

The paid artifact itself is the authority for the Week-3 QB C2 selector population. It preserves:
- `data/qb_c2_production_starter_audit.csv` with exactly 30 slate teams;
- `data/qb_c2_production_integration_audit.csv` with exactly 30 primary QBs;
- `selected_qb_rows=30` and all 30 paid-slate primary QBs on `C2_SELECTED`.

For this audit, replay those exact frozen 30 starter/selector decisions and the unchanged lower-level C2 generator (`apply_c2`) for every entitlement stage. Do not re-resolve a different 32-team calendar universe after the fact. The final WR-R15 replay must reproduce the frozen paid C2 mean/SD/quantile audit before pass-yard materiality is accepted.

This amendment changes no scoring rule and is frozen before any materiality result is computed.


## Pre-scoring implementation-fidelity amendment — provider event/player lookup aliases

The certified production wrapper simulates on sportsbook-independent canonical game IDs and suffix-safe player keys, then installs provider event IDs and provider player keys only after simulation for exact paid-offer lookup.

This audit must replay both post-simulation lookup layers exactly:
- canonical schedule game -> frozen provider event ID;
- suffix-safe canonical player key -> preserved paid provider player_clean_key.

Protection remains defined only on the canonical parent entitlement trace before these aliases. For downstream paid-board comparisons, protected canonical identities are translated through the exact deterministic production lookup alias contract. No sportsbook line/odds or outcome defines protection.

This amendment is implementation fidelity only and is frozen before any materiality result is computed.
