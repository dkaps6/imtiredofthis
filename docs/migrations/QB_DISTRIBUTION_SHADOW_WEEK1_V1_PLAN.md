# QB Distribution Shadow Week-1 V1 — Frozen Deployment Plan

## Purpose

Validate the deployable Phase-J QB distribution selector inside the actual 2026 Week-1 Full Slate data path without changing production pricing.

This is a deployment/parity test, not a new scientific feature search and not a mean-model migration.

## Frozen authorities

- QB point mean: `QB_PASS_SYNTHESIS_V1` / M89-M90 authority.
- QB distribution-state selector: `QB_DISTRIBUTION_STATE_SELECTOR_V1` from Phase J.
- Candidate QB distribution: frozen C2 QB conservation distribution only.
- WR: M38 authority unchanged.
- TE: TE-R5P authority unchanged.
- RB rushing: RB-P3 Week-1 authority unchanged.
- Sportsbook information is downstream audit only and may not enter the selector, player-universe construction, starter selection, or football distribution construction.
- Roster membership/depth fallback: Ourlads current offensive depth-chart snapshot.
- Exact game-starter authority: a newer versioned official team announcement/depth chart may supersede an older Ourlads depth ordering for the exact season/week. Every override must record source type, source date, URL, reason, season, week, team, and player in `config/qb_starter_authority_v1.csv`.

### Week-1 authority correction recorded before result

Atlanta's Ourlads page was updated 2026-09-05 and still ordered Michael Penix Jr. ahead of Tua Tagovailoa. Atlanta officially named Tua Tagovailoa the Week-1 starter on 2026-09-07 and stated Michael Penix Jr. would be inactive. Therefore the official 2026-09-07 game-specific announcement supersedes the older Ourlads ordering for ATL Week 1. This correction is football-only and was frozen before observing any C2 shadow result.

## Exact deployment test

1. Dispatch `replay-paid-full-slate-artifact-v1.yml` on the exact branch SHA. It must reuse paid source run `34152868136`; OddsAPI may not be refetched or charged.
2. Free/current football information is not conceptually frozen by the no-credit rule. The paid sportsbook artifact is frozen only to prevent another paid request. Starter authority may use newer football-only evidence with explicit provenance as described above.
3. Do not modify production pricing outputs in this shadow migration.
4. Download the exact no-credit Full Slate replay artifact, including the sportsbook-independent 469-player football simulation universe and the downstream priced artifact.
5. Build exact Week-1 QB starter identity from the football universe. Use the versioned official starter authority when present; otherwise use Ourlads QB depth order as the fallback. Sportsbook rows are not consulted during this selection.
6. Compare the already-selected football starter to the downstream paid pass-yard identity only after football starter selection. Any unexplained mismatch blocks the shadow as an identity/freshness issue, not a scientific C2 failure.
7. Rebuild strict-prior Phase-J team environment context from completed historical PBP and the authoritative 2026 team-week map.
8. Reconstruct the canonical QB Monte Carlo from the sportsbook-independent full football universe using `simulate_with_states`, which must retain parity with canonical simulation when C2 is off.
9. Convert the reconstructed QB pass-opportunity distribution using the exact football-derived `qb_attempt_conversion` and `qb_pass_att_share` carried through production pricing.
10. Verify the reconstructed pre-synthesis MC mean equals the production `mc_proj`.
11. Anchor both the canonical shadow distribution and any selected C2 distribution to the exact production `model_proj` / `qb_synthesis_proj` mean.
12. Apply `QB_DISTRIBUTION_STATE_SELECTOR_V1` using only:
   - pass opportunity spot
   - pass efficiency spot
   - rush opportunity spot
   - rush efficiency spot
   - promoted predicted QB attempts
   - week
13. Expose the C2 QB array in the shadow sidecar only when the selector predicts a positive pass-attempt delta. Otherwise retain the canonical shadow distribution.
14. Never write the selected shadow distribution back into `outputs/props_priced_clean.csv` in this migration.

## Frozen gates

All must pass for `QB_DISTRIBUTION_FULL_ROSTER_SHADOW_PASS`:

1. Football-only starter authority covers all 32 teams and uses zero sportsbook inputs.
2. Any official game-specific starter authority is applied before downstream paid identity comparison.
3. Downstream pass-yard identity matches the already-selected football starter for every team being audited.
4. Real-slate reconstructed `mc_proj` parity: maximum absolute gap <= `1e-8`.
5. Canonical shadow mean anchor: maximum absolute gap to promoted M89/M90 mean <= `1e-8`.
6. Selected shadow mean anchor: maximum absolute gap to promoted M89/M90 mean <= `1e-8`.
7. Selector activates for at least one priced QB.
8. Every priced QB distribution is covered by the shadow audit.
9. Selector sportsbook inputs are exactly zero.
10. Production pricing is not modified by the shadow evaluator.

No gate may be relaxed after results.

## Outputs

- `config/qb_starter_authority_v1.csv`
- `data/qb_c2_shadow_primary_qb_audit.csv`
- `outputs/qb_distribution_shadow_v1.csv`
- `outputs/qb_distribution_shadow_v1_result.json`

The sidecar records player/team/opponent, football starter provenance, Phase-J state inputs, predicted attempt delta, selector decision, canonical and selected distribution moments/quantiles, exact mean/parity gaps, and downstream-only probability deltas versus the posted line when available.

## Interpretation

- PASS authorizes a subsequent explicit production-integration migration for the QB distribution selector. It does not itself modify production.
- A football starter/depth freshness conflict is mechanical/data-authority work, not a scientific C2 failure.
- FAIL stays failed when the scientific/parity gates are actually reached and missed. Repair only mechanical data/path/parity defects; do not tune selector coefficients, C2 distribution parameters, thresholds, or gates based on this Week-1 result.
