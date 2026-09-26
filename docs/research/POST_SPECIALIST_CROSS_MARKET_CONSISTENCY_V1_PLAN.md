# Post-Specialist Cross-Market Consistency Audit V1 — Frozen Diagnostic Plan

Date: 2026-09-26

Status: **FROZEN BEFORE DIAGNOSTIC EXECUTION**

Branch: `research-post-specialist-cross-market-consistency-v1`

## Why this exists

Current production combines football quantities that are jointly generated inside the Monte Carlo simulation, then calibrates several markets independently with market-specific ML/State blends.

The already-frozen RB Vacancy Opportunity V1 Week-3 pregame lock exposed a structural seam without using outcomes:

- the candidate changed only `rules_rush_share`;
- `rules_ypc`, ML, State and ensemble weights were held fixed;
- Monte Carlo implied YPC remained effectively invariant;
- final independently blended `rush_att` and `rush_yards` means produced materially different implied-YPC responses.

This audit asks whether that seam is a general property of the current production architecture and whether any already-promoted algebraic identities are violated.

This is **diagnostic only**. It does not construct, score or promote a repair.

## Immutable authority

Use the sportsbook-free 2026 Week-3 Full Slate artifact:

- run: `36204768034`
- source main SHA: `f7d2011b73950488ea209124ba895b92c401b2b1`
- artifact: `10892728623`
- digest: `sha256:f52b36fb7a9c929fadca473bd8303823fb9f63bd42c884341cd6c3b52c26ed67`
- live odds / sportsbook inputs upstream: **0**

Reconstruct the football universe from the exact frozen source checkout and frozen artifact only.

## Frozen scope

### A. RB/FB rushing cross-market identity

For every production-eligible 2026 Week-3 RB/FB with finite paired `rush_att` and `rush_yards` components:

1. reproduce exact joint Monte Carlo means;
2. reproduce exact generic ensemble means independently for `rush_att` and `rush_yards`;
3. record:
   - `rules_ypc`;
   - MC implied YPC = `mc_rush_yards / mc_rush_att`;
   - final generic ensemble implied YPC = `ensemble_rush_yards / ensemble_rush_att`;
   - component and ensemble weights for both markets.

No outcome fields are permitted.

### B. Exact local opportunity-to-efficiency sensitivity

Do **not** invent or simulate a new candidate.

For each paired RB/FB row, analytically measure the derivative of final implied YPC with respect to a proportional scaling of the **MC opportunity path only**, holding MC efficiency fixed and holding ML/State components fixed.

Let:

- `A(k) = wA_mc * (k * mc_att) + C_A`
- `Y(k) = wY_mc * (k * mc_yards) + C_Y`
- `R(k) = Y(k) / A(k)`

where `C_A` and `C_Y` are the frozen non-MC ensemble contributions.

Report `dR/dk` at `k=1` exactly.

Interpretation:
- the joint MC path has fixed implied efficiency under proportional opportunity scaling;
- a non-zero derivative in the final blended ratio is therefore created by independent downstream market blending, not by a change to football efficiency.

This is an architecture diagnostic, not a scored candidate and not a tuned threshold test.

### C. Team finite-opportunity accounting

For each team:

- record mean simulated team rushing attempts from the canonical joint simulator;
- sum player MC `rush_att` means;
- sum player generic-ensemble `rush_att` means for production-eligible RB/FBs;
- preserve the simulator residual bucket separately.

This is descriptive. No claim that the generic ensemble must equal the entire team rushing total is preregistered.

### D. Protected RB Rush+Receiving Conservation V2 identity

For production-eligible non-Week-1 RB rows, reproduce the promoted `RB_RUSH_REC_CONSERVATION_V2` target mean and verify the exact protected identity:

`rush_rec_yards target mean = final standalone rush_yards mean + final standalone rec_yards mean`

This is an exact mechanical gate and must hold within `1e-8`.

The protected V2 formula is **not** being reopened or retuned.

## Outputs

Persist:

- row-level RB/FB rushing cross-market audit;
- team finite-opportunity summary;
- RB rush+receiving V2 identity audit;
- aggregate descriptive summary with counts, median / p90 / max absolute implied-YPC gaps and local sensitivity;
- provenance manifest.

## Frozen integrity gates

The run is valid only if all are true:

1. source run/artifact/SHA/digest exactly match the immutable authority above;
2. target season/week are exactly 2026 / Week 3;
3. sportsbook inputs used = 0;
4. target-game outcomes read = 0;
5. candidate variants constructed = 0;
6. parameters fit = 0;
7. production mutations = 0;
8. every audited RB/FB identity is unique;
9. paired `rush_att` / `rush_yards` ensemble weights reproduce the frozen production weights;
10. RB Rush+Receiving Conservation V2 exact identity max gap <= `1e-8`.

## Stopping rule

This diagnostic may establish a concrete structural contradiction, but it does **not** authorize a repair.

If the audit shows that independent post-MC blending systematically creates opportunity-dependent implied-efficiency movement while the joint MC football efficiency remains fixed, freeze a **separate repair hypothesis** before any historical or outcome scoring.

Do not:
- refit ensemble weights;
- alter M96 / P3 / RB Vacancy V1;
- alter YPC;
- reopen RB Rush+Receiving Conservation V2;
- use Week-3 outcomes;
- use sportsbook information;
- score multiple repair variants;
- search thresholds, blend coefficients or routers.

Candidate variants scored: **0**  
Parameters fit: **0**  
Sportsbook inputs: **0**  
Production mutations: **0**
