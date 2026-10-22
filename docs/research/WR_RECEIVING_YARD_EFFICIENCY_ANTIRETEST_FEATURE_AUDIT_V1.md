# WR Receiving-Yard Efficiency Anti-Retest + Feature-Availability Audit V1

**Status:** RESEARCH GOVERNANCE / AUDIT ONLY. NO CANDIDATE RESULT. NO PRODUCTION CHANGE. NO PAID FULL SLATE.

**Audit branch:** `research-wr-yard-efficiency-feature-audit-v1`  
**Exact base main:** `de6aed84d474867d81427f4e8277219868ac9d50`  
**Date:** 2026-09-14

## 1. Purpose

The active WR problem is no longer generic target entitlement. The promoted M38 hierarchy plus `WR_R15_PRODUCTION_MODEL_V1` remain the football authority for WR opportunity. Current evidence says WR receptions/opportunity are comparatively healthy while receiving-yard translation remains weak, especially high-efficiency/right-tail games.

This document performs the required audit **before any new WR-yard candidate is run**. It maps the relevant feature families already available or previously researched, their source and temporal safety, what prior migration actually tested them, whether the test was marginal or interactive, and whether a genuinely new mechanism still exists.

The north star is individual football projection accuracy. Sportsbook comparison is downstream only and is not an upstream feature source.

## 2. Canonical authority and current failure state

### Promoted opportunity authorities

- M38 WR hierarchy remains canonical.
- `WR_R15_PRODUCTION_MODEL_V1` remains canonical for the qualified secondary-WR target allocation layer.
- WR-R15 canonical run: `34238301577`, artifact `10061328722`, digest `sha256:8df31...` as recorded in the current handoff.
- M38 was later confirmed across 2020-2025 without retuning: receiving-yard MAE improved in all six seasons versus M37. See `docs/migrations/WR_R1_MULTISEASON_REPLICATION_RESULT.md` on `research-wr-nd6-player-level-explosive-ceiling`, run `34058453941`, artifact `9997412312`, digest `sha256:d75bf1233bdd48d119508c0631a0f904e0256239e471bb72804d705d73ed6cfe`.

### Authority-exact WR1 decomposition

Canonical branch/run/artifact:

- branch: `research-wr1-yardage-decomposition-v1`
- head: `25d904f7d8abe9c2af6698c1dee5568f812eac51`
- run: `34858963515`
- artifact: `10353787250`
- digest: `sha256:3e6fb21956e0f2681a732379412b6eff0c3e17dd561d7358e8995e279fb42f5d`

Key authority facts:

- Scientific WR1 identity is the actual M38/R15 hierarchy / `wr_rank == 1`, not literal historical `role == WR1`.
- Current-production WR1 receiving-yard direction was ~47.98%.
- WR1 yardage error is approximately half target/opportunity error and half YPT/efficiency error after collapsing the decomposition.
- Actual 100+ receiving-yard WR1 games are a severe miss cluster.
- Top actual-YPT games are heavily underprojected.
- Opportunity-matched WR2+ receiving-yard performance is also weak, so this is broader than WR1.

### Closed C2 -> WR1 shared-tail lane

This is a preserved failure, not a tuning opportunity.

- canonical cohort: 884 rows (444 in 2024, 440 in 2025)
- frozen holdout gates failed
- Claude independently built a second pipeline and reached the same substantive failure on an 849-row actual-usage sensitivity cohort

Do not retry upper95, alternate percentiles, skew, or threshold tuning.

## 3. What production currently uses for WR yardage

The present architecture makes the open problem unusually clear.

### 3.1 Projected opportunity

M38/R15 produce WR target entitlement. R15's feature family is role/participation/room-state oriented: secondary-room pool/size, prior offensive snaps and snap percentages, secondary snap shares, same-team and any-team historical participation, with WR1 and room-mass conservation protections.

These features govern **how many targets** a WR receives. They are not a structural yards-per-target model.

### 3.2 Historical YPT prior

`scripts/modeling/bayesian_v2.py` builds `bayes_ypt` as empirical-Bayes shrinkage of historical player YPT toward a position prior. It uses prior/current evidence from the already cutoff player-form layer. This is leakage-safe historical evidence, but it is still a **realized historical YPT prior**, not a decomposition of target depth, completion quality, route type, or post-catch opportunity.

### 3.3 Generic context/rule layer

`scripts/modeling/rules_v2.py` / `simulation_rules.py` can modify YPT with broad team-context pass-efficiency / coarse coverage rules. This remains generic context, not a promoted WR-specific efficiency mechanism.

### 3.4 Simulation translation

`scripts/simulation_v2.py` effectively translates WR receiving-yard mean as projected targets x YPT (with broad pass-efficiency shock in simulation), while receptions are simulated separately.

**Conclusion:** production contains a strong opportunity architecture and a historical-YPT prior, but no promoted structural WR yardage-per-opportunity model.

## 4. Feature / mechanism anti-retest matrix

| Feature / mechanism family | Existing source | Historical leakage-safe? | Prior test / authority | Marginal or interactive? | Result / status | Retest status |
|---|---|---|---|---|---|---|
| M38/R15 hierarchy, secondary-room role, prior snaps, room shares | Historical player logs / role state / R15 frozen assets | Yes under frozen walk-forward construction | M38 + WR-R15 | Interactive allocation model / protected conservation architecture | Promoted; WR-R15 improved targets and yards | **KEEP; do not retune blindly** |
| Historical player YPT | `player_form_v2.py` -> `bayesian_v2.py` | Yes if cutoff path preserved | current production | Marginal/shrunk prior | Production baseline | **KEEP as baseline; structural add-on may be new** |
| Historical catch rate / receptions per target | weekly player stats -> player form | Yes if cutoff path preserved | current production / prior WR decomposition | Marginal historical translation | comparatively healthier than yardage | **Not current priority** |
| Route rate / YPRR schema slots | weekly source only when a *real routes field* exists | Conditionally; source coverage must be proven | current `player_form_v2.py` schema | Marginal | Code explicitly refuses to fake routes from targets/dropbacks | **AVAILABLE ONLY WHEN TRUE ROUTES EXIST; no assumed historical coverage** |
| Prior player EXP20/EXP40 per target | nflverse PBP | Yes with prior-game cutoff | WR-R7; WR-ND6 | Marginal | Failed; ND6 strongest EXP40 near-signal still failed gates | **CLOSED** |
| Prior player YAC/reception | nflverse PBP | Yes | WR-R7; WR-ND6 | Marginal | Failed | **CLOSED as simple persistent trait** |
| Prior player mean air yards / target | nflverse PBP | Yes | WR-R7; WR-ND6; also adjacent M72/M75 | Marginal | Failed / often wrong direction | **CLOSED as simple mean-depth trait** |
| Opponent EXP20/EXP40, YAC, mean air allowed | nflverse PBP | Yes | WR-R7; WR-ND6; M72 | Marginal | Failed | **CLOSED** |
| Player explosive/YAC/air x opponent explosive/YAC/air weakness | nflverse PBP | Yes | M72 | Explicit interactions (`bridge_exp20`, `bridge_exp40`, `bridge_yac`, `bridge_air`, top-weapon x defense) | No actionable new-information bridge | **CLOSED** |
| NGS separation, cushion, mean aDOT / intended air, YACOE / expected YAC, secondary quality | NGS/vendor tracking + defense aggregates | Strict-prior versions were built | M75 | Both marginal and explicit interactions (separation x def YPT, YACOE x def YAC, aDOT x def aDOT, top target share x weak-secondary YPT) | Failed to establish actionable incremental signal | **CLOSED under this family/specification** |
| NGS WR separation/cushion/intended-air share / aDOT for target entitlement | NGS, WR-R10 strict-prior eligible data | Yes; R10 cleared source gates | WR-R11, 6,383 eligible OOS rows | Fixed Ridge residual target model; no post-hoc interactions | `WR_NGS_TARGET_MODEL_FAIL`; receiving-yard MAE worsened +1.1598 and regressed all four seasons | **CLOSED for generic NGS residual-lift target model** |
| Prior-game snap level / acceleration | participation/snap data | Yes when timestamped prior | WR-R8 (inside exact TARGETS-dominant class) | Marginal | Failed continuous allocation gates | **CLOSED; no ND5 retry** |
| Static depth top-2 state / depth-rank promotion | historical depth/role source | Yes under source audit | WR-R8 | Marginal | Failed; top-2 state direction was negative | **CLOSED** |
| Vacated targets / higher-usage WR absence counts | historical role / availability construction | Yes under frozen ND3 | WR-ND3 | Marginal/dynamic entitlement | `NO_ACTIONABLE_DYNAMIC_ENTITLEMENT_SIGNAL` | **CLOSED** |
| Prior signed WR projection residual / player difficulty / extreme-miss history | frozen historical prediction residuals | Yes under walk-forward construction | WR-R3 diagnostic + later combined calibration | Mean calibration + uncertainty/tail integration | R3 persistence existed, but integration missed promotion gate; follow-up Ridge/HGB did not rescue | **CLOSED as generic residual-history correction** |
| Team man/zone mix | coverage-v2/team coverage | Can be prior-safe | prior feature ablation / coverage novelty check | Marginal | Near-null for WR yards/receptions | **CLOSED as simple team coverage-rate signal** |
| True receiver-to-defender responsibility / player WR-CB assignment | would require historical charted responsibility/alignment | No honest historical source in repo | M84 source audit | N/A | `HOLD_SOURCE_BLOCKED_NEW_INFORMATION` | **BLOCKED; do not fake assignment** |
| Vendor NGS richer tracking source broadly | NGS | R10 proved strict-prior availability | R9-R11 | source audit + model | Scientific fail for tested target mechanism | **Do not rerun same vendor-NGS lane under new label** |
| Shared QB C2 distribution -> WR1 tail selector | promoted QB C2 distribution + WR1 authority | Yes under frozen cohort | C2->WR1 shared-tail test + Claude independent pipeline | Conditional/shared-tail interaction | Failed both pipelines | **CLOSED; no tail-selector rescue** |
| Shared QB/receiver target-mass correction | existing projected QB/receiver state | leakage-safe in prior design | C1 | joint/conditional | Failed / closed | **CLOSED** |
| Broad joint QB/receiver combination | existing QB/WR projected state | leakage-safe in prior design | C3 | joint | Failed / closed | **CLOSED** |
| QB CPOE / completed-air / deep efficiency history as generic QB signal | nflverse PBP | Yes when strict-prior | M70/M71 family | Marginal/regime/uncertainty on QB residuals | pregame efficiency/volatility families did not clear QB residual gates | **Not a brand-new football mechanism; WR use would need materially different causal framing** |

## 5. Exact closed-lane details that constrain the next hypothesis

### 5.1 WR-ND6 closes the simple explosive-ceiling formulation

Canonical ND6:

- branch: `research-wr-nd6-player-level-explosive-ceiling`
- tested SHA: `ce17a3d8c6f5cd9ed1bb63249e61e72540199f8a`
- run: `34056970854`
- artifact: `9996440756`
- digest: `sha256:2382554e2c723589ae502a1bee7bb0df08e4f811cf4b83d1c8b7e380269edc81`
- disposition: `NO_ACTIONABLE_EXPLOSIVE_CEILING_SIGNAL`

It tested prior-8 player EXP20/EXP40 per target, YAC/reception, mean air/target, plus opponent explosive/YAC/air allowances. No signal passed. Because no player-side and no defense-side marginal passed, its frozen plan did **not** authorize player x defense interaction rescue.

This rules out repackaging “explosive WR + explosive-vulnerable defense” as the next experiment.

### 5.2 WR-R8 closes static role/participation rescue

Canonical R8:

- branch: `research-wr-r8-target-dominant-role-signals`
- tested SHA: `9b55bad23b9db2fb6e00b87df4103ab37457a242`
- run: `34072748077`
- artifact: `10000997724`
- digest: `62ec2ac728b7e58cf355077e82d8b3bed0677b1f14d02af3cd3eafbd43a4028c`
- disposition: `NO_ACTIONABLE_WR_TARGET_DOMINANT_ROLE_SIGNAL`

Prior-game snap level, snap acceleration, top-2 depth state, and depth-rank promotion all failed. Do not run another ND5/snap-depth variant.

### 5.3 WR-R11 closes generic NGS entitlement lift

Canonical R11:

- branch: `research-wr-r11-strict-prior-ngs-target-model`
- frozen plan: `b30d9808e9b8f1e7c5268c8a19972cd85fc13f7f`
- implementation: `f402a01b38d4e5bf18747c6a97e47f13dc5e2ff3`
- run head: `d2f251dc267db854e6747734d6cbe692d56f93ae`
- run: `34124533822`
- artifact: `10019545653`
- digest: `sha256:388973f0fc67c6a2f0e4868a7c874d310ff7711cd7b0a6fb38c74128dcd43370`
- disposition: `WR_NGS_TARGET_MODEL_FAIL`

Strict-prior separation, cushion, intended air yards, intended-air share, and B0 expected targets produced only a small target-MAE gain and materially worsened receiving-yard MAE. No alpha/cap/window rescue is authorized.

### 5.4 M75 already tested interactions, not just marginals

M75 is important to the anti-retest rule because the repo did more than inspect simple columns. It constructed strict-prior receiver tracking and secondary aggregates and explicitly tested interaction terms between receiver separation/YACOE/aDOT/target concentration and opponent secondary quality. Therefore a new proposal cannot claim novelty merely because it multiplies a player depth/separation variable by a defense metric.

### 5.5 M84 blocks fake WR-CB richness

The desired player-level coverage/alignment lane is scientifically attractive but historically unavailable in the repo at the required responsibility grain. M84 correctly stopped rather than inventing responsibility from participation. That stop remains binding.

## 6. Feature availability: what can honestly be reconstructed now?

### A. High-confidence strict-prior sources

1. **Weekly player outcomes / usage**
   - targets, receptions, receiving yards, team target denominators
   - historical player/team identity through GSIS when available
   - safe when target week is excluded

2. **nflverse play-by-play target events**
   - receiver player ID and name
   - `air_yards`
   - completion indicator
   - passing/receiving yards
   - yards after catch
   - CPOE where available
   - can be aggregated strictly before the target game/week

3. **NGS historical features already source-audited by R10**
   - separation, cushion, intended-air metrics and related fields
   - source eligible, but the tested generic R11 target mechanism failed

4. **Historical depth/snap/role state already used in ND/R15 work**
   - usable only under the prior timestamp rules already established

### B. Conditionally available / must not be assumed

1. **True route counts / route participation / YPRR**
   - `player_form_v2.py` explicitly populates route metrics only when the weekly source actually contains a real `routes` or `routes_run` field.
   - It deliberately does not relabel targets/dropbacks as routes.
   - Historical completeness therefore must be measured before any route-based hypothesis is frozen.

2. **Expected-YAC / expected target-quality fields**
   - some expected-YAC information exists in earlier tracking work and raw/PBP-related source audits, but novelty versus M75's expected-YAC/YACOE family is not established.
   - any future use requires an exact source/coverage audit and a materially different causal mechanism, not “expected YAC but for WRs.”

### C. Not honestly available at required historical grain

- true WR-CB responsibility
- assignment-specific help responsibility
- route-by-route alignment versus exact defender
- historical manual matchup charting equivalent to a true coverage responsibility feed

These remain source blocked.

## 7. Candidate inventory after the anti-retest audit

This section is **not permission to run any candidate**. It only identifies whether a mechanism appears open enough to send to Claude for independent falsification.

### Candidate family A — targeted-pass depth DISTRIBUTION / regime shape conditional on projected opportunity

**Current status: OPEN ENOUGH FOR INDEPENDENT REVIEW; NOT YET FROZEN.**

Mechanism distinction:

- R7/ND6 tested **mean** player air yards per target.
- M75 tested average intended-air/aDOT-type tracking features and player-defense interactions.
- M72 tested mean weapon air per target, realized explosive rates, and player/team x defense bridges.
- R11 used average intended air yards / intended-air share primarily for target entitlement.
- None of the audited result lineage establishes a frozen WR receiving-yard test of the **shape of the targeted-pass depth distribution itself** (dispersion / deep-target mass / bimodality-like regime), conditional on already-projected M38/R15 opportunity.

Why this could be football-real rather than a relabel:

A WR receiving 8 projected targets at a stable 9-yard mean depth can have a materially different yardage distribution than a WR whose target history is a mixture of screens and 30-yard shots even when the two receivers share a similar average air/target. The hypothesis would be about the *distribution of opportunity quality*, not persistent realized YPT and not a simple high-aDOT adjustment.

Honest source:

- raw nflverse PBP targeted-pass `air_yards`
- receiver GSIS ID should be the primary identity
- strict-prior target events only

Critical caveats before freeze:

- targeted-pass depth is not true route depth; it only describes targets actually thrown.
- minimum prior target-event support must be prospectively defined.
- feature family must be kept small and frozen; no percentile fishing after holdout.
- exact interaction with projected opportunity must be defined before holdout.
- Claude must first verify no prior branch tested the same distribution-shape mechanism under another name.

### Candidate family B — QB x WR delivery state (existing WR-R16 draft)

**Current status: BLOCKED_PENDING_ANTIRETEST_AND_CLAUDE_REVIEW. DO NOT RUN.**

Existing branch: `research-wr-r16-qb-wr-delivery-state-v1`, head `08331f169f255f3530d539f7e91d66fc69bfb3be`.

The draft includes WR target CPOE, WR completed-air/target, target-depth SD, recent-vs-long deltas, team CPOE, team air/attempt, team deep15 completion rate, completed-air/attempt, and QB/WR combination signals.

Novel pieces may exist, especially target-depth dispersion and a predeclared WR x QB delivery interaction. But the family is **not cleanly novel as drafted** because:

- QB CPOE/completed-air/deep efficiency families were already studied in M70/M71 in QB residual research;
- mean air-depth and related player traits are already covered by R7/ND6/M75;
- C3 already closes broad joint QB/receiver combinations;
- the implementation loads `receiver_player_id` but currently keys receiver history primarily through canonicalized `receiver_player_name`, which is weaker than the repo's current GSIS-first identity authority.

Therefore R16 is not authorized to run from its present freeze. If Claude believes a narrow subset is genuinely new, it should be rewritten as a smaller mechanism with GSIS-first identity and a new prospective freeze.

### Candidate family C — expected target quality / xYAC

**Current status: HOLD / NOVELTY NOT ESTABLISHED.**

An expected-quality decomposition is conceptually interesting, but M75 already included expected-YAC/YACOE-style information. A raw xYAC feature would therefore not be presumed new. It would need to answer a different football question and clear an exact source/coverage/anti-retest audit first.

### Candidate family D — WR-specific uncertainty / width

**Current status: CLOSED FOR GENERIC RESIDUAL-HISTORY VERSION; OPEN ONLY IF A NEW SOURCE-DRIVEN REGIME EXISTS.**

R3 established some player difficulty/extreme-miss persistence, but its combined integration did not pass promotion and follow-up model-family rescue was non-actionable. Do not simply reapply prior abs-error/extreme-miss history as a width multiplier.

A future uncertainty candidate would need a real football-state input (for example, a proven opportunity-quality regime) rather than the same prior-error history.

### Candidate family E — richer coverage/alignment/separation

**Current status: SOURCE BLOCKED OR ALREADY TESTED.**

- team coverage-rate context: already near-null
- NGS separation/cushion/aDOT/YACOE: already tested
- player-defense interactions: M75 tested
- true assignment/responsibility: M84 says no honest historical source

No candidate here should run unless a genuinely new historical responsibility source is obtained.

## 8. Provisional frontier recommendation for Claude to challenge

The cleanest still-open mechanism after this audit is **targeted-pass depth distribution/regime shape conditioned on already-projected M38/R15 opportunity**, not mean aDOT, not realized explosive rate, not historical YPT correction, and not player x defense matchup multiplication.

This is only a recommendation for independent challenge. It is **not yet a frozen experiment**.

Claude should independently answer:

1. Did any prior WR branch already test target-depth distribution shape (not mean air/target) against future receiving-yard residuals?
2. Is targeted-pass depth distribution a meaningful pregame football signal or merely noisy selection on thrown targets?
3. What minimum prior-target support is defensible without destroying coverage?
4. Should the mechanism predict YPT residual, receiving-yard residual conditional on projected targets, or uncertainty/tail width rather than the mean?
5. Is there a better genuinely new mechanism in existing leakage-safe data?
6. Does the existing WR-R16 draft duplicate too much prior M70/M71/R7/M75/C3 work to justify continuing?

## 9. Freeze gate before any result run

No candidate may run until a single hypothesis survives Claude's independent audit and is prospectively frozen with all of:

- exact authority cohort and identity rules;
- exact development and untouched holdout split;
- exact feature definitions and transformations;
- exact source fields;
- strict temporal cutoff / source availability rules;
- exact baseline and production parity requirements;
- football metrics (MAE, RMSE, bias, correlation, direction where meaningful);
- high-opportunity and high-yardage/tail protection gates;
- per-role and temporal replication/protection gates;
- leakage gates;
- stop rules forbidding threshold/window/percentile rescue;
- no sportsbook inputs upstream;
- no paid Full Slate run.

Until that freeze exists, the correct disposition is:

**`WR_YARD_EFFICIENCY_AUDIT_COMPLETE_CANDIDATE_NOT_YET_AUTHORIZED`**
