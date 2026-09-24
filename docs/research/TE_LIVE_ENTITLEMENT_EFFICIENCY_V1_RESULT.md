# TE Live Entitlement vs Efficiency V1 — Result

**Status:** COMPLETE / RESEARCH ONLY  
**Disposition:** `CURRENT_SNAP_ENTITLEMENT_SIGNAL_SUPPORTED_EFFICIENCY_REMAINS_PRIMARY_LIVE_GAP`

Canonical execution:

- branch head: `8018f78bd305faf5592ac05c9c7a6838b5bff161`
- workflow run: `35939588747`
- job: `107444309642`
- artifact: `10784323345` / `te-live-entitlement-efficiency-v1`
- artifact digest: `sha256:50fcbe73794dd6111480b9ac0b3e3286d57514515bbe189463571552c8975409`
- Week-2 origin run: `35282021679`
- Week-2 origin artifact: `10523345092`
- Week-2 origin digest: `sha256:6024ed21d7032e6e6440145277d9d4b74e98f319ef286962bac2a786a8d28e3c`
- production changed: **false**
- new sportsbook inputs used by football model: **0**

Run #1 (`35939373232`) failed closed before scientific output because the
sportsbook ledger used provider event hashes while the football origin trace
used schedule game IDs. The frozen hypothesis/gates were not changed. Commit
`8018f78b...` repaired only the identity join using deterministic
team + suffix-normalized player identity; the successful run above is the
first scientific result.

## 1. Week-2 all-TE entitlement result

72 exact Week-2 TE rows matched final outcomes.

| metric | production TE-R5P | W1-2026-snap counterfactual |
|---|---:|---:|
| target-share MAE | 0.05395 | **0.05270** |
| TE-room-share MAE | 0.23389 | **0.22923** |
| worst production-error quartile target-share MAE | 0.12149 | **0.11364** |

Relative target-share MAE improvement: **2.32%**.

All frozen support gates passed:

- all-TE target-share MAE gain >=2%: **PASS**
- all-TE TE-room-share MAE non-worse: **PASS**
- selected-TE target-share MAE not worse >1%: **PASS**
- team TE-pool conservation <=1e-12: **PASS**
- sportsbook inputs in candidate football calculation: **0**

This directly supports the source-continuation decision already merged in
PR #627: strict-prior current-season snap participation contains useful TE
entitlement information and belongs in Week-3+ TE-R5P.

This result does **not** authorize coefficient refitting from Week 2 and does
not rewrite historical Week-2 production.

## 2. Canonical selected TE receiving-yard cohort

Exact Week-2 canonical TE receiving-yard cohort:

- rows: **34**
- target-outcome matches: **34**
- betting record: **13-21**
- production target-share MAE: **0.06950**
- current-snap candidate target-share MAE: **0.06753** (**2.84% better**)
- zero-actual-target rows: **5**

Despite better target-share allocation, simply translating the current-snap
target change through the frozen final implied efficiency did **not** improve
receiving-yard point MAE:

- final production rec-yard MAE: **20.87 yd**
- current-snap entitlement counterfactual MAE: **21.00 yd**

That is an important result, not a contradiction: **the current-snap signal
improves opportunity estimation, but opportunity alone is not sufficient to
repair the live TE receiving-yard problem.**

## 3. Live mechanism attribution

For the selected Week-2 TE receiving-yard cohort:

- mean recoverable absolute error with perfect target entitlement:
  **+4.11 yd**
- mean recoverable absolute error with perfect realized efficiency
  on nonzero-target rows: **+5.68 yd**
- among the 29 nonzero-target selected rows, the larger row-level recoverable
  component was efficiency in **19** rows and entitlement in **10** rows.

This is directionally consistent with historical TE-R1:

- TARGETS = 45.2% of historical error mass;
- YPR = 29.5%;
- CATCH_RATE = 25.3%;
- combined efficiency ~=55%.

The live sample therefore says the same broad thing as the historical
decomposition: **entitlement is materially wrong in some games, but the larger
remaining receiving-yard problem is downstream efficiency/translation.**

## 4. Final ensemble versus MC

On the same selected Week-2 TE receiving-yard cohort:

- final production MAE: **20.87 yd**
- raw MC MAE: **20.63 yd**
- final-minus-MC absolute-error delta: **+0.24 yd**

MC was closer than the final projection on 22 of 34 rows.

This is a small degradation, not evidence for discarding the ensemble, but it
shows that the downstream blend did not rescue TE receiving yards in Week 2.
The next diagnostic must keep **efficiency mean** and **distribution/ensemble**
questions separate.

## 5. Immediate scientific interpretation

The project now has a real Week-3-relevant conclusion:

1. **Keep PR #627 current-season snap continuation.** It passed a direct
   research-only Week-2 counterfactual on the exact paid origin trace.
2. **Do not retune TE entitlement coefficients from two live weeks.** The
   source continuation itself is supported; the larger residual problem is
   elsewhere.
3. **Move the primary TE lane to receiving efficiency / translation.**
   Current-only efficiency is known to be noisy early, so the next candidate
   must be historically justified and heavily shrunk rather than chasing Week-2
   realizations.
4. **Audit TE distribution/ensemble separately.** The final blend was slightly
   worse than MC, and the live scorecard already shows receiving-yard error
   dispersion materially wider than stated model SD.
5. Week 1 remains scoreboard context only for this mechanism study because its
   exact full pregame entitlement trace is not independently recoverable.

## 6. Next frozen question

The next TE study should determine which part of the remaining efficiency lane
is actionable **before Week 3**:

- production YPT / receiving-yard translation mean;
- catch-rate / reception translation;
- or distribution width / ensemble calibration.

It must compare existing historical/prior/current/Bayesian football-only
efficiency states without fitting coefficients to 2026 Week-1/Week-2 outcomes.

No sportsbook line may enter the football candidate, and no global SD multiplier
is licensed by this result.
