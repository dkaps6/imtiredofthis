# 2026 Weeks 1-3 Production Postmortem Findings V1

Status: COMPLETE — DIAGNOSTIC ONLY  
Date: 2026-09-29  
Branch: `research-week3-postmortem-execution-v1`  
Authority run: `36622608145`  
Artifact: `11059171429`  
Artifact digest: `sha256:5e05afca1f94afaa04a4da6f93fa396714df2234d283cc75302b87eac97d38a6`

## Scope and integrity

This result grades the canonical paid production boards for 2026 Weeks 1-3 using
frozen pregame publication/eligibility evidence, postgame nflverse actuals,
verified identity aliases, and the exact Week-3 Tyson Bagent nonparticipation
settlement evidence. No sportsbook refetch, model fitting, coefficient tuning,
or production change occurred.

Week 3 source completeness and settlement passed with zero unresolved rows before
the cumulative report was scored.

## Headline production record

- selected settlement rows: 1,260
- decided bets: 1,240
- voids: 20
- wins/losses: 629-611
- win rate: 50.7%
- units: -40.72
- ROI per unit: -3.28%
- model MAE: 17.55
- selected market-line MAE: 16.36
- model closer than selected line: 45.8%
- model signed bias: -5.36
- selected market-line signed bias: -2.20

By week:
- Week 1: 204-205, -20.92u, 49.9%, model MAE 18.22 vs line 16.93
- Week 2: 192-198, -22.06u, 49.2%, model MAE 18.40 vs line 17.69
- Week 3: 233-208, +2.26u, 52.8%, model MAE 16.17 vs line 14.65

Week 3 improved the betting record, but the selected market line still beat the
model on absolute error and the model remained negatively biased.

## Position and market findings

Cumulative position record:
- QB: 80-62, +10.63u, 56.3%; model MAE 34.62 vs line 34.25
- RB: 221-214, -15.00u, 50.8%; model MAE 18.97 vs line 16.58; model bias -8.17
- WR: 230-224, -16.11u, 50.7%; model MAE 13.45 vs line 12.88
- TE: 98-109, -18.24u, 47.3%; model MAE 11.85 vs line 11.37

Cumulative market record:
- pass_yards: 47-30, +11.82u, 61.0%; model MAE 53.28 vs line 53.72
- rush_yards: 111-101, -0.11u, 52.4%; model bias -7.73
- rec_yards: 216-219, -26.60u, 49.7%; model bias -6.50
- receptions: 205-210, -19.08u, 49.4%
- rush_rec_yards: 50-51, -6.75u, 49.5%; model MAE 31.75 vs line 26.60; model bias -17.63

The strongest mean-construction concern remains rush+receiving yards, especially
RB rush+receiving yards (bias -17.99). TE remained the weakest cumulative
position, although Week 3 itself improved to 40-34 (+3.21u).

## Probability / confidence findings

Declared probability remained materially overconfident:
- mean stated fair probability across decided bets: approximately 68.7%
- realized win rate: 50.7%
- 70%-100% stated band: 502 bets, mean stated 81.0%, realized 51.8%
  (calibration gap -29.2 percentage points)

Edge ordering was not monotonic:
- smallest-edge quintile: 45.4%, -36.39u
- largest-edge quintile: 52.0%, +1.00u
- largest-edge quintile model closer-than-line rate: only 37.0%

Both OVER and UNDER sides lost cumulatively, so the evidence does not support a
simple side-direction correction.

## Cluster-aware slice inference

The frozen clustered inference tested 54 eligible slices using game clusters and
BH-FDR q=0.10.

Result:
`NO_SLICE_SURVIVES_MULTIPLE_COMPARISONS_CORRECTION`

The best nominal slice was pass_yards OVER (39 bets, 66.7%, +10.13u,
cluster p=0.0231), but it did not survive the preregistered multiple-comparisons
gate. Pass yards overall (77 bets, 61.0%, +11.82u) had cluster p=0.0682 and also
did not survive FDR.

Therefore Weeks 1-3 do not authorize a new position/market/side carveout.

## Interpretation

The three-week evidence supports three distinct observations without promoting
any new production rule:

1. Probability confidence is badly miscalibrated relative to realized outcomes.
2. The model still carries a systematic low-mean bias, concentrated most heavily
   in rush+receiving yards and RB yardage construction.
3. No simple categorical betting slice is statistically robust enough to justify
   a production carveout from this three-week sample.

Week 3 is encouraging as a single week, but it does not erase the cumulative
calibration and mean-error problems.

## Research boundary / next step

Do NOT:
- fit a Week-3-only rescue;
- add a QB/pass-yards carveout from the nominal slice result;
- add a side rule;
- rescale all probabilities from these three weeks;
- reopen already-closed historical calibration experiments under a new label.

Next step is a reconciliation audit against the already-completed historical
fair-probability, distribution-widening, strong-gate probability-calibration,
RB width, and specialist-RNG research. The purpose is to identify what the new
Weeks 1-3 evidence actually adds beyond those prior results before freezing any
new experiment.
