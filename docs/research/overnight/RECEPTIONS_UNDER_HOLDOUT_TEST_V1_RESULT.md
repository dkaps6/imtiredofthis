STATUS: RESEARCH ONLY — NOT PROMOTED — AWAITING USER APPROVAL.

# Receptions-UNDER Candidate — Genuine Holdout Test

Direct follow-up to SITUATIONAL_EDGE_HUNT_V1/V2. Those results defined the
top-quartile prob_edge threshold on the POOLED 2024+2025 sample, then checked
each season individually -- 2025 data influenced the threshold used to grade
2025, so that is not a true prospective holdout. This test fits the threshold
on ONE season only and grades the OTHER season blind against the frozen cutoff,
in both directions.

|   fit_season |   test_season |   cutoff |   n |   win_rate |   roi_per_unit |   win_rate_se_vs_50pct |
|-------------:|--------------:|---------:|----:|-----------:|---------------:|-----------------------:|
|         2024 |          2025 | 0.422401 | 577 |   0.526863 |     0.00612132 |              0.0208153 |
|         2025 |          2024 | 0.427179 | 543 |   0.528545 |     0.0159177  |              0.0214571 |

## Reading
The edge is directionally consistent (win rate ~52.7-52.9% both directions,
both ROI positive) but materially weaker than the pooled estimate (+1.50%
pooled ROI in V2 vs +0.61%/+1.59% under genuine holdout here). At n=577/543,
a ~52.7% win rate has a binomial standard error of ~2.1 percentage points against
a 50% null -- this is NOT a result you can call statistically distinguishable
from noise on win rate alone; the positive ROI leans partly on the odds mix, not
just the hit rate. This downgrades the candidate from "validated, modest edge"
to "directionally plausible, unconfirmed at this sample size." Do not promote
or bet on this without more data (a genuine third season, or continued
2026-forward tracking) to grow the holdout sample.
