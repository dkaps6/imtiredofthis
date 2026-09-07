# TE-R4 Strict-Prior Participation Source — Result

## Disposition

`STRICT_PRIOR_TE_PARTICIPATION_ELIGIBLE`

This was a source/integrity audit only. No model was fit and no production projection changed.

## Canonical evidence

- Workflow run: `34127474412`
- Job: `101759439856`
- Tested SHA: `ffe4e1101193b3502a6b069d2b77347049719b1d`
- Artifact: `10020700686` (`te-r4-strict-prior-participation-source`)
- Artifact digest: `sha256:56de7efe302cdf9c329c3a0386d798789d15e6a495c865535291173b6840c163`
- Source: `nflreadpy.load_snap_counts`
- Source seasons: 2020-2025
- Source rows: 150,909
- Target rows: 5,371 across 2021-2025
- Sportsbook inputs: 0
- Same/future observations used: 0
- Duplicate rate: 0
- Same-game exact-key source match rate: 0.986780860175014

## Strict-prior availability

Pooled availability on TE target rows:

- prior-1 any-team participation: `0.9791472723887544`
- prior-3 any-team participation: `0.9592254701172966`
- prior-1 same-team participation: `0.9659281325637684`
- prior-3 same-team participation: `0.9178923850307206`
- prior-1 offense percentage non-null: `0.9791472723887544`
- prior-1 offense snaps non-null: `0.9791472723887544`
- prior-3 offense percentage non-null: `0.9592254701172966`
- prior-3 offense snaps non-null: `0.9592254701172966`

Every season passed the frozen prior-1 any-team coverage gate; 2025 remained `0.96260017809439` despite being the lowest season.

## Frozen gates

All integrity gates passed:

- all six source seasons present;
- duplicate rate <= 0.01;
- sportsbook inputs = 0;
- same/future observations used = 0.

All coverage gates passed:

- pooled prior-1 any-team >= 0.75;
- pooled prior-1 same-team >= 0.55;
- pooled prior-3 any-team >= 0.60;
- every season prior-1 any-team >= 0.65;
- both prior-1 snap fields >= 0.70.

## Scientific implication

Historical TE participation is not source-blocked. Strict-prior offensive snap percentage and offensive snap count are sufficiently broad and stable to enter a separately frozen individual TE entitlement/allocation experiment.

This result does **not** prove that snaps improve projections. It only authorizes science using pregame participation history.

The next TE experiment must use participation at the individual-allocation layer, not as a blanket positive TE target correction. TE-R3 showed that team-pool corrections can improve target MAE while worsening receiving-yard MAE when the correction is insufficiently differentiated across players. Participation therefore belongs in a hierarchical opportunity model: team pass state -> team TE pool -> individual TE entitlement -> catch/yard efficiency.
