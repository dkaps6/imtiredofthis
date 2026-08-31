# Flagged follow-up: intrinsic QB quality audit after RB research

Status: **FLAGGED — DO NOT REOPEN QB MEAN RESEARCH DURING CURRENT RB SEQUENCE**

The promoted M89/M90 QB passing-yards synthesis remains frozen and production-valid. This note exists so the following narrow audit is not lost after RB work.

## Why this is flagged

M89/M90 already treats quarterbacks individually through player-specific component projections, predicted YPA, prior attempts, prior YPA and football environment. However, the final promoted mean synthesis does not explicitly decompose QB-created passing ability from surrounding offensive environment to the same degree now being researched for RBs.

## Post-RB question

After the RB sequence is complete, test whether the following **pregame, player-specific QB quality families** contain incremental forward information after controlling for the frozen M90 projection:

- CPOE / accuracy over expectation and completion-rate stability
- EPA/dropback and success rate
- pressure-adjusted passing efficiency
- sack avoidance / pressure-to-sack tendency
- deep-ball / air-yard efficiency and explosive throw creation
- yards per completion and air-yards-per-completion
- QB dependence on receiver YAC versus QB-created air production
- turnover-worthy / interception tendency where a reliable historical source exists
- red-zone passing efficiency where sample size supports it
- stability/volatility of the above metrics across recent and longer windows

Also separate those QB-created traits from environment families such as pass protection, receiver/target quality, YAC environment, offensive structure/play calling and opponent pass defense.

## Scientific boundary

Do **not** rebuild the QB model from scratch and do **not** disturb M89/M90 unless a narrow residual audit demonstrates stable forward incremental value. The correct future experiment is:

`frozen M90 projection + explicit QB intrinsic-quality residual information`

versus the unchanged frozen M90 projection, under temporal validation and with no sportsbook inputs.

If no stable forward gain exists, keep M90 untouched. If stable independent signal exists, consider a small validated residual layer only.
