# Hierarchical Receiver Mean Reconciliation V1 — Pass-Attempt Input Amendment

Date: 2026-09-24

Status: **FROZEN BEFORE OUTCOME SCORING**

This clarification changes no scientific gate and uses no target-game outcome.

The frozen uncertainty formula requires a team pass-attempt opportunity term
`A`.

Historical `simulation_explicit_entitlement_v1` intentionally returns player
arrays but not the internal team `pass_att` draws. Rather than introduce a
Monte Carlo-estimated weight that changes with simulation seed, V1 will use the
exact deterministic canonical pre-simulation opportunity inputs already consumed
by `simulation_v2`:

`plays_mean, pass_rate_mean = simulation_v2._team_inputs(team_rows)`

and

`A = plays_mean * pass_rate_mean`

This is:
- strict-prior/pregame;
- sportsbook-independent;
- deterministic;
- parameter-free;
- directly tied to the canonical simulation inputs.

No fitted correction or new threshold is introduced.

The uncertainty formula remains:

`V_i = (A * YPT_i * SD_share_i)^2 + (A * Share_i * SD_ypt_i)^2`

All other frozen candidate rules and gates remain unchanged.
