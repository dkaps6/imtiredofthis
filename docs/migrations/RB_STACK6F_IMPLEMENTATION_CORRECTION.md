# RB STACK6F — Implementation Correction

Status: FROZEN BEFORE SCORED RERUN

The first STACK6F Action (`33578334642`) failed before writing any score because the retention function accessed the pandas Series field `corr` through attribute syntax (`cur.corr`), which resolves to the Series correlation method rather than the stored metric value.

Correction only:

- replace `cur.corr` / `base.corr` with explicit indexed metric access `cur["corr"]` / `base["corr"]`;
- make no change to the frozen feature set, Ridge alpha, temporal split, arms, blend weight, population, outcomes, or retention thresholds;
- rerun the same frozen STACK6F protocol.

Run `33578334642` is an implementation failure and is not football evidence.
