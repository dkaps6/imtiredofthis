# RB STACK6I — Implementation Correction

The first execution of the frozen STACK6I protocol failed before any oracle outputs were written.

Failure: pandas `Series.corr` namespace collision in the integrity check. The code used `base.corr`, which resolves to the pandas method, instead of the stored metric field `base["corr"]`.

This is a mechanical implementation correction only.

Frozen equations, population, oracle arms, metrics, attribution thresholds, input artifacts, and no-fit/no-sportsbook constraints are unchanged.

The corrected execution must still reproduce the frozen M94C W6-18 baseline before any oracle result is interpreted.
