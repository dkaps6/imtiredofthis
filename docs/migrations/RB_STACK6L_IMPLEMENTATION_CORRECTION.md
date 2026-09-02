# RB STACK6L — Implementation Correction

The first STACK6L execution failed before PBP reconstruction or any scientific output.

Failure: `nflreadpy.load_pbp(...).to_pandas()` requires `pyarrow`, but the workflow dependency list omitted it.

Correction: add frozen `pyarrow==17.0.0` to the workflow dependency install.

No changes are made to:
- the PBP state definitions;
- the eight correction subsets;
- Shapley mathematics;
- populations;
- integrity requirements;
- attribution thresholds;
- downstream interpretation rules.

The corrected run remains the same frozen STACK6L protocol.
