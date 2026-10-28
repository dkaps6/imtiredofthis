# replay_inputs — reproduction fixtures, NOT research inputs

Each subdirectory is one slate's model-side pricing inputs, gzipped, exactly as
production had them. They exist so the production pricing chain can be re-run later.

**These files are deliberately NOT denylist-stripped.** Unlike `../explanatory/`, which
is scrubbed of every sportsbook and decision column for research use, a replay needs the
inputs production actually consumed. Do not use anything here to construct a projection.

`manifest.json` in each directory carries a sha256 per file, so a replay can prove it ran
on the same bytes.
