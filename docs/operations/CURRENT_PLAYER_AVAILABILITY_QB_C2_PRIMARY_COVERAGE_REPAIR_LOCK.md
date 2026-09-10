# Current Player Availability — QB C2 Primary-Coverage Repair Lock

Status: `LOCKED BEFORE REGRESSION / NO 35-GATE RESULT`

Frozen repair plan commit: `c79871c16ff9006715b59557fedd8ca9f61abb47`

Frozen components:
- protected QB C2 adapter blob: `7b677470b27b6776055c75c924a0ddf22d724a44`
- first starter-audit coverage transformer blob: `c7569c54cda779eb04bed7dbf2b22b9ec4fb526b`
- second primary-QB coverage transformer blob: `3d24597727d496d2e2547c3a01b0f995700a7b82`
- shared eligible-team helper blob: `77b591e431378ec984c51e8a032262e673d4c843`

The dedicated regression may test only the mechanical coverage semantics frozen in the repair plan. It may not run the 35-gate evaluator/finalizer, change any scientific parameter, or produce a promotion disposition.
