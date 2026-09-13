#!/usr/bin/env python3
"""Reconcile stale QB starter authority with definitive current availability.

Mechanical game-day certification seam only. The versioned official starter still
wins whenever that player is present in the active football roster. If and only
if that exact authority player is absent AND current_player_availability.csv
proves the player definitively unavailable, the already-reconciled unique QB1
fallback remains primary. Missing or ambiguous evidence still fails closed.

No selector feature, C2 formula, mean, distribution, sportsbook input, or model
parameter is changed.
"""
from pathlib import Path

PATH = Path("scripts/modeling/qb_c2_production_adapter_v1.py")

OLD = '''            matches = ranked.loc[ranked["_identity_key"].eq(wanted)]\n            if len(matches) != 1:\n                sample = ranked[["player", "depth_role"]].to_dict("records")\n                raise RuntimeError(\n                    f"official QB starter not uniquely present in football roster team={team} "\n                    f"starter={ar['starter']} roster={sample}"\n                )\n            primary_idx = matches.index[0]\n            source = str(ar["authority_type"])\n            authority_date = str(ar["authority_date"])\n            source_url = str(ar["source_url"])\n            reason = str(ar["reason"])\n'''

NEW = '''            matches = ranked.loc[ranked["_identity_key"].eq(wanted)]\n            if len(matches) > 1:\n                sample = ranked[["player", "depth_role"]].to_dict("records")\n                raise RuntimeError(\n                    f"official QB starter ambiguously present in football roster team={team} "\n                    f"starter={ar['starter']} roster={sample}"\n                )\n            if len(matches) == 1:\n                primary_idx = matches.index[0]\n                source = str(ar["authority_type"])\n                authority_date = str(ar["authority_date"])\n                source_url = str(ar["source_url"])\n                reason = str(ar["reason"])\n            else:\n                # A newer definitive availability fact may invalidate an older\n                # official starter announcement. Ordinary roster absence is NOT\n                # enough: explicit identity-matched unavailable evidence is\n                # required or the existing fail-closed behavior remains.\n                from scripts.utils.qb_starter_availability_v1 import definitive_unavailable_evidence\n                evidence = definitive_unavailable_evidence(team, str(ar["starter"]))\n                if evidence is None:\n                    sample = ranked[["player", "depth_role"]].to_dict("records")\n                    raise RuntimeError(\n                        f"official QB starter not uniquely present in football roster team={team} "\n                        f"starter={ar['starter']} roster={sample}; no definitive unavailable evidence"\n                    )\n                primary_idx = fallback_idx\n                source = f"availability_reconciled_from_{ar['authority_type']}"\n                authority_date = str(ar["authority_date"])\n                source_url = str(ar["source_url"])\n                reason = (\n                    f"{ar['reason']} | superseded by current availability: "\n                    f"{ar['starter']}={evidence['final_availability_state']} "\n                    f"authority={evidence['availability_authority']}; "\n                    f"using reconciled unique QB1={frame.at[fallback_idx, 'player']}"\n                )\n'''


def transform(text: str) -> str:
    if text.count(OLD) != 1:
        raise RuntimeError(f"QB C2 stale-starter authority anchor count={text.count(OLD)}")
    out = text.replace(OLD, NEW)
    # Preserve the existing unique-QB1 fallback and sportsbook-leakage guards.
    required = [
        'raise RuntimeError(f"team={team} has ambiguous Ourlads QB1 fallback: {sample}")',
        'if not audit["sportsbook_inputs_used"].eq(0).all():',
        'FORBIDDEN_SELECTOR_FIELDS',
    ]
    missing = [token for token in required if token not in out]
    if missing:
        raise RuntimeError(f"QB C2 protected starter guards changed unexpectedly: {missing}")
    return out


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    PATH.write_text(transform(text), encoding="utf-8")
    print("CURRENT_AVAILABILITY_QB_C2_STALE_STARTER_SEAM_TRANSFORM_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
