#!/usr/bin/env python3
"""Mechanical R26N staging repair for production identity-key representation.

This helper never changes the immutable downloaded parent. It copies the parent to
an isolated staging tree and adds model_context_bridge.player_clean_key only by an
exact one-to-one (team, player display name) join from PlayerForm. No fuzzy/name
normalization and no football-value changes are permitted.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

EXPECTED_ROWS = 468


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--staged-root", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    a = ap.parse_args()

    source = a.source_root.resolve()
    staged = a.staged_root.resolve()
    if not source.is_dir():
        raise RuntimeError(f"R26N repair source root missing: {source}")
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(source, staged)

    src_data = source / "data"
    stage_data = staged / "data"
    src_form_path = src_data / "player_form_consensus.csv"
    src_context_path = src_data / "model_context_bridge.csv"
    stage_context_path = stage_data / "model_context_bridge.csv"
    for p in (src_form_path, src_context_path, stage_context_path):
        if not p.is_file() or p.stat().st_size <= 0:
            raise RuntimeError(f"R26N repair required file missing/empty: {p}")

    form = pd.read_csv(src_form_path, low_memory=False)
    context = pd.read_csv(src_context_path, low_memory=False)
    if len(form) != EXPECTED_ROWS or len(context) != EXPECTED_ROWS:
        raise RuntimeError(f"R26N repair row-count drift form={len(form)} context={len(context)}")

    required_form = {"team", "player", "player_clean_key"}
    required_context = {"team", "player"}
    if required_form - set(form.columns):
        raise RuntimeError(f"R26N repair PlayerForm missing {sorted(required_form-set(form.columns))}")
    if required_context - set(context.columns):
        raise RuntimeError(f"R26N repair model-context missing {sorted(required_context-set(context.columns))}")
    if "player_clean_key" in context.columns:
        raise RuntimeError("R26N repair expected model-context to lack player_clean_key; parent representation changed")

    form_display_dupes = int(form.duplicated(["team", "player"]).sum())
    context_display_dupes = int(context.duplicated(["team", "player"]).sum())
    if form_display_dupes or context_display_dupes:
        raise RuntimeError(
            f"R26N repair display identity duplicates form={form_display_dupes} context={context_display_dupes}"
        )

    form_display = set(zip(form.team.astype(str), form.player.astype(str)))
    context_display = set(zip(context.team.astype(str), context.player.astype(str)))
    missing_from_form = sorted(context_display - form_display)
    extra_in_form = sorted(form_display - context_display)
    if missing_from_form or extra_in_form:
        raise RuntimeError(
            "R26N repair parent display identities are not exact; "
            f"missing_from_form={missing_from_form[:20]} extra_in_form={extra_in_form[:20]}"
        )

    keymap = form[["team", "player", "player_clean_key"]].copy()
    keymap["player_clean_key"] = keymap["player_clean_key"].astype("string").fillna("").str.strip()
    blank_keys = int(keymap.player_clean_key.eq("").sum())
    key_dupes = int(keymap.duplicated(["team", "player_clean_key"]).sum())
    if blank_keys or key_dupes:
        raise RuntimeError(f"R26N repair invalid PlayerForm key map blank={blank_keys} key_dupes={key_dupes}")

    staged_context = context.merge(
        keymap,
        on=["team", "player"],
        how="left",
        validate="one_to_one",
    )
    missing_keys = int(staged_context.player_clean_key.astype("string").fillna("").str.strip().eq("").sum())
    if len(staged_context) != EXPECTED_ROWS or missing_keys:
        raise RuntimeError(
            f"R26N repair staged context invalid rows={len(staged_context)} missing_keys={missing_keys}"
        )

    # Prove no source row/value was changed except adding the derived key column.
    original_cols = list(context.columns)
    left = context.sort_values(["team", "player"], kind="mergesort").reset_index(drop=True)
    right = staged_context[original_cols].sort_values(["team", "player"], kind="mergesort").reset_index(drop=True)
    if not left.equals(right):
        raise RuntimeError("R26N repair altered model-context source values beyond adding player_clean_key")

    staged_context.to_csv(stage_context_path, index=False)

    staged_form = pd.read_csv(stage_data / "player_form_consensus.csv", low_memory=False)
    staged_context_check = pd.read_csv(stage_context_path, low_memory=False)
    form_keys = set(zip(staged_form.team.astype(str), staged_form.player_clean_key.astype(str)))
    context_keys = set(zip(staged_context_check.team.astype(str), staged_context_check.player_clean_key.astype(str)))
    if form_keys != context_keys or len(form_keys) != EXPECTED_ROWS:
        raise RuntimeError("R26N repair staged compact identity sets are not exact")

    payload = {
        "candidate": "RB_R26N_RUN1_MECHANICAL_IDENTITY_KEY_STAGING_REPAIR_V1",
        "source_root_untouched": True,
        "source_rows": int(len(context)),
        "staged_rows": int(len(staged_context)),
        "display_identity_missing_from_form": 0,
        "display_identity_extra_in_form": 0,
        "form_display_duplicates": form_display_dupes,
        "context_display_duplicates": context_display_dupes,
        "blank_playerform_keys": blank_keys,
        "staged_missing_keys": missing_keys,
        "staged_compact_identity_count": int(len(form_keys)),
        "football_values_changed": False,
        "players_added_or_removed": False,
        "fuzzy_matching_used": False,
        "normalization_heuristic_used": False,
        "repair": "add staged model_context_bridge.player_clean_key by exact one-to-one immutable PlayerForm (team, player) mapping",
        "pass": True,
    }
    a.audit.parent.mkdir(parents=True, exist_ok=True)
    a.audit.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("R26N_MECHANICAL_IDENTITY_KEY_STAGING_REPAIR_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
