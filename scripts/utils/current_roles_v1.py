#!/usr/bin/env python3
"""Explicit current-role input seam for availability-aware production candidates.

Raw ``data/roles_ourlads.csv`` remains the immutable provider artifact. Current
availability-aware stages opt in by setting ``ACTIVE_ROLES_CSV`` to the
reconciled active-role artifact. Historical/backtest jobs and protected
production retain the raw Ourlads default when no override is supplied.
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
RAW_ROLES=Path("data/roles_ourlads.csv")
DEFAULT_ACTIVE_ROLES=Path("data/roles_ourlads_active_v1.csv")
def resolve_current_roles_path(*,require_active:bool=False)->Path:
    token=os.getenv("ACTIVE_ROLES_CSV","").strip(); path=Path(token) if token else (DEFAULT_ACTIVE_ROLES if require_active else RAW_ROLES)
    if not path.exists() or path.stat().st_size<=0:
        kind="active reconciled roles" if require_active or token else "raw Ourlads roles"; raise RuntimeError(f"{kind} missing/empty: {path}")
    return path
def load_current_roles(*,require_active:bool=False)->pd.DataFrame:
    path=resolve_current_roles_path(require_active=require_active); out=pd.read_csv(path,low_memory=False); out.columns=[str(c).strip().lower() for c in out.columns]
    required={"team","player","role","position"}; missing=required-set(out.columns)
    if missing: raise RuntimeError(f"current-role artifact missing {sorted(missing)}: {path}")
    if "player_clean_key" not in out.columns and "player_key" not in out.columns: raise RuntimeError(f"current-role artifact missing player identity key: {path}")
    if out.empty: raise RuntimeError(f"current-role artifact has zero rows: {path}")
    return out
