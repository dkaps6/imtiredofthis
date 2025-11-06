#!/usr/bin/env python3
"""
Backwards-compatible name canonicalization utilities.

Exports:
  - canonicalize_player_name(source_key: str) -> str
      (legacy API expected by make_player_form.py)
  - build_roles_map(), build_manual_map() helpers
  - norm_key(), strip_middle_initial()

The function canonicalize_player_name uses:
  1) roles_ourlads.csv (player_key -> full name, ignoring middle initials)
  2) data/manual_name_overrides.csv (player_source_name -> full name), overrides roles
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# ---------- helpers ----------

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

_SPACES = re.compile(r"\s+")
_TRAIL_TAG = re.compile(r"\s+[A-Z]{1,3}\d{1,2}(?:/[A-Z]\d{1,2})?$")

def strip_middle_initial(full_name: str) -> str:
    if not isinstance(full_name, str):
        return ""
    parts = re.split(r"\s+", full_name.replace(",", " ").strip())
    parts = [p.replace(".", "") for p in parts if p.strip()]
    if not parts:
        return ""
    first = parts[0]
    last = parts[-1]
    if last.lower() in SUFFIXES and len(parts) >= 3:
        last = parts[-2] + " " + parts[-1]
    return f"{first} {last}".strip()


def _strip_ourlads_noise(name: str) -> str:
    """Normalize raw roster names into "First Last" style strings."""

    if not isinstance(name, str):
        return ""

    working = name.strip()
    if not working:
        return ""

    # remove trailing roster codes like "24/1", "CF25", "U/NE"
    working = _TRAIL_TAG.sub("", working)

    # flip "Last, First" ordering when present
    if "," in working:
        last, first = [p.strip() for p in working.split(",", 1)]
        working = f"{first} {last}".strip()

    # collapse whitespace + normalize apostrophes
    working = working.replace("’", "'")
    working = _SPACES.sub(" ", working)

    return working.strip()

def norm_key(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    for ch in (" ", "'", "-", "."):
        s = s.replace(ch, "")
    return s

# ---------- map builders (cached) ----------

@lru_cache(maxsize=1)
def build_roles_map(roles_path: str = "data/roles_ourlads.csv") -> dict:
    p = Path(roles_path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    if "player_key" not in cols or "player" not in cols:
        return {}
    key_col = cols["player_key"]
    name_col = cols["player"]
    df["_key"] = df[key_col].astype(str).map(norm_key)
    df["_full"] = df[name_col].astype(str).map(strip_middle_initial)
    return dict(zip(df["_key"], df["_full"]))

@lru_cache(maxsize=1)
def build_manual_map(overrides_path: str = "data/manual_name_overrides.csv") -> dict:
    p = Path(overrides_path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    need = {"player_source_name", "full_name"}
    if not need.issubset(df.columns):
        return {}
    df["_key"] = df["player_source_name"].astype(str).map(norm_key)
    return dict(zip(df["_key"], df["full_name"]))

# ---------- legacy API (kept for make_player_form.py) ----------

def canonicalize_player_name(source_key: str) -> str:
    """Return a cleaned "First Last" style name for varied upstream inputs."""

    if source_key is None:
        return ""

    cleaned = _strip_ourlads_noise(str(source_key))
    if not cleaned:
        return ""

    k = norm_key(cleaned)
    manual = build_manual_map()
    if k in manual:
        return manual[k]
    roles = build_roles_map()
    return roles.get(k, cleaned)

# Optional: explicit alias used in some places
canonicalize_name = canonicalize_player_name


_UNMAPPED_LOG = os.environ.get("UNMAPPED_NAME_LOG", "data/_debug/unmapped_names.jsonl")


def log_unmapped_variant(
    source: str,
    raw_name: str,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Best-effort logging for unmapped canonical name variants."""

    payload = {"source": source, "raw_name": raw_name}
    if context:
        try:
            payload.update(dict(context))
        except Exception:
            payload["context"] = repr(context)

    try:
        log_path = Path(_UNMAPPED_LOG)
        os.makedirs(log_path.parent, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # This logger is intentionally fail-safe; never raise upstream.
        pass
