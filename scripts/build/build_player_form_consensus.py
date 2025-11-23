#!/usr/bin/env python3
"""
Build player_form_consensus.csv from data/player_form.csv using existing enrichment helpers.
"""

import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from scripts.enrich_player_form import build_player_form_consensus

logger = logging.getLogger(__name__)

PLAYER_FORM_PATH = Path("data") / "player_form.csv"
OUT_PATH = Path("data") / "player_form_consensus.csv"


def main(player_form_path: Path = PLAYER_FORM_PATH, out_path: Path = OUT_PATH) -> None:
    logging.basicConfig(level=logging.INFO)
    os.makedirs(out_path.parent, exist_ok=True)

    if not player_form_path.exists():
        raise FileNotFoundError(f"Missing player form input: {player_form_path}")

    df = pd.read_csv(player_form_path)
    consensus_df = build_player_form_consensus(df)

    if consensus_df.empty:
        # Instead of stopping the pipeline, write an empty CSV with headers and log a warning.
        logger.warning("Player form consensus is empty; writing header only and continuing.")
        consensus_df.to_csv(out_path, index=False)
        return

    consensus_df.to_csv(out_path, index=False)
    print(f"[build_player_form_consensus] Wrote {len(consensus_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
