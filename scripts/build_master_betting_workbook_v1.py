#!/usr/bin/env python3
"""Stable Full Slate entrypoint for the master betting workbook publisher."""
from scripts.master_betting_workbook_core_v2 import main as build_workbook
from scripts.bet_evidence_v0_overlay import main as add_bet_evidence_v0

if __name__ == '__main__':
    build_workbook()
    add_bet_evidence_v0()
