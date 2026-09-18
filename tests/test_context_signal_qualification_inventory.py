import importlib.util
from pathlib import Path
import pandas as pd

P=Path(__file__).parents[1]/"scripts/research/build_context_signal_qualification_inventory.py"
spec=importlib.util.spec_from_file_location("q",P); q=importlib.util.module_from_spec(spec); spec.loader.exec_module(q)

def row(**kw):
    d=dict(feature_name="x",family="role",grain="player_game",seasons_available="2021-2025",eligible_rows=1000,pregame_coverage=.95,stable_id_coverage=1.0,unknown_rate=.02,duplicate_key_count=0,fanout_count=0,prior_support_median=8,stability_stat="spearman",stability_value=.6,intended_component="targets",redundancy_notes="not direct duplicate",source_class="CANONICAL",mechanism_note="role transition can stale recent usage")
    d.update(kw); return pd.Series(d)

def test_ready(): assert q.qualify(row(),.8,500)[0]=="READY_FOR_FROZEN_EXPERIMENT"
def test_fanout_fails_closed(): assert q.qualify(row(fanout_count=1),.8,500)[0]=="REJECTED_INTEGRITY"
def test_identity_fails_closed(): assert q.qualify(row(stable_id_coverage=.98),.8,500)[0]=="REJECTED_INTEGRITY"
def test_thin_support_not_ready(): assert q.qualify(row(eligible_rows=99),.8,500)[0]=="ENGINEERING_READY_SOURCE_THIN"
def test_missing_stability_not_ready(): assert q.qualify(row(stability_value=float('nan')),.8,500)[0]=="ENGINEERING_READY_SOURCE_THIN"
def test_no_mechanism_descriptive(): assert q.qualify(row(mechanism_note=""),.8,500)[0]=="DESCRIPTIVE_ONLY"
def test_blocked_source(): assert q.qualify(row(source_class="BLOCKED"),.8,500)[0]=="SOURCE_BLOCKED"
