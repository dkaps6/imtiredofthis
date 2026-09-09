"""Deployable Phase-J QB distribution-state selector.

The selector is mean-neutral. It predicts whether to expose the C2-derived QB
passing-yard distribution while preserving the promoted M89/M90 point mean.
Sportsbook inputs are prohibited.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

MODEL_PATH=Path('model/qb_distribution_state_selector_v1.json')


def load_artifact(path:Path=MODEL_PATH)->dict:
    obj=json.loads(path.read_text())
    req={'version','feature_contract','scaler_mean','scaler_scale','ridge_coef','ridge_intercept','selector_rule'}
    miss=req-set(obj)
    if miss: raise RuntimeError(f'QB distribution-state artifact missing {sorted(miss)}')
    n=len(obj['feature_contract'])
    if not (len(obj['scaler_mean'])==len(obj['scaler_scale'])==len(obj['ridge_coef'])==n==6):
        raise RuntimeError('QB distribution-state artifact dimension mismatch')
    return obj


def predict_delta(features:dict,artifact:dict|None=None)->float:
    art=artifact or load_artifact(); names=art['feature_contract']
    x=np.array([float(features.get(n,np.nan)) for n in names],dtype=float)
    if not np.isfinite(x).all():
        bad=[names[i] for i,v in enumerate(x) if not np.isfinite(v)]
        raise RuntimeError(f'QB distribution-state features non-finite: {bad}')
    mean=np.asarray(art['scaler_mean'],float); scale=np.asarray(art['scaler_scale'],float); coef=np.asarray(art['ridge_coef'],float)
    z=(x-mean)/np.where(np.abs(scale)>1e-12,scale,1.0)
    return float(art['ridge_intercept']+np.dot(z,coef))


def select_c2(features:dict,artifact:dict|None=None)->tuple[bool,float,str]:
    art=artifact or load_artifact(); delta=predict_delta(features,art)
    return bool(delta>0.0),delta,str(art['version'])
