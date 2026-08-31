"""M95D authoritative estimator-alignment shim.

Keeps all M95D structural features, interactions, temporal splits and gates
unchanged. The prior control-alignment run restored M95C's exact feature list
but exposed one remaining control mismatch: M95D's base evaluator used missing-
value indicator columns whereas M95C/M95B did not. This shim restores the exact
M95C estimator pipeline so the M95C environment control must reproduce before
M95D is interpreted.
"""
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import scripts.backtest.evaluate_rb_rushing_environment_scheme_v3 as v3  # noqa:F401
import scripts.backtest.evaluate_rb_rushing_environment_scheme as m


def ridge_exact():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=20.0)),
    ])


def logit_exact():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=.20,max_iter=2500,random_state=95)),
    ])

m.ridge = ridge_exact
m.logit = logit_exact

if __name__ == "__main__":
    raise SystemExit(m.main())
