"""CUPED and CUPAC variance reduction.

CUPED (Deng et al., 2013): Y_adj = Y - theta * (X - mean(X)), theta = cov(Y, X) / var(X),
using a pre-experiment covariate X that is independent of treatment assignment.

CUPAC (DoorDash, 2020): use a cross-fitted ML prediction of Y from pre-treatment features as
the covariate. Predictions are made out-of-fold and never use the treatment indicator, so the
covariate stays independent of assignment and the estimator stays unbiased.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from .data import feature_matrix


def diff_in_means(y, t):
    y1, y0 = y[t == 1], y[t == 0]
    est = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return est, se


def cuped_adjust(y, x):
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean()), theta


def cupac_covariate(df: pd.DataFrame, outcome: str, seed: int = 0, n_splits: int = 5):
    X = feature_matrix(df).values
    y = df[outcome].values.astype(float)
    pred = np.zeros(len(y))
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=seed)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def analyze(df: pd.DataFrame, outcome: str, seed: int = 0) -> dict:
    """Compare naive, CUPED (history), and CUPAC estimates on the full two-arm sample."""
    y = df[outcome].values.astype(float)
    t = df["treated"].values
    rows = {}

    est, se = diff_in_means(y, t)
    rows["naive"] = dict(estimate=est, se=se)

    y_cuped, theta = cuped_adjust(y, df["history"].values.astype(float))
    est, se = diff_in_means(y_cuped, t)
    rows["cuped_history"] = dict(estimate=est, se=se, theta=theta)

    cov = cupac_covariate(df, outcome, seed=seed)
    y_cupac, theta = cuped_adjust(y, cov)
    est, se = diff_in_means(y_cupac, t)
    rows["cupac"] = dict(estimate=est, se=se, theta=theta,
                         corr_with_outcome=float(np.corrcoef(cov, y)[0, 1]))

    base_var = rows["naive"]["se"] ** 2
    for k, r in rows.items():
        r["variance_reduction"] = 1 - r["se"] ** 2 / base_var
        # Sample size needed for the same power scales with variance.
        r["sample_size_reduction"] = r["variance_reduction"]
        r["ci95"] = (r["estimate"] - 1.96 * r["se"], r["estimate"] + 1.96 * r["se"])
        r["p_value"] = float(2 * stats.norm.sf(abs(r["estimate"] / r["se"])))
    return rows


def power_simulation(df: pd.DataFrame, outcome: str, n_per_arm: int, reps: int = 1000, seed: int = 0) -> dict:
    """Empirical power and A/A false-positive rate at a fixed sample size, naive vs CUPAC.

    Each rep draws n_per_arm customers per arm without replacement. The CUPAC covariate is
    computed once, cross-fitted on the full pooled sample without the treatment indicator.
    """
    rng = np.random.default_rng(seed)
    cov = cupac_covariate(df, outcome, seed=seed)
    y = df[outcome].values.astype(float)
    t = df["treated"].values
    idx1, idx0 = np.where(t == 1)[0], np.where(t == 0)[0]

    def test(yv, cv, tv, adjust):
        if adjust:
            yv, _ = cuped_adjust(yv, cv)
        est, se = diff_in_means(yv, tv)
        return abs(est / se) > 1.96

    res = {"naive_power": 0, "cupac_power": 0, "naive_aa_fpr": 0, "cupac_aa_fpr": 0}
    tv = np.r_[np.ones(n_per_arm, int), np.zeros(n_per_arm, int)]
    for _ in range(reps):
        s = np.r_[rng.choice(idx1, n_per_arm, replace=False), rng.choice(idx0, n_per_arm, replace=False)]
        res["naive_power"] += test(y[s], cov[s], tv, False)
        res["cupac_power"] += test(y[s], cov[s], tv, True)
        # A/A: both "arms" drawn from control only.
        a = rng.choice(idx0, 2 * n_per_arm, replace=False)
        res["naive_aa_fpr"] += test(y[a], cov[a], tv, False)
        res["cupac_aa_fpr"] += test(y[a], cov[a], tv, True)
    return {k: v / reps for k, v in res.items()} | {"n_per_arm": n_per_arm, "reps": reps}
