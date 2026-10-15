"""Uplift modeling (heterogeneous treatment effects) with Qini evaluation.

Learners: S-learner, T-learner, X-learner (gradient boosting base models), plus two baselines:
a "response model" that ranks by P(outcome | treated) (what most teams ship), and random targeting.
All evaluation is on a held-out, stratified test split.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from .data import feature_matrix


def _clf(seed):
    # Strong regularization: uplift signals are small relative to outcome noise.
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.03, max_leaf_nodes=7,
                                          min_samples_leaf=1000, random_state=seed)


def _reg(seed):
    return HistGradientBoostingRegressor(max_iter=100, learning_rate=0.03, max_leaf_nodes=7,
                                         min_samples_leaf=1000, random_state=seed)


def fit_predict(Xtr, ttr, ytr, Xte, seed=0):
    preds = {}
    # S-learner: one model with treatment as a feature
    s = _clf(seed).fit(np.c_[Xtr, ttr], ytr)
    preds["S-learner"] = s.predict_proba(np.c_[Xte, np.ones(len(Xte))])[:, 1] - \
        s.predict_proba(np.c_[Xte, np.zeros(len(Xte))])[:, 1]
    # T-learner: separate outcome models per arm
    m1 = _clf(seed).fit(Xtr[ttr == 1], ytr[ttr == 1])
    m0 = _clf(seed).fit(Xtr[ttr == 0], ytr[ttr == 0])
    mu1_te, mu0_te = m1.predict_proba(Xte)[:, 1], m0.predict_proba(Xte)[:, 1]
    preds["T-learner"] = mu1_te - mu0_te
    # X-learner (Kunzel et al., 2019)
    d1 = ytr[ttr == 1] - m0.predict_proba(Xtr[ttr == 1])[:, 1]
    d0 = m1.predict_proba(Xtr[ttr == 0])[:, 1] - ytr[ttr == 0]
    tau1 = _reg(seed).fit(Xtr[ttr == 1], d1)
    tau0 = _reg(seed).fit(Xtr[ttr == 0], d0)
    g = ttr.mean()  # randomized experiment: propensity is constant
    preds["X-learner"] = g * tau0.predict(Xte) + (1 - g) * tau1.predict(Xte)
    # Baselines
    preds["Response model"] = mu1_te
    preds["Random"] = np.random.default_rng(seed).random(len(Xte))
    return preds


def qini_curve(score, y, t, n_points=100):
    """Qini curve: incremental outcomes vs fraction targeted (Radcliffe, 2007).

    Q(phi) = Y_t(phi) - Y_c(phi) * N_t(phi) / N_c(phi), computed on the top-phi fraction by score.
    """
    order = np.argsort(-score, kind="stable")
    y, t = y[order], t[order]
    ct, cc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    q = ct - cc * np.divide(nt, nc, out=np.zeros_like(nt, dtype=float), where=nc > 0)
    idx = np.linspace(0, len(y) - 1, n_points).astype(int)
    frac = (idx + 1) / len(y)
    return np.r_[0, frac], np.r_[0, q[idx]]


def qini_auc(frac, q):
    """Area between the Qini curve and the random-targeting line (normalized by population)."""
    random_line = frac * q[-1]
    trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2.x name, with 1.x fallback
    return float(trapz(q - random_line, frac))


def evaluate(df, outcome="visit", n_splits=20, test_size=0.3, seed=0):
    """Repeated stratified train/test splits; report mean and standard error across splits.

    qini_auc is normalized as a percentage of the total incremental outcome on each test set,
    so it is comparable across splits. 0 = random targeting.
    """
    X = feature_matrix(df).values
    y = df[outcome].values.astype(int)
    t = df["treated"].values.astype(int)
    per = {}
    curves = {}
    for k in range(n_splits):
        s = seed + k
        Xtr, Xte, ytr, yte, ttr, tte = train_test_split(X, y, t, test_size=test_size,
                                                        random_state=s, stratify=t * 2 + y)
        ate = yte[tte == 1].mean() - yte[tte == 0].mean()
        for name, score in fit_predict(Xtr, ttr, ytr, Xte, s).items():
            frac, q = qini_curve(score, yte, tte)
            r = per.setdefault(name, {})
            r.setdefault("qini_auc_pct", []).append(qini_auc(frac, q) / q[-1] * 100)
            for pct in (0.2, 0.3, 0.5):
                r.setdefault(f"share_of_incremental_at_top_{int(pct*100)}pct", []).append(q[np.searchsorted(frac, pct)] / q[-1])
            top = np.argsort(-score)[: len(score) // 10]
            r.setdefault("top_decile_uplift_vs_ate", []).append(
                (yte[top][tte[top] == 1].mean() - yte[top][tte[top] == 0].mean()) / ate)
            curves.setdefault(name, []).append(q / q[-1])
    out = {"outcome": outcome, "n_splits": n_splits, "n_test": int(len(y) * test_size),
           "ate": float(y[t == 1].mean() - y[t == 0].mean()), "models": {}}
    for name, r in per.items():
        out["models"][name] = {k: {"mean": float(np.mean(v)), "se": float(np.std(v, ddof=1) / np.sqrt(len(v)))}
                               for k, v in r.items()}
    mean_curves = {n: (frac.tolist(), np.mean(c, axis=0).tolist()) for n, c in curves.items()}
    return out, mean_curves
