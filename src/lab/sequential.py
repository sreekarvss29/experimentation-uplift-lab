"""Sequential testing: why naive peeking fails, and two valid alternatives.

Simulates experiments that stream real customers from the Hillstrom arms and are analyzed at
K equally spaced interim looks:
  * naive peeking   - stop at the first look with |z| > 1.96 (inflates false positives)
  * O'Brien-Fleming - group-sequential boundary c * sqrt(K / k); c is calibrated by simulation
                      so the overall two-sided alpha is 0.05 (c ~ 2.04 for K = 5)
  * mSPRT           - always-valid mixture SPRT (Johari et al., 2017); reject when the
                      likelihood ratio exceeds 1 / alpha, valid at any number of looks
"""
import numpy as np
from scipy import stats


def required_n_per_arm(p0, p1, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(np.ceil((za + zb) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p1 - p0) ** 2))


def _z_path(y1, y0, looks):
    """z statistics at each look from cumulative sums (pooled-variance two-proportion test)."""
    c1, c0 = np.cumsum(y1), np.cumsum(y0)
    zs = []
    for n in looks:
        p1, p0 = c1[n - 1] / n, c0[n - 1] / n
        p = (c1[n - 1] + c0[n - 1]) / (2 * n)
        se = np.sqrt(max(p * (1 - p), 1e-12) * 2 / n)
        zs.append((p1 - p0) / se)
    return np.array(zs)


def _msprt_path(y1, y0, looks, tau2, alpha=0.05):
    """Normal-mixture mSPRT on the difference in means; returns True at looks where it rejects."""
    c1, c0 = np.cumsum(y1), np.cumsum(y0)
    out = []
    for n in looks:
        p1, p0 = c1[n - 1] / n, c0[n - 1] / n
        p = (c1[n - 1] + c0[n - 1]) / (2 * n)
        sigma2 = max(2 * p * (1 - p), 1e-12)  # variance of a paired difference
        delta = p1 - p0
        v = sigma2 / n
        lr = np.sqrt(v / (v + tau2)) * np.exp(tau2 * delta ** 2 / (2 * v * (v + tau2)))
        out.append(lr >= 1 / alpha)
    return np.array(out)


def calibrate_obf(K=5, alpha=0.05, sims=200_000, seed=1):
    """Find c so that P(any |Z_k| > c*sqrt(K/k)) = alpha under H0 (Brownian-motion simulation)."""
    rng = np.random.default_rng(seed)
    incr = rng.standard_normal((sims, K))
    z = np.cumsum(incr, axis=1) / np.sqrt(np.arange(1, K + 1))
    scale = np.sqrt(K / np.arange(1, K + 1))
    lo, hi = 1.5, 4.0
    for _ in range(50):
        c = (lo + hi) / 2
        rate = np.mean(np.any(np.abs(z) > c * scale, axis=1))
        lo, hi = (c, hi) if rate > alpha else (lo, c)
    return (lo + hi) / 2


def simulate(df, outcome="conversion", K=5, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = df[outcome].values
    y1_pool, y0_pool = y[df["treated"].values == 1], y[df["treated"].values == 0]
    p1, p0 = y1_pool.mean(), y0_pool.mean()
    n_max = required_n_per_arm(p0, p1)
    looks = np.linspace(n_max / K, n_max, K).round().astype(int)
    c_obf = calibrate_obf(K)
    obf_bounds = c_obf * np.sqrt(K / np.arange(1, K + 1))
    tau2 = (p1 - p0) ** 2  # mixture variance set to the scale of plausible effects

    def run(h1: bool):
        stats_ = {m: {"reject": 0, "n_used": 0} for m in ["fixed", "peeking", "obf", "msprt"]}
        for _ in range(reps):
            a = rng.choice(y1_pool if h1 else y0_pool, n_max, replace=True)
            b = rng.choice(y0_pool, n_max, replace=True)
            z = _z_path(a, b, looks)
            # fixed horizon: single test at n_max
            stats_["fixed"]["reject"] += abs(z[-1]) > 1.96
            stats_["fixed"]["n_used"] += n_max
            for name, hits in [("peeking", np.abs(z) > 1.96),
                               ("obf", np.abs(z) > obf_bounds),
                               ("msprt", _msprt_path(a, b, looks, tau2))]:
                k = np.argmax(hits) if hits.any() else K - 1
                stats_[name]["reject"] += bool(hits.any())
                stats_[name]["n_used"] += looks[k]
        return {m: {"reject_rate": v["reject"] / reps, "avg_n_per_arm": v["n_used"] / reps,
                    "avg_n_vs_fixed": v["n_used"] / reps / n_max} for m, v in stats_.items()}

    return {
        "outcome": outcome, "p_control": p0, "p_treatment": p1, "n_max_per_arm": n_max,
        "looks": looks.tolist(), "obf_c": c_obf, "reps": reps,
        "null": run(False), "alternative": run(True),
    }
