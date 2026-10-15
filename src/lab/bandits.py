"""Multi-armed bandits vs fixed-split A/B/n testing, replayed on real customer outcomes.

Arms = the three Hillstrom campaigns. When a policy picks an arm, the reward is the outcome of
a real customer drawn at random from that arm, so arm reward distributions are the empirical ones.
"""
import numpy as np

ARMS = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]


def _policy_run(pools, horizon, policy, rng, eps=0.1, batch=100):
    k = len(pools)
    succ, pulls = np.zeros(k), np.zeros(k)
    total = 0.0
    for start in range(0, horizon, batch):
        n = min(batch, horizon - start)
        if policy == "fixed":
            arms = np.arange(start, start + n) % k
        elif policy == "thompson":
            draws = rng.beta(1 + succ[:, None], 1 + pulls[:, None] - succ[:, None], size=(k, n))
            arms = draws.argmax(axis=0)
        elif policy == "eps_greedy":
            means = np.divide(succ, pulls, out=np.full(k, np.inf), where=pulls > 0)
            arms = np.where(rng.random(n) < eps, rng.integers(0, k, n), np.argmax(means))
        else:
            raise ValueError(policy)
        for a in range(k):
            m = int((arms == a).sum())
            if m:
                r = rng.choice(pools[a], m, replace=True)
                succ[a] += r.sum(); pulls[a] += m; total += r.sum()
    return total, pulls


def simulate(df, outcome="conversion", horizon=30_000, reps=300, seed=0):
    rng = np.random.default_rng(seed)
    pools = [df.loc[df["segment"] == a, outcome].values.astype(float) for a in ARMS]
    means = np.array([p.mean() for p in pools])
    best = int(means.argmax())
    out = {"outcome": outcome, "arm_means": dict(zip(ARMS, means.round(5).tolist())),
           "horizon": horizon, "reps": reps, "policies": {}}
    for pol in ["fixed", "eps_greedy", "thompson"]:
        totals, best_share = [], []
        for _ in range(reps):
            tot, pulls = _policy_run(pools, horizon, pol, rng)
            totals.append(tot); best_share.append(pulls[best] / horizon)
        totals = np.array(totals)
        out["policies"][pol] = {
            "mean_total_reward": float(totals.mean()),
            "sd_total_reward": float(totals.std(ddof=1)),
            "regret_vs_oracle": float(horizon * means[best] - totals.mean()),
            "share_traffic_to_best_arm": float(np.mean(best_share)),
        }
    f = out["policies"]["fixed"]["mean_total_reward"]
    for pol, r in out["policies"].items():
        r["lift_vs_fixed"] = r["mean_total_reward"] / f - 1
    return out
