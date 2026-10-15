"""Run every analysis, write results/metrics.json and figures/*.png.

Usage:  python run_all.py            (full run, ~a few minutes)
        python run_all.py --quick    (fewer simulation reps, for a smoke test)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from lab import bandits, cuped, data, sequential, uplift  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
SEED = 42


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def main(quick=False):
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    df = data.load()
    mens = data.two_arm(df, "Mens E-Mail", "No E-Mail")
    out = {"dataset": {"rows": len(df), "arms": df["segment"].value_counts().to_dict()}}

    t0 = time.time()
    # 1. CUPED / CUPAC
    out["cuped"] = {o: cuped.analyze(mens, o, seed=SEED) for o in ["visit", "spend"]}
    out["cuped_power"] = cuped.power_simulation(mens, "visit", n_per_arm=250,
                                                reps=200 if quick else 1000, seed=SEED)
    print(f"CUPED done ({time.time()-t0:.0f}s)")

    # 2. Sequential testing
    out["sequential"] = sequential.simulate(mens, "conversion", K=5, reps=300 if quick else 3000, seed=SEED)
    print(f"Sequential done ({time.time()-t0:.0f}s)")

    # 3. Bandits
    out["bandits"] = bandits.simulate(df, "conversion", horizon=30_000, reps=40 if quick else 300, seed=SEED)
    print(f"Bandits done ({time.time()-t0:.0f}s)")

    # 4. Uplift
    womens = data.two_arm(df, "Womens E-Mail", "No E-Mail")
    out["uplift"], curves = uplift.evaluate(womens, "visit", n_splits=8 if quick else 20, seed=SEED)
    out["uplift_mens_counterexample"], _ = uplift.evaluate(mens, "visit", n_splits=8 if quick else 20, seed=SEED)
    print(f"Uplift done ({time.time()-t0:.0f}s)")

    (RES / "metrics.json").write_text(json.dumps(to_jsonable(out), indent=2))
    make_figures(out, curves)
    print("Wrote results/metrics.json and figures/")


def make_figures(out, curves):
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

    # CUPED: CI comparison for visit
    c = out["cuped"]["visit"]
    names = {"naive": "Difference in means", "cuped_history": "CUPED (past-year spend)", "cupac": "CUPAC (ML covariate)"}
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for i, (k, lab) in enumerate(names.items()):
        lo, hi = c[k]["ci95"]
        ax.errorbar(c[k]["estimate"] * 100, i, xerr=[[(c[k]["estimate"] - lo) * 100], [(hi - c[k]["estimate"]) * 100]],
                    fmt="o", capsize=5, color="#1F3A5F")
        ax.text(hi * 100 + 0.05, i, f"variance -{c[k]['variance_reduction']*100:.1f}%", va="center", fontsize=8)
    ax.set_yticks(range(3), names.values()); ax.invert_yaxis()
    ax.set_xlabel("Treatment effect on visit rate (pp), 95% CI")
    ax.set_title("Variance reduction: Mens e-mail vs no e-mail")
    fig.tight_layout(); fig.savefig(FIG / "cuped_ci.png"); plt.close(fig)

    # Sequential: false positives and sample size
    s = out["sequential"]
    methods = ["fixed", "peeking", "obf", "msprt"]
    labels = ["Fixed horizon", "Naive peeking", "O'Brien-Fleming", "mSPRT"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].bar(labels, [s["null"][m]["reject_rate"] * 100 for m in methods], color=["#7A8CA5", "#C2410C", "#1F3A5F", "#1F3A5F"])
    axes[0].axhline(5, ls="--", c="k", lw=0.8); axes[0].set_ylabel("False-positive rate (%)"); axes[0].set_title("A/A tests (no true effect)")
    axes[1].bar(labels, [s["alternative"][m]["avg_n_vs_fixed"] * 100 for m in methods], color=["#7A8CA5", "#C2410C", "#1F3A5F", "#1F3A5F"])
    axes[1].set_ylabel("Avg sample used (% of fixed)"); axes[1].set_title("Real effect present")
    for a in axes:
        a.tick_params(axis="x", labelsize=8, rotation=15)
    fig.tight_layout(); fig.savefig(FIG / "sequential.png"); plt.close(fig)

    # Bandits
    b = out["bandits"]["policies"]
    fig, ax = plt.subplots(figsize=(6, 3))
    pol = ["fixed", "eps_greedy", "thompson"]
    ax.bar(["Fixed 1/3 split", "Epsilon-greedy", "Thompson sampling"], [b[p]["mean_total_reward"] for p in pol],
           yerr=[b[p]["sd_total_reward"] for p in pol], color=["#7A8CA5", "#7A8CA5", "#1F3A5F"], capsize=4)
    ax.set_ylabel(f"Conversions per {out['bandits']['horizon']:,} customers")
    ax.set_title("Bandit allocation vs fixed split")
    fig.tight_layout(); fig.savefig(FIG / "bandits.png"); plt.close(fig)

    # Uplift Qini curves
    fig, ax = plt.subplots(figsize=(6.5, 4))
    style = {"X-learner": "#1F3A5F", "T-learner": "#3D6A9E", "S-learner": "#7FA3C8", "Response model": "#C2410C", "Random": "#999999"}
    for name, (f, q) in curves.items():
        ax.plot(np.array(f) * 100, np.array(q) * 100,
                label=f"{name} (Qini {out['uplift']['models'][name]['qini_auc_pct']['mean']:.1f})",
                color=style[name], lw=2 if "learner" in name else 1.4, ls="--" if name == "Random" else "-")
    ax.set_xlabel("% of customers targeted (ranked by model)"); ax.set_ylabel("% of all incremental visits captured")
    ax.set_title(f"Qini curves, Women's e-mail (mean of {out['uplift']['n_splits']} test splits)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "qini.png"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
