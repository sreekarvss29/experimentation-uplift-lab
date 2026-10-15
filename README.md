# Experimentation & Uplift Lab

Four experimentation techniques that product data science teams use, applied to a real randomized
experiment: **CUPED/CUPAC variance reduction, sequential testing, multi-armed bandits, and uplift
modeling**. Every number below comes from `python run_all.py` and is saved in `results/metrics.json`.

## Data

[Hillstrom MineThatData e-mail experiment](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html):
64,000 customers randomly assigned to a **Men's e-mail**, **Women's e-mail**, or **No e-mail** group
(about 21.3K each). Outcomes were measured over the following two weeks: site visit, conversion, and spend.
Pre-treatment features include recency, past-year spend, product lines bought, zip-code type,
new-customer flag, and purchase channel. `run_all.py` downloads the file on first run.

| Arm | Visit rate | Conversion rate | Spend per customer |
|---|---|---|---|
| Men's e-mail | 18.3% | 1.25% | $1.42 |
| Women's e-mail | 15.1% | 0.88% | $1.08 |
| No e-mail | 10.6% | 0.57% | $0.65 |

## Results

### 1. CUPED / CUPAC variance reduction (`src/lab/cuped.py`)
Treatment effect of the Men's e-mail on visit rate: **+7.66 pp** (95% CI 6.99 to 8.32).

| Estimator | Variance reduction |
|---|---|
| CUPED, past-year spend as covariate | 0.4% |
| CUPAC, cross-fitted gradient-boosting prediction from all pre-treatment features | 2.5% |

A 1,000-rep simulation at 250 customers per arm confirmed both estimators keep the A/A
false-positive rate at about 5%. Power rose only from 69.1% to 70.0%.

**Takeaway:** CUPED is only as good as its covariate. This dataset has no pre-period *visit* history,
and past-year spend is barely correlated with a two-week visit, so the gain is small. With a lagged
version of the outcome itself, reductions of 20–50% are typical. CUPAC recovers more because it
combines every pre-treatment signal.

### 2. Sequential testing (`src/lab/sequential.py`)
Men's vs No e-mail on **conversion** (0.57% → 1.25%). The fixed-horizon design needs **3,063 customers
per arm** for 80% power. There are five equally spaced interim looks, and each design was run on
3,000 simulated experiments drawn from the real arms.

| Design | False-positive rate (A/A) | Power | Avg sample vs fixed (true effect) |
|---|---|---|---|
| Fixed horizon | 5.1% | 80.9% | 100% |
| Naive peeking (stop at first p < 0.05) | **13.2%** | 85.1% | 58% |
| O'Brien-Fleming boundaries (c = 2.04) | **4.9%** | 80.0% | **80%** |
| mSPRT (always valid) | 0.5% | 43.6% | 89% |

**Takeaway:** Peeking more than doubles false positives. O'Brien-Fleming keeps alpha at 5% with
essentially unchanged power, and it stops early often enough to save about **20% of the sample**.
mSPRT is valid at any number of looks, but with the sample truncated at the fixed-horizon size it
is very conservative.

### 3. Multi-armed bandits (`src/lab/bandits.py`)
The three campaigns are the arms. Each reward is a real customer's conversion, drawn from the arm
that was chosen. Each run allocates 30,000 customers in batches of 100, and results are averaged
over 300 runs.

| Policy | Conversions per 30K | vs fixed split | Traffic to best arm | Regret vs oracle |
|---|---|---|---|---|
| Fixed 1/3 split (A/B/n test) | 272.5 | — | 33% | 103.4 |
| Epsilon-greedy (ε = 0.1) | 359.4 | +31.9% | 87% | 16.6 |
| Thompson sampling (Beta-Bernoulli) | 354.7 | **+30.1%** | 84% | 21.3 |

**Takeaway:** When arms differ this much (the best arm converts at 2.2× control), adaptive
allocation recovers about 80% of the conversions a fixed split gives up. The trade-off is
precision: a bandit gives less clean effect estimates on the losing arms than a fixed A/B test.

### 4. Uplift modeling (`src/lab/uplift.py`)
Women's e-mail vs No e-mail, outcome = visit (ATE +4.52 pp). Models were evaluated on 20 repeated
stratified 70/30 splits; values are mean ± standard error.

| Targeting model | Qini AUC (% of total lift) | Lift captured, top 30% | Lift captured, top 50% |
|---|---|---|---|
| **X-learner** | **16.6 ± 0.8** | **49%** | **83%** |
| S-learner | 16.2 ± 0.6 | 47% | 82% |
| T-learner | 15.5 ± 0.7 | 49% | 81% |
| Response model, P(visit \| e-mailed) | 9.5 ± 0.6 | 45% | 66% |
| Random | −0.4 ± 0.8 | 31% | 51% |

**Takeaway:** E-mailing the top half of customers ranked by the X-learner captures **83% of the
campaign's incremental visits with half the sends**, versus 66% when ranking by response
likelihood. Response models target people who would visit anyway.

**Counter-example (also in `metrics.json`):** On the Men's e-mail arm, the response model beat
every uplift learner (Qini 4.2 vs ≤ 2.2). There, uplift is concentrated among already-likely
visitors, so ranking by response is a good proxy. Uplift modeling pays off when treatment effects
are heterogeneous in ways that baseline propensity doesn't capture, and this check is how you find out.

## Figures
`figures/cuped_ci.png`, `figures/sequential.png`, `figures/bandits.png`, `figures/qini.png`

## Run it
```bash
pip install -r requirements.txt
python run_all.py          # about 1–2 minutes; writes results/ and figures/
python run_all.py --quick  # smoke test with fewer simulation reps
```
All randomness is seeded (`SEED = 42` in `run_all.py`), so reruns reproduce the numbers above.

## Methods notes
- **CUPAC covariate:** 5-fold cross-fitted predictions from pre-treatment features only, never the
  treatment flag, so the adjusted estimator stays unbiased.
- **O'Brien-Fleming:** the constant is calibrated by Brownian-motion simulation to an overall
  two-sided α of 0.05 for 5 looks.
- **mSPRT:** normal-mixture likelihood ratio (Johari et al., 2017), with mixture variance set to the
  squared observed effect.
- **Qini:** Radcliffe's Qini curve; the AUC is the area above the random-targeting line, normalized by
  total incremental outcome so it's comparable across splits.
- **Uplift learners:** heavily regularized gradient boosting (7 leaves, at least 1,000 samples per
  leaf), because the uplift signal is small relative to outcome noise.

## References
Deng et al. (2013), *Improving the Sensitivity of Online Controlled Experiments by Utilizing
Pre-Experiment Data* · Tang et al. (2020), *CUPAC* (DoorDash) · Jennison & Turnbull (2000), *Group
Sequential Methods* · Johari et al. (2017), *Peeking at A/B Tests* · Russo et al. (2018), *A
Tutorial on Thompson Sampling* · Künzel et al. (2019), *Metalearners for Estimating Heterogeneous
Treatment Effects* · Radcliffe (2007), *Using Control Groups to Target on Predicted Lift*.
