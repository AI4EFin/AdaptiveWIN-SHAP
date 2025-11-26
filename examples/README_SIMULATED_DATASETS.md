# Simulated Datasets for AdaptiveWIN-SHAP Benchmarking

This directory contains scripts and datasets for benchmarking AdaptiveWIN-SHAP methods on various simulated time series with different non-stationary characteristics.

## Available Datasets

All datasets have 1500 time points with piecewise stationary regimes or smoothly varying parameters:

### 1. Piecewise AR(3) - Rotating Dominant Lag (Baseline)

**Description:** Pure univariate AR(3) process with regime-dependent dominant lag.

**Mathematical Formulation:**

$$X_t = \phi_1^{(k)} X_{t-1} + \phi_2^{(k)} X_{t-2} + \phi_3^{(k)} X_{t-3} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)$$

**Regime-specific coefficients:**
- Regime 1 (t=1..500): $\boldsymbol{\phi}^{(1)} = [0.9, 0.01, 0.01]$ (lag-1 dominates)
- Regime 2 (t=501..1000): $\boldsymbol{\phi}^{(2)} = [0.01, 0.9, 0.01]$ (lag-2 dominates)
- Regime 3 (t=1001..1500): $\boldsymbol{\phi}^{(3)} = [0.01, 0.01, 0.9]$ (lag-3 dominates)

**True importances:** 3 lag features, normalized as $\text{Imp}_j^{(k)} = |\phi_j^{(k)}| / \sum_{i=1}^3 |\phi_i^{(k)}|$

---

### 2. ARX - Rotating Covariate Drivers

**Description:** ARX model with 3 covariates (demand D, fuel F, renewables R) and regime-dependent covariate dominance.

**Mathematical Formulation:**
$$Y_t = \sum_{j=1}^3 \phi_j Y_{t-j} + \beta_1^{(k)} D_t + \beta_2^{(k)} F_t + \beta_3^{(k)} R_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)$$

**Coefficients:**
- AR lags: $\boldsymbol{\phi} = [0.6, 0.2, 0.1]$ (fixed across regimes)
- Regime 1 (t=1..500): $\boldsymbol{\beta}^{(1)} = [0.2, 1.2, -0.1]$ (fuel dominates)
- Regime 2 (t=501..1000): $\boldsymbol{\beta}^{(2)} = [1.2, 0.2, -0.1]$ (demand dominates)
- Regime 3 (t=1001..1500): $\boldsymbol{\beta}^{(3)} = [0.2, 0.1, -1.2]$ (renewables dominate)

**Covariates:**
- $D_t \sim \mathcal{N}(0, 1) + 0.3\sin(2\pi t/24)$ (demand with daily cycle)
- $F_t \sim \mathcal{N}(0, 1) + 0.1t/T$ (fuel with slight trend)
- $R_t \sim \mathcal{N}(0, 1) + 0.5\sin(2\pi t/168)$ (renewables with weekly cycle)

**True importances:** 6 features (3 lags + 3 covariates), standardized by feature scales

---

### 3. Trend + Seasonality + AR Break

**Description:** Deterministic trend and multi-scale seasonality with rotating AR coefficients.

**Mathematical Formulation:**
$$Y_t = \mu_t + S_t + X_t$$
where:
- Trend: $\mu_t = 0.002t$
- Seasonality: $S_t = \sin(2\pi t/24) + 0.5\cos(2\pi t/24) + 0.6\sin(2\pi t/168)$
- AR(3) component: $X_t = \phi_1^{(k)} X_{t-1} + \phi_2^{(k)} X_{t-2} + \phi_3^{(k)} X_{t-3} + \varepsilon_t$

**Regime-specific AR coefficients:** Same as piecewise_ar3

**True importances:** 3 lag features from the AR component

---

### 4. Spike/Jump Process - Regime-Dependent Spike Drivers

**Description:** AR process with regime-dependent spike occurrence driven by different covariates.

**Mathematical Formulation:**
$$Y_t = X_t + J_t \cdot S_t$$
where:
- Base process: $X_t = 0.7 X_{t-1} + 0.2 X_{t-2} + 0.05 X_{t-3} + \varepsilon_t$
- Spike indicator: $J_t \sim \text{Bernoulli}(p_t)$ with $\text{logit}(p_t) = \gamma_1^{(k)} D_t + \gamma_2^{(k)} R_t - 2.0$
- Spike size: $S_t \sim \text{LogNormal}(1.0, 0.6)$

**Regime-specific spike drivers:**
- Regime 1 (t=1..750): $\boldsymbol{\gamma}^{(1)} = [1.5, 0.0]$ (demand-driven spikes)
- Regime 2 (t=751..1500): $\boldsymbol{\gamma}^{(2)} = [0.0, 1.5]$ (renewables-driven spikes)

**True importances:** 5 features (3 AR lags + 2 covariates through spike probability)

---

### 5. Time-Varying Parameter ARX (TVP-ARX)

**Description:** ARX model with smoothly time-varying coefficients following sinusoidal patterns.

**Mathematical Formulation:**
$$Y_t = \sum_{j=1}^3 \phi_j(t) Y_{t-j} + \beta_1(t) Z_{1,t} + \beta_2(t) Z_{2,t} + \varepsilon_t$$

**Time-varying coefficients:**
- $\phi_1(t) = 0.7 + 0.2\sin(2\pi t/T)$
- $\phi_2(t) = 0.1 + 0.1\cos(2\pi t/T)$
- $\phi_3(t) = 0.05$ (constant)
- $\beta_1(t) = 1.0 + 0.8\cos(2\pi t/T)$
- $\beta_2(t) = 0.2 + 0.8\sin(2\pi t/T)$

**Covariates:** $Z_{1,t}, Z_{2,t} \sim \mathcal{N}(0, 1)$ (i.i.d.)

**True importances:** 5 features (3 lags + 2 covariates), varying smoothly over time

---

### 6. GARCH Returns - Regime-Shifting Factor Loadings

**Description:** Asset returns with GARCH volatility and regime-dependent factor exposures.

**Mathematical Formulation:**
$$r_t = \beta_1^{(k)} M_t + \beta_2^{(k)} V_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, h_t)$$

**GARCH(1,1) volatility:**
$$h_t = \omega^{(k)} + \alpha^{(k)} \eta_{t-1}^2 + \beta^{(k)} h_{t-1}$$

**Regime-specific parameters:**
- Regime 1 (calm, t=1..750):
  - Factor loadings: $\boldsymbol{\beta}^{(1)} = [1.2, 0.2]$ (market-driven)
  - GARCH: $(\omega, \alpha, \beta) = (0.02, 0.05, 0.9)$
- Regime 2 (crisis, t=751..1500):
  - Factor loadings: $\boldsymbol{\beta}^{(2)} = [0.3, 1.2]$ (volatility-driven)
  - GARCH: $(\omega, \alpha, \beta) = (0.05, 0.12, 0.85)$

**Factors:** $M_t$ (market), $V_t$ (volatility risk) $\sim \mathcal{N}(0, 1)$

**True importances:** 2 factor features

---

### 7. Cointegration Break

**Description:** Multivariate cointegration with regime-dependent coupling strength.

**Mathematical Formulation:**
$$Y_t = \beta_1^{(k)} X_{1,t} + \beta_2^{(k)} X_{2,t} + \beta_3^{(k)} X_{3,t} + \beta_4^{(k)} X_{4,t} + u_t$$
$$u_t = \rho^{(k)} u_{t-1} + e_t, \quad e_t \sim \mathcal{N}(0, \sigma^2)$$

**Driving processes (random walks):**
- $X_{1,t} = X_{1,t-1} + \varepsilon_{1,t}$, $\varepsilon_{1,t} \sim \mathcal{N}(0, 0.25)$
- $X_{2,t} = X_{2,t-1} + \varepsilon_{2,t}$, $\varepsilon_{2,t} \sim \mathcal{N}(0, 0.20)$
- $X_{3,t}, X_{4,t} \sim \mathcal{N}(0, 0.5)$ (white noise, not cointegrated)

**Regime-specific coefficients:**
- Regime 1 (t=1..750): $\boldsymbol{\beta}^{(1)} = [1.2, 0.8, 0.05, -0.05]$, $\rho^{(1)} = 0.5$ (strong cointegration)
- Regime 2 (t=751..1500): $\boldsymbol{\beta}^{(2)} = [1.5, 0.2, 0.05, -0.05]$, $\rho^{(2)} = 0.98$ (weak cointegration)

**True importances:** 4 covariate features

## Evaluation Metrics

### SHAP Values

**Definition:** SHAP (SHapley Additive exPlanations) values represent the contribution of each feature to a prediction, based on cooperative game theory.

**Computation:** For a prediction $f(\mathbf{x})$, the SHAP value $\phi_j$ for feature $j$ satisfies:

$$f(\mathbf{x}) = f(\mathbb{E}[\mathbf{x}]) + \sum_{j=1}^d \phi_j$$

where $\mathbb{E}[\mathbf{x}]$ is the expected prediction over the background distribution.

**Kernel SHAP approximation:**
$$\phi_j = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|! (d - |S| - 1)!}{d!} [f(S \cup \{j\}) - f(S)]$$

where $\mathcal{F}$ is the set of all features, $S$ is a subset of features, and $f(S)$ is the model prediction with only features in $S$ present.

**Normalization:** SHAP values are normalized to importance scores:
$$\text{Imp}_j = \frac{|\phi_j|}{\sum_{i=1}^d |\phi_i|}$$

This ensures $\sum_{j=1}^d \text{Imp}_j = 1$ and all importances are non-negative.

---

### True Importance (Ground Truth)

**Definition:** For simulated datasets where true data-generating process is known, true importance is computed from the standardized absolute coefficients.

**For AR/ARX models with coefficients $\boldsymbol{\theta} = [\phi_1, \ldots, \phi_p, \beta_1, \ldots, \beta_k]$:**

$$\text{TrueImp}_j = \frac{|\theta_j| \cdot \sigma_j}{\sum_{i=1}^{p+k} |\theta_i| \cdot \sigma_i}$$

where:
- $\theta_j$ is the coefficient for feature $j$
- $\sigma_j$ is the standard deviation of feature $j$ (1 for lags, empirical std for covariates)
- $p$ is the number of AR lags
- $k$ is the number of covariates

**Properties:**
- Accounts for feature scaling (standardized importance)
- Sums to 1: $\sum_{j=1}^{p+k} \text{TrueImp}_j = 1$
- Varies over time for regime-switching and time-varying models

---

### Faithfulness Metrics

**Purpose:** Measure how well SHAP values reflect the model's actual decision-making by perturbing important features and measuring prediction change.

#### Perturbation Analysis (PRTB)

**Method:** Flip the values of the top-$q$% most important features and measure prediction change.

**Perturbation operation:**
$$x'_{t,j} = \max_t(x_{t,j}) - x_{t,j} \quad \text{for top-}q\% \text{ features}$$

**Score:**
$$\text{Faithfulness}_{\text{PRTB}} = \frac{1}{N} \sum_{i=1}^N |f(\mathbf{x}_i) - f(\mathbf{x}'_i)|$$

**Interpretation:** Higher is better — larger prediction change indicates SHAP correctly identified important features.

#### Sequence Analysis (SQNC)

**Method:** Zero out sequences starting from the top-$q$% most important time steps.

**Perturbation operation:**
$$x'_{t:t+L,j} = 0 \quad \text{for top-}q\% \text{ features, sequence length } L$$

**Score:**
$$\text{Faithfulness}_{\text{SQNC}} = \frac{1}{N} \sum_{i=1}^N |f(\mathbf{x}_i) - f(\mathbf{x}'_i)|$$

**Parameters:** Typically evaluated at percentiles $q \in \{50, 70, 90\}$ (top 50%, 30%, 10%)

---

### Ablation Metrics

**Purpose:** Systematically remove features in order of importance and measure cumulative prediction degradation. Considered one of the most rigorous XAI evaluation metrics.

#### Most Important First (MIF)

**Method:** Remove features in decreasing order of SHAP importance.

**Procedure:**
1. Sort features by $|\phi_j|$ in descending order: $j_1, j_2, \ldots, j_d$
2. For $k = 1, \ldots, \lceil (1-q/100) \cdot d \rceil$:
   - Set $x_{j_k} = 0$ (ablate feature $j_k$)
   - Compute $\Delta_k = |f(\mathbf{x}) - f(\mathbf{x}^{(k)})|$
3. Average degradation: $\text{Ablation}_{\text{MIF}} = \frac{1}{K} \sum_{k=1}^K \Delta_k$

**Interpretation:** Higher is better — removing important features first should cause larger prediction changes.

#### Least Important First (LIF)

**Method:** Remove features in ascending order of SHAP importance.

**Procedure:** Same as MIF but sort in ascending order of $|\phi_j|$

**Interpretation:** Lower is better — removing unimportant features first should cause smaller prediction changes.

**Expected relationship:** $\text{Ablation}_{\text{MIF}} > \text{Ablation}_{\text{LIF}}$

Good explanations show a large gap between MIF and LIF, indicating clear distinction between important and unimportant features.

---

### Evaluation Summary

| Metric | Type | Range | Higher is Better? | Description |
|--------|------|-------|-------------------|-------------|
| Faithfulness (PRTB) | Perturbation | $[0, \infty)$ | ✓ | Prediction change from flipping important features |
| Faithfulness (SQNC) | Perturbation | $[0, \infty)$ | ✓ | Prediction change from zeroing sequences |
| Ablation (MIF) | Sequential | $[0, \infty)$ | ✓ | Avg. change removing most important features first |
| Ablation (LIF) | Sequential | $[0, \infty)$ | ✗ | Avg. change removing least important features first |
| MIF/LIF Ratio | Combined | $[0, \infty)$ | ✓ | Discrimination power of explanations |

---

## Workflow

### 1. Generate Datasets

```bash
python examples/generate_simulated_datasets.py
```

This creates:
- `examples/datasets/simulated/{dataset_name}/data.csv` - Main data file with target (N) and covariates (Z_0, Z_1, ...)
- `examples/datasets/simulated/{dataset_name}/true_importances.csv` - Ground truth feature importances

### 2. Run Window Detection

For a single dataset:
```bash
python examples/lstm_simulation.py --dataset piecewise_ar3 --n0 100 --jump 1 --num-runs 1
```

For all datasets:
```bash
bash examples/run_all_simulations.sh
```

This computes adaptive window sizes and saves to:
- `examples/results/LSTM/{dataset_name}/Jump_1_N0_100/run_0.csv` - Detection results
- `examples/results/LSTM/{dataset_name}/Jump_1_N0_100/windows.csv` - Aggregated window sizes

### 3. Run Benchmarking

Update `examples/benchmark.py` to use the new dataset format and run:

```bash
python examples/benchmark.py
```

This will compare:
- **Vanilla SHAP** - Global model with standard kernel SHAP
- **Rolling Window SHAP** - Fixed window size
- **Adaptive SHAP** - Using detected window sizes

Results are saved to:
- `examples/results/benchmark_{dataset_name}/global_shap_results.csv`
- `examples/results/benchmark_{dataset_name}/rolling_shap_results.csv`
- `examples/results/benchmark_{dataset_name}/adaptive_shap_results.csv`
- `examples/results/benchmark_{dataset_name}/benchmark_summary.csv`

## Command-Line Options

### lstm_simulation.py

```
--dataset DATASET    Dataset name (default: piecewise_ar3)
--n0 N0             Initial window size (default: 100)
--jump JUMP         Jump size for detection (default: 1)
--num-runs RUNS     Number of detection runs (default: 1)
```

### Example Usage

```bash
# Run on trend_season dataset with custom parameters
python examples/lstm_simulation.py --dataset trend_season --n0 150 --jump 2 --num-runs 3

# Run on cointegration dataset
python examples/lstm_simulation.py --dataset cointegration
```

## Dataset Format

Each dataset directory contains:

**data.csv:**
```
N,Z_0,Z_1,...
value1,cov1_1,cov1_2,...
value2,cov2_1,cov2_2,...
...
```
- `N`: Target time series (always present)
- `Z_i`: Covariates (optional, if applicable)

**true_importances.csv:**
```
true_imp_0,true_imp_1,...
imp1_0,imp1_1,...
imp2_0,imp2_1,...
...
```
- `true_imp_i`: Ground truth importance for feature i
- Features ordered as: [lag_1, lag_2, ..., lag_p, covariate_1, ..., covariate_k]

## Notes

- All datasets use sequence length 3 (AR order p=3 for lag-based features)
- Window detection uses LSTM models trained on each adaptive window
- True importances are computed using standardized absolute coefficient values
- For datasets with covariates, importances account for feature scaling
- The `piecewise_ar3` dataset is equivalent to the original `ar/3` dataset but in the new format

## Troubleshooting

**Dataset not found:**
Ensure you ran `generate_simulated_datasets.py` first.

**MPS/CUDA errors:**
The script automatically detects available devices (CPU, CUDA, MPS). If issues occur, modify the DEVICE setting in `lstm_simulation.py`.

**Memory issues:**
Reduce `t_workers` and `b_workers` in the `cd.detect()` call within `lstm_simulation.py`.

## Citation

If you use these datasets, please cite:
[Your paper citation here]