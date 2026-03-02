# DGP Parameter Robustness: Centroid Interpolation Methodology

## 1. Baseline Piecewise AR(3)

The baseline data generating process is a piecewise-stationary AR(3) with three regimes of equal length ($T_k = 500$, total $T = 1500$). Each regime has a single dominant lag:

$$
X_t = \phi_1^{(k)} X_{t-1} + \phi_2^{(k)} X_{t-2} + \phi_3^{(k)} X_{t-3} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, 1)
$$

where $k \in \{0, 1, 2\}$ denotes the active regime and the coefficient vectors are:

$$
\phi^{(0)} = (0.90,\; 0.01,\; 0.01), \quad
\phi^{(1)} = (0.01,\; 0.90,\; 0.01), \quad
\phi^{(2)} = (0.01,\; 0.01,\; 0.90)
$$

Breakpoints occur at $t = 500$ and $t = 1000$.

The maximum pairwise $\ell_2$ distance between any two regimes is:

$$
d_{\max} = \max_{i \neq j} \|\phi^{(i)} -\phi^{(j)}\|_2 = \|(0.89, -0.89, 0)\|_2 = 0.89\sqrt{2} \approx 1.259
$$


## 2. Centroid Interpolation

To systematically reduce regime contrast, we interpolate each regime's coefficient vector toward the centroid of all three regimes.

### 2.1 Centroid

The centroid $\bar{\phi}$ is the elementwise mean of the three baseline regimes:

$$
\bar{\phi} = \frac{1}{3}\sum_{k=0}^{2} \phi^{(k)} = \left(\frac{0.92}{3},\; \frac{0.92}{3},\; \frac{0.92}{3}\right) \approx (0.3067,\; 0.3067,\; 0.3067)
$$

Note that $\bar{\phi}$ lies at equal distance from all three vertices, so interpolating toward it shrinks the regime differences uniformly.

### 2.2 Interpolation Formula

For a reduction factor $\tau \in [0, 1]$, the modified regime coefficients are:

$$
\phi^{(k)}_\tau = (1 - \tau)\,\phi^{(k)} + \tau\,\bar{\phi}, \quad k = 0, 1, 2
$$

- $\tau = 0$: baseline (no change)
- $\tau = 1$: all three regimes collapse to the centroid (no regime differences)

### 2.3 $\ell_2$ Distance Reduction

The pairwise distance between any two modified regimes scales linearly:

$$
\|\boldsymbol{\phi}^{(i)}_\tau - \boldsymbol{\phi}^{(j)}_\tau\|_2
= \|(1 - \tau)(\boldsymbol{\phi}^{(i)} - \boldsymbol{\phi}^{(j)})\|_2
= (1 - \tau)\,\|\boldsymbol{\phi}^{(i)} - \boldsymbol{\phi}^{(j)}\|_2
$$

Therefore the maximum pairwise distance reduces by exactly the factor $(1 - \tau)$:

$$
d_{\max}(\tau) = (1 - \tau)\,d_{\max}(0)
$$

A "$\tau$% reduction" means the $\ell_2$ distance is reduced by $\tau$ relative to baseline. For example, $\tau = 0.50$ halves the maximum pairwise distance.


## 3. Tested Scenarios

| Scenario | $\tau$ | $d_{\max}$ | Regime 0 | Regime 1 | Regime 2 |
|----------|--------|------------|----------|----------|----------|
| Baseline | 0.00 | 1.259 | (0.900, 0.010, 0.010) | (0.010, 0.900, 0.010) | (0.010, 0.010, 0.900) |
| l2_10 | 0.10 | 1.133 | (0.841, 0.040, 0.040) | (0.040, 0.841, 0.040) | (0.040, 0.040, 0.841) |
| l2_20 | 0.20 | 1.007 | (0.781, 0.069, 0.069) | (0.069, 0.781, 0.069) | (0.069, 0.069, 0.781) |
| l2_30 | 0.30 | 0.881 | (0.722, 0.099, 0.099) | (0.099, 0.722, 0.099) | (0.099, 0.099, 0.722) |
| l2_40 | 0.40 | 0.755 | (0.663, 0.129, 0.129) | (0.129, 0.663, 0.129) | (0.129, 0.129, 0.663) |
| l2_50 | 0.50 | 0.629 | (0.603, 0.158, 0.158) | (0.158, 0.603, 0.158) | (0.158, 0.158, 0.603) |
| l2_75 | 0.75 | 0.315 | (0.455, 0.232, 0.232) | (0.232, 0.455, 0.232) | (0.232, 0.232, 0.455) |
| l2_90 | 0.90 | 0.126 | (0.366, 0.277, 0.277) | (0.277, 0.366, 0.277) | (0.277, 0.277, 0.366) |


## 4. True Feature Importances

At each time step $t$ in regime $k$, the true feature importance for lag $j$ is defined as the normalised absolute coefficient:

$$
I_j(t) = \frac{|\phi_j^{(k)}_\tau|}{\sum_{m=1}^{3}|\phi_m^{(k)}_\tau|}
$$

As $\tau$ increases, the importances converge toward $(1/3, 1/3, 1/3)$, making it progressively harder to distinguish which lag drives the process.


## 5. Geometric Interpretation

The three baseline regimes lie at the vertices of an equilateral triangle in the coefficient simplex (since $\phi_1 + \phi_2 + \phi_3 = 0.92$ is constant). Increasing $\tau$ moves all three vertices uniformly toward the centroid:

```
        phi^(2) = (0.01, 0.01, 0.90)
            /\
           /  \
          / tau \         <- interpolation shrinks the triangle
         /  .C   \            toward centroid C
        /________\
  phi^(0)      phi^(1)
```

At $\tau = 1$, the triangle collapses to a single point (the centroid) and all regimes become indistinguishable.


## 6. Stationarity Guarantee

An AR(3) process is stationary if and only if all roots of the characteristic polynomial (in lag operator form)

$$
1 - \phi_1 z - \phi_2 z^2 - \phi_3 z^3 = 0
$$

lie strictly outside the unit circle ($|z| > 1$).

**Claim.** All interpolated regimes remain stationary for every $\tau \in [0, 1]$.

**Proof.** Since all coefficients $\phi_j^{(k)}_\tau > 0$ for every scenario (the minimum coefficient value is $0.01$ at baseline, increasing with $\tau$), we have $|\phi_j| = \phi_j$ and the sum is:

$$
\sum_{j=1}^{3} \phi_j^{(k)}_\tau = (1-\tau)\sum_j \phi_j^{(k)} + \tau\sum_j \bar{\phi}_j = (1-\tau)(0.92) + \tau(0.92) = 0.92 \quad \forall\, \tau, k
$$

For any $z$ with $|z| \leq 1$, by the triangle inequality:

$$
|1 - \phi_1 z - \phi_2 z^2 - \phi_3 z^3| \geq 1 - |\phi_1||z| - |\phi_2||z|^2 - |\phi_3||z|^3 \geq 1 - \sum_{j=1}^3 |\phi_j| = 0.08 > 0
$$

Therefore no root can lie on or inside the unit circle, and every interpolated regime is stationary. $\square$

This is verified numerically for all scenarios; the smallest root modulus across all regimes and all $\tau$ values is $\min |z| \approx 1.029$ (Regime 2 at baseline).