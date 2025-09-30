import time
import math

import numpy as np
import pandas as pd
import torch

from .model import AdaptiveModel


class ChangeDetector:
    def __init__(self, model: AdaptiveModel, data, weights="mammen", debug=False):
        self.model = model
        self.data = data
        self.debug = debug
        self.weights = weights

    @staticmethod
    def draw_rademacher(m, rng):
        return (rng.integers(0, 2, size=m) * 2 - 1).astype(np.float32)

    @staticmethod
    def draw_mammen(m, rng):
        p = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
        a = (1 - np.sqrt(5)) / 2  # ≈ -0.618
        b = (1 + np.sqrt(5)) / 2  # ≈  1.618
        u = rng.random(m)
        w = np.where(u < p, a, b).astype(np.float32)
        return w

    # ----------------------------
    # Detection run (with weighted bootstrap retraining)
    # ----------------------------
    def detect(self, min_window=3, n_0=200, jump=10, search_step=5, alpha=0.95, num_bootstrap=50):
        """
        data: 1D numpy array (time series)
        Returns (DataFrame, tests) with diagnostics per step.
        Uses OOS MSE and **per-batch weighted retraining** in bootstrap if enabled.
        """
        DT_N = pd.DataFrame({"Date": np.arange(len(self.data)), "N": self.data})
        windows, mse_vals, rmse_vals, likelihoods, scaled_windows = [], [], [], [], []
        tests = 0

        io = self.data.shape[0]  # starting from final time
        I_0 = list(self.data[max(0, io - n_0):io])

        n_k_minus1 = n_0
        I_k_minus1 = I_0

        for l in range(0, self.data.shape[0], jump):
            t0 = time.time()
            io = self.data.shape[0] - l

            # arithmetic schedule
            K = int(io / n_0)

            I_0 = list(self.data[max(0, io - n_0):io])
            I_k_minus1 = I_0
            n_k_minus1 = n_0

            for k in range(1, K + 1):
                # arithmetic
                n_k = (k + 1) * n_0
                n_k_plus1 = (k + 2) * n_0

                I_k = list(self.data[max(0, io - n_k):io])
                I_k_plus1 = list(self.data[max(0, io - n_k_plus1):io])

                # Pooled (observed) on I_k_plus1, out-of-sample
                # === Precompute all sequences for the CURRENT window I_k_plus1 once ===
                y_win = np.asarray(I_k_plus1, dtype=np.float32)
                start_abs = max(0, io - n_k_plus1)
                X_all, y_all, t_abs = self.model.prepare_data(y_win, start_abs)
                if X_all is None:
                    print("Window too small for sequences")
                    continue

                # --- Global null fit on I_{k+1} (observed) ---
                likelihood_i, yhat_i, resid_i, _, _, _ = self.model.fit(X_all, y_all).diagnostics(X_all, y_all)

                # Candidate split range
                J_start = max(min_window, io - n_k)  # have enough left history
                J_end = io - n_k_minus1  # ensure right side at least n_0
                if J_end <= J_start:
                    print(f"J_end ({J_end}) < J_start ({J_start})")
                    continue

                # Candidate split range in ABSOLUTE target indices
                J_abs = np.arange(J_start, J_end, search_step, dtype=np.int64)

                T_vals = []

                # --- Observed T(i) across splits ---
                for i_abs in J_abs:
                    if self.debug == True:
                        print(f"tau={i_abs}")
                    # Strict no-leak masks by target index
                    Lmask = t_abs <= i_abs
                    Rmask = t_abs > i_abs
                    mA = int(np.sum(Lmask))
                    mB = int(np.sum(Rmask))
                    if mA < 20 or mB < 20:  # min targets per side; tune as needed
                        T_vals.append(0.0)
                        continue
                    likelihood_a, _, _, _, _, _ = self.model.fit(X_all[Lmask], y_all[Lmask]).diagnostics(X_all[Lmask], y_all[Lmask])
                    likelihood_b, _, _, _, _, _ = self.model.fit(X_all[Rmask], y_all[Rmask]).diagnostics(X_all[Rmask], y_all[Rmask])

                    Ti = likelihood_a + likelihood_b - likelihood_i
                    T_vals.append(max(0.0, Ti))

                # --- Bootstrap critical values via wild residual bootstrap under the null ---
                if num_bootstrap <= 0:
                    raise ValueError(f"Num bootstrap must be at least 1. {num_bootstrap} provided")

                rng = np.random.default_rng()  # SEED
                Sup_boot = np.empty(num_bootstrap, dtype=np.float64)  # store sup over splits for each b

                for b in range(num_bootstrap):
                    # multipliers (choose Mammen or Rademacher)
                    w = None
                    if self.weights == "mammen":
                        w = ChangeDetector.draw_mammen(len(resid_i), rng)
                    elif self.weights == "rademacher":
                        w = ChangeDetector.draw_rademacher(len(resid_i), rng)
                    else:
                        raise ValueError(f"Weights {self.weights} not supported. Use 'mammen' or 'rademacher'")

                    y_star = (yhat_i + w * resid_i)
                    y_star_tensor = torch.from_numpy(y_star)

                    # Refit global null on y* to get SSE_I*
                    likelihood_i_b, _, _, _, _, _ = self.model.fit(X_all, y_star_tensor).diagnostics(X_all, y_star_tensor)

                    # Sweep splits and take sup
                    sup_b = 0.0
                    for i_abs in J_abs:
                        Lmask = t_abs <= i_abs
                        Rmask = t_abs > i_abs
                        mA = int(np.sum(Lmask))
                        mB = int(np.sum(Rmask))
                        if mA < 20 or mB < 20:  # same min-seg rule
                            continue
                        likelihood_a_b, _, _, _, _, _ = self.model.fit(X_all[Lmask], y_star_tensor[Lmask]).diagnostics(X_all[Lmask], y_star_tensor[Lmask])
                        likelihood_b_b, _, _, _, _, _ = self.model.fit(X_all[Rmask], y_star_tensor[Rmask]).diagnostics(X_all[Rmask], y_star_tensor[Rmask])

                        Ti_b = likelihood_a_b + likelihood_b_b - likelihood_i_b

                        if Ti_b > sup_b: sup_b = Ti_b
                    Sup_boot[b] = max(0.0, sup_b)

                # --- Decision for the current window ---
                if len(T_vals) > 0:
                    test_value = float(np.max(T_vals))
                    critical_value = float(np.quantile(Sup_boot, alpha))
                    print(
                        f"[QLR] step={l} |I_k+1=[{max(0, io - n_k_plus1)}, {io}] | I_k=[{max(0, io - n_k)}, {io}]  |"
                        f"I_k-1=[{max(0, io - n_k_minus1)}, {io}] | J_k=[{J_start}, {J_end}] | k={k} | "
                        f"SupLR={test_value:.3f} | crit({alpha:.2f})={critical_value:.3f} | #splits={len(T_vals)} | B={num_bootstrap}")
                else:
                    test_value, critical_value = 0.0, math.inf

                if test_value > critical_value:
                    print(f"Found break at step {l} (window size {len(I_k)}).")
                    break
                else:
                    I_k_minus1 = I_k
                    n_k_minus1 = n_k
                    continue

            # Record diagnostics
            if K == 0:
                I_k = I_0

            # For the per-step diagnostic, compute OOS MSE on I_k
            MSE_I_k = 0  # segment_oos_mse(I_k, seq_len=seq_len, epochs=epochs, error_type=error_type)
            RMSE_I_k = 0  # math.sqrt(max(MSE_I_k, 0.0))
            Likelihood_I_k = 0

            windows.append(len(I_k))
            mse_vals.append(MSE_I_k)
            rmse_vals.append(RMSE_I_k)
            likelihoods.append(Likelihood_I_k)

            scaled_windows.append(len(I_k) / io)

            t1 = time.time()
            print(f"Step {l:4d} | time/step={t1 - t0:.2f}s | window={len(I_k)} | RMSE={RMSE_I_k:.4f}")

        # Reverse to align like your original flow
        windows.reverse()
        mse_vals.reverse()
        rmse_vals.reverse()
        scaled_windows.reverse()
        DT_N["windows"] = pd.Series(windows)
        DT_N["scaled_windows"] = pd.Series(scaled_windows)
        DT_N["MSE"] = pd.Series(mse_vals)
        DT_N["RMSE"] = pd.Series(rmse_vals)

        return DT_N, tests
