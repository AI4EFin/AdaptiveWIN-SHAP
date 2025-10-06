import time
import math

import numpy as np
import pandas as pd
import torch

from joblib import Parallel, delayed

from .model import AdaptiveModel


class ChangeDetector:
    def __init__(self, model: AdaptiveModel, data, weights="mammen", debug=False, force_cpu=False):
        self.model = model
        self.data = data
        self.debug = debug
        self.weights = weights
        self.previous_device = 0
        self.force_cpu = force_cpu

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

    # A small factory so each worker uses a fresh model instance.
    # If your model can be reconstructed via something like `type(self.model)(**self.model.init_kwargs)`,
    # encode that here. Otherwise, if self.model is stateless & picklable, you can pass a lambda returning self.model.
    def construct_new_model(self):
        # Example: clone by re-calling the class with same config
        # Adjust to your actual model construction needs.
        # CUDA first (multi-GPU)
        kwargs = self.model.init_kwargs
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            kwargs["device"] = torch.device(f"cuda:{self.previous_device}")
            self.previous_device = (self.previous_device + 1) if self.previous_device < torch.cuda.device_count() - 1 else 0
        if self.force_cpu == True:
            kwargs["device"] = torch.device("cpu")
        return type(self.model)(**kwargs)

    # ----------------------------
    # Detection run (with weighted bootstrap retraining)
    # ----------------------------
    def detect(self, min_window=3, n_0=200, jump=10, search_step=5, alpha=0.95, num_bootstrap=50, t_workers=-1, b_workers=-1, one_b_threads=-1):
        """
        t_workers: int, default -1 (use all available). Workers used to parallelize the test statistic.
        b_workers: int, default -1 (use all available). Workers used to parallelize the bootstrap.
        one_b_threads: int, default -1 (use all available). Workers used to parallelize the inner loop of a bootstrap.
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
                start_k_time = time.time()
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
                likelihood_i, yhat_i, resid_i, _, _, _ = self.construct_new_model().fit(X_all, y_all).diagnostics(X_all, y_all)

                # Candidate split range
                J_start = max(min_window, io - n_k)  # have enough left history
                J_end = io - n_k_minus1  # ensure right side at least n_0
                if J_end <= J_start:
                    print(f"J_end ({J_end}) < J_start ({J_start})")
                    continue

                # Candidate split range in ABSOLUTE target indices
                J_abs = np.arange(J_start, J_end, search_step, dtype=np.int64)

                # --- Observed T(i) across splits ---
                T_vals = self.compute_T_vals(X_all, y_all, likelihood_i, J_abs, t_abs, t_workers)

                # --- Bootstrap critical values via wild residual bootstrap under the null ---
                if num_bootstrap <= 0:
                    raise ValueError(f"Num bootstrap must be at least 1. {num_bootstrap} provided")

                Sup_boot = self.calculate_t_bootstrap(X_all, yhat_i, resid_i, J_abs, t_abs, num_bootstrap, min_seg=min_window, n_jobs=b_workers, n_inner_threads=one_b_threads, batch_size=512)

                end_k_time = time.time()
                # --- Decision for the current window ---
                if len(T_vals) > 0:
                    test_value = float(np.max(T_vals))
                    critical_value = float(np.quantile(Sup_boot, alpha))
                    print(
                        f"[QLR] step={l} |I_k+1=[{max(0, io - n_k_plus1)}, {io}] | I_k=[{max(0, io - n_k)}, {io}]  |"
                        f"I_k-1=[{max(0, io - n_k_minus1)}, {io}] | J_k=[{J_start}, {J_end}] | k={k} | "
                        f"SupLR={test_value:.3f} | crit({alpha:.2f})={critical_value:.3f} | #splits={len(T_vals)} | B={num_bootstrap} | time/k={end_k_time - start_k_time:.2f}s")
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

    def compute_T_vals(self, X_all, y_all, likelihood_i, J_abs, t_abs, max_processes):
        def safe_calc(i_abs):
            try:
                Ti = self.calculate_t(X_all, y_all, likelihood_i, i_abs, t_abs)
                return max(0.0, Ti)
            except ValueError:
                return 0.0

        # prefer="processes" → multi-process;
        T_vals = Parallel(n_jobs=max_processes, prefer="processes", batch_size='auto')(delayed(safe_calc)(i) for i in J_abs)
        return T_vals

    def calculate_t(self, X_all, y_all, likelihood_i, i_abs, t_abs):
        if self.debug == True:
            print(f"tau={i_abs}")
        # Strict no-leak masks by target index
        Lmask = t_abs <= i_abs
        Rmask = t_abs > i_abs
        mA = int(np.sum(Lmask))
        mB = int(np.sum(Rmask))
        if mA < 20 or mB < 20:  # min targets per side; tune as needed
            raise ValueError(f"Too few targets for split {i_abs}")
        likelihood_a, _, _, _, _, _ = self.construct_new_model().fit(X_all[Lmask], y_all[Lmask]).diagnostics(X_all[Lmask],
                                                                                             y_all[Lmask])
        likelihood_b, _, _, _, _, _ = self.construct_new_model().fit(X_all[Rmask], y_all[Rmask]).diagnostics(X_all[Rmask],
                                                                                             y_all[Rmask])

        Ti = likelihood_a + likelihood_b - likelihood_i
        return Ti

    def _Ti_for_mask(self, X_all, y_star_t, Lmask, Rmask, min_seg, likelihood_i_b):
        # returns 0.0 if segment too small
        mA = int(Lmask.sum())
        mB = int(Rmask.sum())
        if mA < min_seg or mB < min_seg:
            return 0.0
        likelihood_a_b, *_ = self.construct_new_model().fit(X_all[Lmask], y_star_t[Lmask]).diagnostics(
            X_all[Lmask], y_star_t[Lmask]
        )
        likelihood_b_b, *_ = self.construct_new_model().fit(X_all[Rmask], y_star_t[Rmask]).diagnostics(
            X_all[Rmask], y_star_t[Rmask]
        )
        return likelihood_a_b + likelihood_b_b - likelihood_i_b

    def _draw_weights(self, kind: str, n: int, rng: np.random.Generator):
        if kind == "mammen":
            # assuming these static methods exist on ChangeDetector
            return ChangeDetector.draw_mammen(n, rng)
        elif kind == "rademacher":
            return ChangeDetector.draw_rademacher(n, rng)
        else:
            raise ValueError(f"Weights {kind} not supported. Use 'mammen' or 'rademacher'.")

    def _one_bootstrap(
            self,
            X_all,
            yhat_i,
            resid_i,
            masks,  # list[(Lmask, Rmask)]
            weights_kind: str,
            min_seg: int = 20,
            inner_jobs: int = 1,  # NEW: threads per worker for masks
    ):
        rng = np.random.default_rng()

        # 1) multipliers & pseudo response
        w = self._draw_weights(weights_kind, len(resid_i), rng)
        y_star = (yhat_i + w * resid_i)
        y_star_t = torch.from_numpy(y_star)

        # 2) global null likelihood (shared within this bootstrap)
        model = self.construct_new_model()
        likelihood_i_b, *_ = model.fit(X_all, y_star_t).diagnostics(X_all, y_star_t)

        # 3) parallel sweep across masks using threads
        if inner_jobs == 1:
            # fast path: no threading, just a loop
            sup_b = 0.0
            for Lmask, Rmask in masks:
                Ti_b = self._Ti_for_mask(X_all, y_star_t, Lmask, Rmask, min_seg, likelihood_i_b)
                if Ti_b > sup_b:
                    sup_b = Ti_b
        else:
            # threaded inner parallelism (avoid processes-inside-processes)
            Ti_list = Parallel(n_jobs=inner_jobs, backend="threading", batch_size="auto")(
                delayed(self._Ti_for_mask)(
                    X_all, y_star_t, Lmask, Rmask, min_seg, likelihood_i_b
                )
                for (Lmask, Rmask) in masks
            )
            sup_b = max(Ti_list) if Ti_list else 0.0

        return max(0.0, sup_b)

    def calculate_t_bootstrap(self, X_all, yhat_i, resid_i, J_abs, t_abs, num_bootstrap, min_seg=20, n_jobs=-1, n_inner_threads=-1, batch_size=512):
        """
                Parallel bootstrap over `b` replicates using joblib.
                Returns: array shape (num_bootstrap,)
                """

        # Precompute masks for every split once (depends only on t_abs & J_abs)
        masks = []
        for i_abs in J_abs:
            Lmask = (t_abs <= i_abs)
            Rmask = ~Lmask  # faster than recomputing
            masks.append((Lmask, Rmask))

        # Parallel loop over bootstrap draws
        Sup_boot_list = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=batch_size)(
            delayed(self._one_bootstrap)(
                X_all=X_all,
                yhat_i=yhat_i,
                resid_i=resid_i,
                masks=masks,
                weights_kind=self.weights,
                min_seg=min_seg,
                inner_jobs=n_inner_threads,
            )
            for _ in range(num_bootstrap)
        )
        return np.asarray(Sup_boot_list, dtype=np.float64)
