import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import store_init_kwargs


class AdaptiveModel(nn.Module):
    def __init__(self, device, batch_size=512, lr=1e-12, epochs=50, type_precision=np.float32):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.type_precision = type_precision

    def fit(self, X, y):
        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        print(self.device)
        model = self.to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        for _ in range(self.epochs):
            model.train()
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        return self

    @staticmethod
    def residuals(y, yhat):
        return y - yhat

    @staticmethod
    def sse(residuals):
        return np.sum(residuals**2)

    @staticmethod
    def mse(sse, n):
        return sse / max(n, 1)

    @staticmethod
    def likelihood(sse, n):
        return -(n / 2) * np.log(sse)

    def prepare_data(self, window, start_abs_idx):
        """
        :param window: The input data.
        :param start_abs_idx: The id from which the target data starts in the original time series. This function takes the window data.
        :return: X_tensor, y_tensor, t_abs
        """
        X, y = window[:, :-1], window[:, -1]
        X_tensor = X if isinstance(X, torch.Tensor) else torch.from_numpy(X)
        y_tensor = y if isinstance(y, torch.Tensor) else torch.from_numpy(y)
        t_abs = np.arange(start_abs_idx, start_abs_idx + len(y), dtype=np.int64)
        return X_tensor, y_tensor, t_abs

    def diagnostics(self, X, y, batch_size=512):
        ds = TensorDataset(X, y)

        self.eval()
        with torch.no_grad():
            all_pred, all_y = [], []
            evl = DataLoader(ds, batch_size=batch_size, shuffle=False)
            for xb, yb in evl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self(xb)
                all_pred.append(pred.cpu().numpy())
                all_y.append(yb.cpu().numpy())
            yhat = np.concatenate(all_pred)
            y_true = np.concatenate(all_y)
        resid = AdaptiveModel.residuals(y_true, yhat)
        sse = AdaptiveModel.sse(resid)
        m = int(y_true.shape[0])
        mse = AdaptiveModel.mse(sse, m)
        likelihood = AdaptiveModel.likelihood(sse, m)
        return likelihood, yhat, resid, sse, mse, m
