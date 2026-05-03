from __future__ import annotations

import numpy as np

from config import GPConfig


class GaussianProcessRegressor2DInput:
    def __init__(self, config: GPConfig):
        self.config = config
        self.x_train: np.ndarray | None = None
        self.alpha: np.ndarray | None = None

    def kernel(self, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        xa = np.asarray(xa, dtype=float)
        xb = np.asarray(xb, dtype=float)
        deltas = (xa[:, None, :] - xb[None, :, :]) / self.config.length_scale_xy
        sqdist = np.sum(deltas**2, axis=2)
        return self.config.signal_variance * np.exp(-0.5 * sqdist)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        x_train = np.asarray(x_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float).reshape(-1, 1)
        k_xx = self.kernel(x_train, x_train)
        jitter = self.config.noise_variance * np.eye(x_train.shape[0])
        cho = np.linalg.cholesky(k_xx + jitter)
        alpha = np.linalg.solve(cho.T, np.linalg.solve(cho, y_train))

        self.x_train = x_train
        self.alpha = alpha

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        if self.x_train is None or self.alpha is None:
            raise RuntimeError("GP model must be fitted before prediction.")

        k_tx = self.kernel(np.asarray(x_test, dtype=float), self.x_train)
        return k_tx @ self.alpha


class VectorFieldGP2D:
    def __init__(self, config: GPConfig):
        self.model_x = GaussianProcessRegressor2DInput(config)
        self.model_y = GaussianProcessRegressor2DInput(config)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model_x.fit(x_train, y_train[:, 0])
        self.model_y.fit(x_train, y_train[:, 1])

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        pred_x = self.model_x.predict(x_test)
        pred_y = self.model_y.predict(x_test)
        return np.column_stack([pred_x.reshape(-1), pred_y.reshape(-1)])


class TimeSliceGPCollection:
    def __init__(self, config: GPConfig):
        self.config = config
        self.slice_times: np.ndarray | None = None
        self.models: list[VectorFieldGP2D] = []

    def fit(
        self,
        slice_times: np.ndarray,
        x_slices: np.ndarray,
        y_slices: np.ndarray,
    ) -> None:
        self.slice_times = np.asarray(slice_times, dtype=float)
        self.models = []
        for x_train, y_train in zip(x_slices, y_slices):
            model = VectorFieldGP2D(self.config)
            model.fit(x_train, y_train)
            self.models.append(model)

    def _slice_index(self, t: float) -> int:
        if self.slice_times is None or not self.models:
            raise RuntimeError("Time-slice GP collection must be fitted before prediction.")
        return int(np.argmin(np.abs(self.slice_times - t)))

    def predict(self, t: float, x: np.ndarray) -> np.ndarray:
        model = self.models[self._slice_index(float(t))]
        return model.predict(np.asarray(x, dtype=float).reshape(-1, 2))
