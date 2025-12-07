import numpy as np
import scipy as sp
import pandas as pd
from descents import BaseDescent
from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, Type, Optional


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()

class LinearRegression:
    def __init__(
        self,
        optimizer: Optional[BaseDescent | str],
        l2_coef: float = 0.0,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
        loss_function: LossFunction = LossFunction.MSE
    ):
        self.optimizer = optimizer
        #if isinstance(optimizer, BaseDescent):
        if hasattr(self.optimizer, "step"):
            self.optimizer.set_model(self)
        self.l2_coef = l2_coef
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.loss_function = loss_function
        self.w = None
        self.X_train = None
        self.y_train = None
        self.loss_history = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: реализовать функцию предсказания в линейной регрессии
        return X @ self.w

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.loss_function is LossFunction.MSE:
            n = X.shape[0]
            residual = X @ self.w - y
            return (2.0 / n) * (X.T @ residual)
        # elif self.loss_function is ...
        return None

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.loss_function is LossFunction.MSE:
            n = X.shape[0]
            residual = X @ self.w - y
            return float((residual @ residual) / n)
        # elif self.loss_function is ...
        return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        # TODO: реализовать обучение модели
        self.X_train, self.y_train = X, y

        if isinstance(self.X_train, (pd.DataFrame, pd.Series)):
            self.X_train = self.X_train.to_numpy()
        if isinstance(self.y_train, (pd.Series, pd.DataFrame)):
            self.y_train = np.asarray(self.y_train).ravel()

        #if isinstance(self.optimizer, BaseDescent):
        if hasattr(self.optimizer, "step"):
            n, d = self.X_train.shape
            if self.w is None:
                self.w = np.zeros(d, dtype=float)

            self.loss_history.append(self.compute_loss(self.X_train, self.y_train))

            for _ in range(self.max_iter):
                # 1 шаг градиентного спуска
                delta_w = self.optimizer.step()

                if np.sum(delta_w ** 2) < self.tolerance:
                    break

                if not np.all(np.isfinite(delta_w)):
                    break
                
                self.loss_history.append(self.compute_loss(self.X_train, self.y_train))

            return self
        elif self.optimizer is None: # аналит ческое решенеи
            XtX = X.T @ X
            Xty = X.T @ y
            
            self.w = np.linalg.solve(XtX, Xty)

            self.loss_history.append(self.compute_loss(X, y))
            return self
        elif self.optimizer == "SVD":
            n, d = X.shape
            k = min(4, n, d)
            U, s, Vt = sp.sparse.linalg.svds(X, k=k, which="LM")
            idx = np.argsort(s)[::-1]
            U, s, Vt = U[:, idx], s[idx], Vt[idx, :]
            self.w = Vt.T @ ((U.T @ y) / s)

            self.loss_history.append(self.compute_loss(X, y))
            return self
        raise NotImplementedError("Linear Regression training is not implemented")
