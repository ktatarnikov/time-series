from typing import Any, Callable, Dict, Sequence, Tuple  # noqa

import numpy as np
import pandas as pd


def regularization_loss(_lambda: float, matrix: np.ndarray) -> float:
    return _lambda * np.sqrt(np.sum(np.sum(matrix * matrix)))


def regularization_update(_lambda: float, matrix: np.ndarray) -> np.ndarray:
    return 2 * _lambda * matrix


class TRMF:
    def __init__(self, M: int, N: int, factors: int, lambda_f: float,
                 alpha: float):
        self.lambda_f = lambda_f
        self.alpha = alpha
        self.F = np.random.rand(M, factors)
        self.X = np.random.rand(factors, N)

    def loss(self, Y: np.ndarray) -> float:
        diff_y = Y - np.dot(self.F, self.X)
        factor_loss = np.sum(np.sum(diff_y * diff_y))
        loss_rf = 0  #regularization_loss(self.lambda_f, self.F)
        return factor_loss + loss_rf

    def update_X(self, Y: np.ndarray) -> np.ndarray:
        # TAR centric update
        diff_xt_m = 0.0
        # diff_xt_m = self.lambda_x * self.diff_xt(self.max_lag, self.X, self.W)
        # dx_reg = regularization_update(self.lambda_x, diff_xt_m)

        # factor centric update
        Y_hat = np.dot(self.F, self.X)
        diff_y = Y - Y_hat  # i * j
        dx_m = 2 * np.dot(diff_y.T, self.F).T

        dx = dx_m  # + diff_xt_m
        self.X = self.X + self.alpha * dx

    def update_F(self, Y: np.ndarray) -> np.ndarray:
        df_reg = regularization_update(self.lambda_f, self.F)

        diff = Y - np.dot(self.F, self.X)  # i * j
        df_m = 2 * np.dot(diff, self.X.T)

        df = df_m + df_reg  # full update
        self.F = self.F + self.alpha * df


class TRMF_AR:
    def __init__(self, k: int, seed: int = 42):
        self.seed = seed
        self.lambda_w = 0.05
        self.lambda_f = 0.05
        self.lambda_x = 0.05
        self.alpha = 0.01
        self.iterations = 1
        self.k = k
        self.max_lag = 10

    def train(self, frame: pd.DataFrame):
        Y = frame.values
        M = Y.shape[1]
        T = Y.shape[0]
        np.random.seed(self.seed)
        W = np.random.rand(M, self.max_lag)
        F = np.random.rand(M, k)
        X = np.random.rand(k, T)
        prev_loss = 0
        for i in range(0, self.iterations):
            loss = self.calculate_loss(Y, X, W, F)
            print(f"loss: {loss}, diff: {loss - prev_loss}")
            prev_loss = loss
            X = self.update_X(F, X, Y)
            W = self.update_W(F, X, Y, W)
            F = self.update_F(F, X, Y)

    def calculate_loss(self, Y: np.ndarray, X: np.array, W: np.array,
                       F: np.array) -> float:
        diff_y = Y - np.dot(F, X)
        factor_loss = np.sum(np.sum(np.power(diff_y, 2)))
        loss_rw = regularization_loss(self.lambda_w, W)
        loss_rf = regularization_loss(self.lambda_f, F)
        loass_tar = self.loss_tar(self.lambda_x, X, W)
        return factor_loss + loss_tar + loss_rw + loss_rf

    def loss_tar(self, l: float, X: np.ndarray, W: np.ndarray) -> float:
        loss = 0
        m = l + 1
        T = X.shape[1]
        for t in range(m, T + 1):
            xt_diff = X[:, t] - np.sum(np.multiply(W, X[:, t - 1:t - m]))
            loss += 2 * np.sum(np.power(xt_diff))
        return loss

    def regularization_loss(self, l: float, matrix: np.ndarray) -> float:
        return l * np.sqrt(np.sum(np.sum(np.power(matrix))))

    def update_X(self, F: np.ndarray, X: np.ndarray,
                 Y: np.ndarray) -> np.ndarray:
        Y_hat = np.dot(F, X)
        diff_y = Y - Y_hat  # i * j
        dx_kj = np.dot(diff.T, F).T
        diff_xt_m = self.diff_xt(self.max_lag, X, W)
        dx_m = 2 * dx_kj  # factor centric update
        diff_xt_m = 2 * self.lambda_x * diff_xt_m  # TAR centric update
        dx = dx_m + diff_xt_m1
        return X + self.alpha * dx

    def diff_xt(self, l: float, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        diff_xt_m = np.zeros(X.shape)
        m = l + 1
        T = X.shape[1]
        for t in range(m, T + 1):
            diff_xt_m[:, t] = X[:, t] - np.sum(
                np.multiply(W, X[:, t - 1:t - m]))
        return diff_xt_m

    def update_W(self, F: np.ndarray, X: np.ndarray, Y: np.ndarray,
                 W: np.ndarray) -> np.ndarray:
        diff_xt_m = self.diff_xt(self.max_lag, X, W)
        dw_m = np.zeros(W.shape)
        dw_reg = 2 * (self.lambda_w / self.lambda_x) * W
        dw = dw_m + dw_reg
        return W + self.alpha * dw

    def update_F(self, F: np.ndarray, X: np.ndarray,
                 Y: np.ndarray) -> np.ndarray:
        Y_hat = np.dot(F, X)
        diff = Y - Y_hat  # i * j
        df_m = np.dot(diff, X.T)
        df_reg = 2 * self.lambda_f * F
        df_m = 2 * df_m
        df = df_m + df_reg  # regularization update
        return F + self.alpha * df
