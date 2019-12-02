import numpy as np


def accuracy(y, y_pred):
    """
    分類問題で正解率を算出する

    Parameters
    ----------
    y: array
        正解ラベル

    y_pred: array
        予測ラベル

    Returns
    -------
    acc: float
        正解率
    """
    y = np.argmax(y, axis=1)

    return np.sum(y_pred == y) / y_pred.shape[0] * 100


def mean_squre_error(y, y_pred):
    """
    回帰問題で平均二乗誤差を算出する

    Parameters
    ----------
    y: array
        正解値

    y_pred: array
        予測値

    Returns
    -------
    loss: float
        平均二乗誤差
    """
    return np.sum((y - y_pred)**2) / y.shape[0]