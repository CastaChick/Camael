import numpy as np


def cross_entropy(y_label, y_pred):
    """
    クロスエントロピー誤差を求める

    Parameters
    ----------
    y_label: array
        正解ラベル

    y_pred: array
        予測ラベル
    """
    assert y_label.shape == y_pred.shape, "引数の形が不正です"
    loss = -np.sum(y_label * np.log(y_pred)) / y_label.shape[0]
    df = y_label
    return loss, df


def MSE(y, y_pred):
    """
    平均二乗誤差を求める

    Parameters
    ----------
    y: array
        正解の値

    y_pred: array
        予測された値
    """
    assert y_label.shape == y_pred.shape, "引数の形が不正です"
    loss = np.sum((y - y_pred)**2) / y.shape[0]
    df = -2*(y - y_pred)
    return loss, df
