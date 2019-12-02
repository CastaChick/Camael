import numpy as np


def accuracy(y, y_pred):
    """
    分類問題で正解率を算出する

    Parameters
    ----------
    y: array
        正解ラベル

    y_pred: array
        予測値
    """
    y = np.argmax(y, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    return np.sum(y_pred==y) / y_pred.shape[0] * 100
