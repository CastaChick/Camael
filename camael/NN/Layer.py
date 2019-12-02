import numpy as np
from copy import deepcopy

class FC:
    """
    ニューラルネットワークの全結合層

    Parameter
    ---------
    input_shape: int (default=None)
        入力サイズ
        入力層の時に指定

    output_shape: int
        出力サイズ
    """
    def __init__(self, output_shape, input_shape=None):
        self._sp = "FC"
        self.input_shape = input_shape
        self.output_shape = output_shape

    def _initiarize(self, _input_shape, _optimizer):
        if not self.input_shape:
            self.input_shape = _input_shape

        self.A = \
            np.random.normal(0, 0.1, (self.input_shape, self.output_shape))
        self.b = np.random.normal(0, 0.1, self.output_shape)
        self.optimizerA = deepcopy(_optimizer)
        self.optimizerb = deepcopy(_optimizer)

    def _forward(self, X):
        self.X = X
        return self.X.dot(self.A) + self.b

    def _backward(self, df):
        dx = df.dot(self.A.T)
        dA = df.T.dot(self.X).T / self.X.shape[1]
        self.A = self.optimizerA._update(self.A, dA)
        db = np.average(df, axis=0)
        self.b = self.optimizerb._update(self.b, db)
        return dx


class ReLU:
    """
    ニューラルネットワークのReLU層
    """
    def __init__(self):
        self._sp = "activate"

    def _forward(self, X):
        self.X = X
        return np.where(self.X > 0, self.X, 0)

    def _backward(self, df):
        return np.where(self.X > 0, df, 0)


class Softmax:
    """
    ニューラルネットワークのソフトマックス層
    """
    def __init__(self):
        self._sp = "activate"

    def _forward(self, X):
        X -= np.max(X)
        X = np.exp(X)
        self.y = X / np.sum(X, axis=1).reshape((-1, 1))
        return self.y

    def _backward(self, t):
        """
        # 実装の単純化のための仮定
        * 損失関数としてクロスエントロピーを用いること
        * 損失関数とソフトマックス層を合わせた微分係数を計算する
        つまりtは正解ラベル
        """
        return self.y - t


class Tanh:
    def __init__(self):
        self._sp = "activate"

    def _forward(self, X):
        self.X = X
        return np.tanh(self.X)

    def _backward(self, df):
        dx = 1 / np.cosh(self.X)**2
        return df*dx
