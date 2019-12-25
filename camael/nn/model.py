import numpy as np
from copy import deepcopy


class Model:
    """
    ニューラルネットワーク全体のモデル
    """
    def __init__(self):
        self.layers = []
        self._input_shape = None

    def add(self, layer):
        self.layers.append(layer)
        if not self._input_shape:
            assert layer.input_shape, "入力のサイズを指定して下さい"
            self._input_shape = layer.input_shape

    def compile(self, optimizer, loss, metrix=None):
        """
        モデルのコンパイル、初期化を行う

        Parameters
        ----------
        optimizer: class
            最適化手法

        loss: func
            損失関数

        metrix: func (default=None)
            モデルの評価
        """
        for layer in self.layers:
            if layer._sp == "FC":
                layer._initiarize(self._input_shape, deepcopy(optimizer))
                self._input_shape = layer.output_shape

        self.loss = loss
        self.metrix = metrix

    def fit(self,
            X, y,
            max_iter=10,
            batch_size=100,
            shuffle=True,
            log=True):
        """
        モデルのトレーニングを行う

        Parameters
        ----------
        X: array, shape=(samples, columns)
            学習データの特徴量

        y: vector, len=(samples)
            学習データの正解ラベル

        max_iter: int (default=10)
            最大エポック数

        batch_size: int (default=100)
            バッチサイズ

        log: boolean (default=True)
            ログを出力するかどうか
        """
        self.X = X
        self.y = y
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.log = log
        mode = "fit"
        for epoch in range(max_iter):
            if self.log:
                print("epoch: {}".format(epoch+1))
            batch = 1
            for X_batch, y_batch in self._get_minibatch():
                out = X_batch
                for layer in self.layers:
                    out = layer._forward(out, mode=mode)
                loss, df = self.loss(y_batch, out)
                for layer in self.layers[::-1]:
                    df = layer._backward(df, mode=mode)

                if self.log:
                    print("batch: {} loss: {}".format(batch, loss))
                batch += 1
            if self.log and self.metrix:
                print("acc: {:.2f}%".format(self.score(self.X, self.y)))

    def predict(self, X):
        """
        予測を行う

        Parameters
        ----------
        X: array, shape=(samples, columns)
            テストデータ
        """
        if self._input_shape == 1:
            return self._predict_reg(X)
        else:
            return self._predict_clf(X)

    def _predict_clf(self, X):
        out = X
        mode = "predict"
        for layer in self.layers:
            out = layer._forward(out, mode=mode)
        pred_label = np.argmax(out, axis=1)
        return pred_label

    def _predict_reg(self, X):
        out = X
        mode = "predict"
        for layer in self.layers:
            out = layer._forward(out, mode=mode)
        return out

    def score(self, X, y):
        """
        モデルの正解率を求める

        Parameters
        ----------
        X: array, shape=(samples, coumns)
            説明変数の行列

        y: array
            教師データの行列
        Returns
        -------
        score: float
            モデルのスコア
        """
        assert X.shape[0] == y.shape[0], "サンプル数を合わせてください"
        return self.metrix(y, self.predict(X))

    def _get_minibatch(self):
        index = np.arange(self.X.shape[0])
        np.random.shuffle(index)
        if self.X.shape[0] % self.batch_size == 0:
            max_batch = self.X.shape[0] // self.batch_size
        else:
            max_batch = self.X.shape[0] // self.batch_size + 1

        for i in range(max_batch):
            yield (self.X[index[i*self.batch_size:(i+1)*self.batch_size]],
                   self.y[index[i*self.batch_size:(i+1)*self.batch_size]])
