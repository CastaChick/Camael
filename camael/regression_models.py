import numpy as np
from numpy import linalg as LA


class LinearRegression:
    """
    線形回帰を行うモデル

    Parameters
    ----------
    intercept: boolean(default=True)
        切片要素を入れるかどうか

    Examples
    ---------
    >>> from load_data import load_boston
    >>> (X_train, y_train), (X_test, y_test) = load_boston()
    >>> reg = LinearRegression()
    >>> reg.fit(X_train, y_train)
    >>> reg.b
    array([ 4.02936706e+01, -1.19997513e-01,  5.70003304e-02,  3.98379660e-03,
            4.12698187e+00, -2.05002963e+01,  3.38024903e+00,  7.56807584e-03,
           -1.71189793e+00,  3.34747537e-01, -1.17797225e-02, -9.02318039e-01,
            8.71912756e-03, -5.55842510e-01])
    >>> reg.score(X_test, y_test)
    23.19559925642053
    """
    def __init__(self, intercept=True):
        self.intercept = intercept

    def fit(self, X, y):
        """
        学習データにフィットさせる

        Parameters
        ----------
        X: array, shape=(samples, colunms)
            説明変数の行列

        y: vector, len=(samples)
            目的変数のベクトル

        Attributes
        ----------
        b: vector
            係数のベクトル

        _error: float
            最適な係数の誤差
        """
        if self.intercept:
            self.b = self._solve(np.hstack((np.ones((X.shape[0], 1)), X)), y)
        else:
            self.b = self._solve(X, y)

        self._error = self._culc_error(y, self.predict(X))

    def predict(self, X):
        """
        fitメソッドで算出した係数ベクトルを用いて予測を行う

        Parameters
        ----------
        X: array, shape=(samples, columns)
            予測したいデータの説明変数

        Returns
        -------
        y: vector, len=(samples)
            予測された目的変数
        """
        if self.intercept:
            y = np.hstack((np.ones((X.shape[0], 1)), X)).dot(self.b)
        else:
            y = X.dot(self.b)

        return y

    def _solve(self, X, y):
        return LA.inv(X.T.dot(X)).dot(X.T.dot(y))

    def _culc_error(self, y, y_pred):
        return LA.norm(y - y_pred)**2 / y.shape[0]

    def score(self, X, y):
        """
        モデルの平均二乗誤差を求める

        Parameters
        ----------
        X: array, shape=(samples, columns)
            説明変数の行列

        y: vector, len=(samples)
            目的変数のベクトル

        Returns
        -------
        error: float
            誤差
        """
        return self._culc_error(y, self.predict(X))


class KNNRegression:
    """
    K近傍法(k-nearest neighbor)による回帰を行う

    Parameters
    ----------
    k: int (default=5)
        考慮する最近傍データの数

    weights: str (default="same")
        重み付けの有無(デフォルトは重み付け無し)

        距離に応じた重みを考慮するときは"distance"を指定

    practice: int (default=2)
        距離計算方法

        * 1:  マンハッタン距離
        * 2:  ユークリッド距離
        * <3: 任意の次元のミンコフスキー距離

    Examples
    --------
    >>> from load_data import load_boston
    >>> (X_train, y_train), (X_test, y_test) = load_boston()
    >>> reg = KNNRegression()
    >>> reg.fit(X_train, y_train)
    >>> reg.score(X_test, y_test)
    36.969313725490196
    """
    def __init__(self, k=5, weight="same", practice=2):
        if type(k) is not int:
            raise TypeError(
                    "k should be int.")
        if weight not in ["same", "distance"]:
            raise ValueError(
                    "weight not recognized: should be 'same' or 'distance'.")
        if type(practice) is not int:
            raise TypeError(
                    "practice should be int.")

        self.k = k
        self.weight = weight
        self.practice = practice

    def fit(self, X, y):
        """
        学習データをインプットする

        Parameters
        ----------
        X: array, shape=(samples, columns)
            説明変数の行列

        y: vector, len=(samples)
            目的変数のベクトル
        """
        self.X = X
        self.y = y

    def _culc_distance(self, sample):
        """
        あるsampleについてトレーニングデータとの距離を求める

        Parameters
        ----------
        sample: vector
            サンプルの特徴量を並べたベクトル

        Returns
        -------
        distance: vector
            各トレーニングデータとの距離
        """
        distance = np.abs(self.X - sample)**self.practice
        return np.sum(distance, axis=1)

    def predict(self, samples):
        """
        複数のsampleについて予測を行う

        Parameters
        ----------
        samples: array, shape=(samples, columns)
            予測したいサンプルの行列

        Returns
        -------
        y: vector, len=(samples)
            予測されたカテゴリ
        """
        y = np.zeros(samples.shape[0])
        for i, sample in enumerate(samples):
            y[i] = self._predict_one(sample)
        return y

    def _predict_one(self, sample):
        """
        １つのサンプルがどのカテゴリに入っているかを確認する

        Parameters
        ----------
        sample: vector
            サンプルの特徴量を並べたベクトル

        Returns
        -------
        result: int
            予測されたカテゴリ番号
        """
        dis = self._culc_distance(sample)
        index = np.arange(self.X.shape[0])
        index = index[np.argsort(dis, axis=0)]
        result = 0

        if self.weight == "same":
            for i in range(self.k):
                result += self.y[index[i]]
            result /= self.k
        elif self.weight == "distance":
            distance = 0
            for i in range(self.k):
                result += self.y[index[i]] * dis[index[i]]
                distance += dis[index[i]]
            result /= distance

        return result

    def score(self, X, y):
        """
        モデルの平均二乗誤差を求める

        Parameters
        ----------
        X: array, shape=(samples, columns)
            説明変数の行列

        y: vector, len=(samples)
            目的変数のベクトル

        Returns
        -------
        error: float
            誤差
        """
        return self._culc_error(y, self.predict(X))

    def _culc_error(self, y, y_pred):
        return LA.norm(y - y_pred)**2 / y.shape[0]
