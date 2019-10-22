import numpy as np


class KNNClassifier:
    """
    K近傍法(k-nearest neighbor)による分類を行う

    Parameters
    ----------
    k: int (default=5)
        考慮する最近傍データの数

    weights: str (default="same")
        重み付けの有無(デフォルトは重み付け無し)

        距離に応じた重みを考慮するときは"distance"を指定

    category: str (default="label")
        カテゴリ変数の形式

        正解ラベルがone-hot表現の時は"one-hot"を指定

    practice: int (default=2)
        距離計算方法

        * 1:  マンハッタン距離
        * 2:  ユークリッド距離
        * <3: 任意の次元のミンコフスキー距離

    Examples
    --------
    >>> from load_data import load_iris
    >>> X, y = load_iris()
    >>> clf = KNNClassifier()
    >>> clf.fit(X, y)
    >>> print("acc: {:.3f}".format(clf.score(X, y)))
    acc: 0.967
    """
    def __init__(self, k=5, weight="same", category="label", practice=2):
        if type(k) is not int:
            raise TypeError(
                    "k should be int.")
        if weight not in ["same", "distance"]:
            raise ValueError(
                    "weight not recognized: should be 'same' or 'distance'.")
        if type(practice) is not int:
            raise TypeError(
                    "practice should be int.")
        if category not in ["label", "one-hot"]:
            raise ValueError(
                    "category not recognized: should be 'label' or 'one-hot'.")

        self.k = k
        self.weight = weight
        self.category = category
        self.practice = practice

    def fit(self, X, y):
        """
        学習データをインプットする

        Parameters
        ----------
        X: array, shape=(samples, columns)
            説明変数の行列

        y: vector or array, shape=(samples, ?)
            * category="label"の時 -> vector
            * category="one-hot"の時 -> array
        """
        self.X = X

        if self.category == "label":
            self.labels = y
        elif self.category == "one-hot":
            self.labels = self._decode(y)

        self.label_to_index = \
            {label: i for i, label in enumerate(set(self.labels))}
        self.index_to_label = \
            {i: label for i, label in enumerate(set(self.labels))}

    def _decode(self, one_hot_labels):
        """
        one-hot表現からベクトル形式のラベルに変換する

        Parameters
        ----------
        one_hot_labels: array
            one-hot表現のカテゴリラベル

        Returns
        -------
        labels: vector
            ベクトル形式にデコードしたラベル
        """
        n_cat = one_hot_labels.shape[1]
        label_dic = \
            {i: np.zeros(n_cat, dtype=np.uint8) for i in range(n_cat)}

        for i in range(n_cat):
            label_dic[i][i] += 1

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
        y: vecttor, len=(samples)
            予測されたカテゴリ
        """
        y = np.zeros(samples.shape[0], dtype=np.uint8)
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
        result_vec = np.zeros(len(self.label_to_index))

        if self.weight == "same":
            for i in range(self.k):
                result_vec[self.label_to_index[self.labels[index[i]]]] += 1
        elif self.weight == "distance":
            for i in range(self.k):
                result_vec[self.label_to_index[self.labels[index[i]]]] \
                    += 1/dis[index[i]] if dis[index[i]] != 0 else 0

        return self.index_to_label[np.argmax(result_vec)]

    def score(self, X, y):
        """
        モデルの正解率を求める

        Parameters
        ----------
        X: array, shape=(samples, coumns)
            説明変数の行列

        y: vector, len=(samples)
            正解カテゴリの行列

        Returns
        -------
        acc: float
            正解率
        """
        return self._culc_acc(y, self.predict(X))

    def _culc_acc(self, y, y_pred):
        return sum(y == y_pred) / y.shape[0]
