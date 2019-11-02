import numpy as np
import cvxopt


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
        y: vector, len=(samples)
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


class LogisticRegressionClassifier:
    """
    ロジスティック回帰による分類を行う

    Parameters
    ----------
    max_iter: int (default=100)
        最大エポック数

    multi_class: boolean (default=False)
        他クラス分類するかどうか
        Trueの時はOvR分類を行う

    batch_size: int (default=1)
        バッチサイズ

    eta: float (default=0.01)
        学習率

    shuffle: boolean (default=True)
        エポックごとにデータをシャッフルするかどうか

    log: boolean (default=True)
        ログを出力するかどうか

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> from load_data import load_iris
    >>> X, y = load_iris()
    >>> clf = LogisticRegressionClassifier(multi_class=True, log=False)
    >>> clf.fit(X, y)
    >>> print("Acc: {:.3f}".format(clf.score(X, y)))
    Acc: 0.933
    """
    def __init__(self,
                 max_iter=100,
                 multi_class=False,
                 eta=0.01,
                 batch_size=1,
                 shuffle=True,
                 log=True):
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.eta = eta
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.log = log

    def fit(self, X, y):
        """
        モデルをデータに適合させる

        Parameters
        ----------
        X: array, shape=(samples, columns)
            学習データの特徴量

        y: vector, len=(samples)
            学習データの正解ラベル
        """
        self.X = np.hstack((X, np.ones((X.shape[0], 1))))
        if self.batch_size == -1:
            self.batch_size = self.X.shape[0]
        self.y = self._label_encode(y)
        self.w_list = []

        if self.multi_class:
            for i in range(len(self.label_to_vector)):
                if self.log:
                    print("Training No.{} classifier".format(i))
                self.w_list.append(
                    self._fit_one(np.array(list(map(int, y == i)))))
        else:
            self.w_list.append(self._fit_one(self.y))

    def _label_encode(self, y):  # 辞書を用意してラベルを0-indexedベクトルに変換する
        self.label_to_vector = \
            {label: i for i, label in enumerate(set(y))}
        self.vector_to_label = \
            {i: label for i, label in enumerate(set(y))}

        return np.array([self.label_to_vector[key] for key in y])

    def _output_func(self, X, w):
        """
        モデルの出力を計算する

        Parameters
        X: array, shape=(self.batch_size, columns)
            学習データのミニバッチ

        w: vector, len=(columns+1)
            現在の重みベクトル

        Returns
        -------
        outputs: vector, len=(self.batch_size)
            各サンプルの予測確率
        """
        net_input = X.dot(w)
        return 1/(1+np.exp(-net_input))

    def _fit_one(self, y):
        """
        二値分類器のトレーニング

        Parameters
        ----------
        y: vector, len=(samples)
            0 or 1の正解ラベル

        Returns
        -------
        w: 学習した重みベクトル
        """
        w = np.random.rand(self.X.shape[1])
        for epoch in range(self.max_iter):
            for X, labels in self._get_minibatch(y):
                diff_w = self._culc_diff(X, labels, w)
                w -= self.eta * diff_w

            if self.log:
                print("Epoch: {}".format(epoch))
                error = self._culc_error(X, labels, w)
                print("Err: {:.4f}".format(error))
                y_pred = self._output_func(X, w) >= 0.5
                acc = self._culc_acc(labels, y_pred)
                print("Acc: {:.3f}".format(acc))
        return w

    def _culc_diff(self, X, y, w):
        diffs = (self._output_func(X, w) - y).dot(X)
        return diffs/X.shape[0]

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
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        y = self._label_encode(y)
        return self._culc_acc(y, self.predict(X))

    def predict(self, X):
        """
        分類を行う

        Parameters
        ----------
        X: array, shape=(samples, columns)
            テストデータ
        """
        pred = [self._output_func(X, w_i) for w_i in self.w_list]
        pred_label = np.array(
            [self.vector_to_label[i] for i in np.argmax(pred, axis=0)]
            )
        return pred_label

    def _get_minibatch(self, y):
        index = np.arange(self.X.shape[0])
        np.random.shuffle(index)
        if self.X.shape[0] % self.batch_size == 0:
            max_batch = self.X.shape[0] // self.batch_size
        else:
            max_batch = self.X.shape[0] // self.batch_size + 1

        for i in range(max_batch):
            yield (self.X[index[i*self.batch_size:(i+1)*self.batch_size]],
                   y[index[i*self.batch_size:(i+1)*self.batch_size]])

    def _culc_acc(self, y, y_pred):
        return np.sum(y == y_pred) / y.shape[0]

    def _culc_error(self, X, y, w):
        return \
            np.sum(-y*np.log(self._output_func(X, w))
                   - (1-y)*np.log((1-self._output_func(X, w))))


class LinearSVC:
    """
    線形SVMによる分類を行う

    Parameters
    ----------
    C: float (default=1.0)
        正則化パラメータ

    multi_class: boolean (default=False)
        他クラス分類するかどうか
        Trueの時はOvR分類を行う

    log: boolean (default=True)
        ログを出力するかどうか

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> from load_data import load_iris
    >>> X, y = load_iris()
    >>> clf = LinearSVC(multi_class=True, log=False)
    >>> clf.fit(X, y)
    >>> print("Acc: {:.3f}".format(clf.score(X, y)))
    Acc: 0.940
    """
    def __init__(self, C=1.0, multi_class=False, log=True):
        self.C = C
        self.multi_class = multi_class
        self.log = log

    def fit(self, X, y):
        """
        モデルをデータに適合させる

        Parameters
        ----------
        X: array, shape=(samples, columns)
            学習データの特徴量

        y: vector, len=(samples)
            学習データの正解ラベル
        """
        self.X = X
        self.N = X.shape[0]
        self.M = X.shape[1]
        self.y = self._label_encode(y)
        self.beta_list = []

        if self.multi_class:
            for i in range(len(self.label_to_vector)):
                if self.log:
                    print("Training No.{} classifier".format(i))
                self.beta_list.append(
                    self._fit_one(np.array(list(map(self.encode, y == i)))))
        else:
            self.beta_list.append(
                self._fit_one(np.array(list(map(self.encode, self.y)))))

    def _label_encode(self, y):  # 辞書を用意してラベルを0-indexedベクトルに変換する
        self.label_to_vector = \
            {label: i for i, label in enumerate(set(y))}
        self.vector_to_label = \
            {i: label for i, label in enumerate(set(y))}

        return np.array([self.label_to_vector[key] for key in y])

    def _fit_one(self, y):
        """
        二値分類器のトレーニング

        Parameters
        ----------
        y: vector, len=(samples)
            0 or 1の正解ラベル

        Returns
        -------
        beta: 学習した重みベクトル
        """
        c = np.zeros(2*self.N)
        c[self.N:] += self.C
        y_ = y.reshape((1, -1))
        o = 0.
        G = np.vstack((-np.eye(self.N), np.eye(self.N)))
        Y = np.diag(y)
        H = Y.dot(self.X).dot(self.X.T).dot(Y.T)
        ones = -np.ones((self.N, 1))
        H, ones, G, c, y_, o = map(cvxopt.matrix, (H, ones, G, c, y_, o))
        cvxopt.solvers.options['show_progress'] = False
        alpha = \
            np.array(cvxopt.solvers.qp(H, ones, G, c, y_, o)['x']).reshape(-1)

        self._sup_vecs = []
        for i in range(self.N):
            if self.C > alpha[i] > 1.e-5:
                self._sup_vecs.append(i)

        beta = np.zeros(self.M)
        for i in range(self.N):
            beta += alpha[i] * y[i] * self.X[i]

        beta_0 = 0
        for i in self._sup_vecs:
            beta_0 += y[i] - self.X[i].dot(beta)
        beta_0 /= len(self._sup_vecs)

        return np.hstack((np.array(beta_0), beta))

    def _output_func(self, X, beta):
        return X.dot(beta[1:]) + beta[0]

    def predict(self, X):
        """
        分類を行う

        Parameters
        ----------
        X: array, shape=(samples, columns)
            テストデータ
        """
        pred = [self._output_func(X, beta_i) for beta_i in self.beta_list]
        pred_label = np.array(
            [self.vector_to_label[i] for i in np.argmax(pred, axis=0)]
            )

        return pred_label

    def _culc_acc(self, y, y_pred):
        return np.sum(y == y_pred) / y.shape[0]

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
        X = X
        self.y = self._label_encode(y)
        return self._culc_acc(y, self.predict(X))

    def encode(self, x):
        return 1. if x else -1.
