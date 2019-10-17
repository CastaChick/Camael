import numpy as np


def train_test_split(*arrays, rate=0.2, random_state=None, shuffle=True):
    """トレーニングデータとテストデータを分割する関数

    Parameters
    ----------
    *arrays: sequence of arrays
        同じサンプル数を持ついくつかの行列

    rate: float(default=0.2)
        全体に対する分割後のテストデータの割合

    random_state: int or None(default=None)
        乱数発生のシード値

    shuffle: boolean(default=True)
        分割前にデータをシャッフルするかどうか

    Returns
    -------
    splited_data: list
        与えられた行列の数 * 2 の長さを持つ

        [array0_train, array0_test, array1_train, array1_test, ...]のような形式

    Examples
    --------
    >>> X = np.arange(10)
    >>> y = np.array([[i] * 3 for i in range(10)])
    >>> X
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> y
    array([[0, 0, 0],
           [1, 1, 1],
           [2, 2, 2],
           [3, 3, 3],
           [4, 4, 4],
           [5, 5, 5],
           [6, 6, 6],
           [7, 7, 7],
           [8, 8, 8],
           [9, 9, 9]])
    >>> X_train, X_test, y_train, y_test = \
        train_test_split(X, y, shuffle=False)
    >>> X_train
    array([0, 1, 2, 3, 4, 5, 6, 7])
    >>> X_test
    array([8, 9])
    >>> y_train
    array([[0, 0, 0],
           [1, 1, 1],
           [2, 2, 2],
           [3, 3, 3],
           [4, 4, 4],
           [5, 5, 5],
           [6, 6, 6],
           [7, 7, 7]])
    >>> y_test
    array([[8, 8, 8],
           [9, 9, 9]])
    """
    num_arrays = len(arrays)
    num_samples = len(arrays[0])
    if random_state:
        np.random.seed(seed=random_state)

    index_list = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(index_list)

    split_index = int(num_samples * (1 - rate))

    splited_data = []

    for array in arrays:
        splited_data.append(
            np.array([array[i] for i in index_list[:split_index]])
        )
        splited_data.append(
            np.array([array[i] for i in index_list[split_index:]])
        )

    assert num_arrays * 2 == len(splited_data)

    return splited_data


class Standardization:
    """データの標準化を行う

    データを標準化する

    デフォルトでは各パラメータについて平均0, 分散1になるようにアフィン変換

    Parameters
    ----------
    mean: float or list (default=0)
        標準化後のデータの平均値

        リストの場合は各特徴量の平均値を順に並べたもの


    variance: float or list (default=1)
        標準化後のデータの分散

        リストの場合は各特徴量の分散を順に並べたもの

    Attributes
    ----------
    _mean: float or vector
        元データの各特徴量の平均値

    _variance: float or vector
        元データの各特徴量の分散

    Examples
    --------
    >>> from load_data import load_boston
    >>> (X_train, _), (X_test, _) = load_boston()
    >>> transformer = Standardization()
    >>> transformer.fit(X_train)
    >>> X_train_std, X_test_std = \
        transformer.transform(X_train, X_test)
    >>> np.mean(X_train_std, axis=0)
    array([-1.01541438e-16,  1.09923072e-17,  1.74337992e-15, -1.26686340e-16,
           -5.25377321e-15,  6.41414864e-15,  2.98441140e-16,  4.94653823e-16,
            1.12671149e-17, -1.98136337e-16,  2.36686358e-14,  5.95679996e-15,
            6.13920356e-16])
    >>> np.mean(X_test_std, axis=0)
    array([-0.0707286 , -0.02435885,  0.02358875,  0.1500709 , -0.11267862,
            0.12282991, -0.07746073,  0.13399985,  0.0621344 ,  0.06981759,
           -0.04617659,  0.09979472, -0.06008184])
    >>> np.var(X_train_std, axis=0)
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    >>> np.var(X_test_std, axis=0)
    array([0.33560812, 0.81778861, 1.07377123, 1.52324933, 0.87327937,
           0.89066136, 1.07193268, 1.36413552, 1.00955956, 1.12845533,
           0.84220326, 0.70192238, 0.84553242])
    """
    def __init__(self, maen=0.0, variance=1.0):
        self.mean = maen
        self.variance = variance

    def fit(self, array, axis=0):
        """データの平均値, 分散を得る

        データの平均値と分散を算出する

        トレーニングデータとテストデータを同じパラメータで標準化するため

        Parameters
        ----------
        array: array shape=(samples, columns)
            代表値を算出する行列

        axis: int(default=0)
            平均, 分散を計算する軸
        """
        self._mean = np.mean(array, axis=axis)
        self._variance = np.var(array, axis=axis)
        if type(self.mean) is float:
            self.mean = np.array([self.mean] * self._mean.shape[0])
        if type(self.variance) is float:
            self.variance = np.array([self.variance] * self._variance.shape[0])

    def transform(self, *arrays):
        """データの標準化を行う

        実際にデータを標準化する

        Parameters
        ----------
        *arrays: sequence of arrays
            標準化を行う行列

            同じカラム数を持つ幾つかの行列

        Returns
        -------
        transformed_arrays: list
            標準化された行列のリスト
        """
        transformed_arrays = []

        for array in arrays:
            transformed_arrays.append(
                (array - self._mean) / self._variance**0.5
            )

        return transformed_arrays

    def fit_transform(self, *arrays, axis=0):
        """代表値の計算と標準化を連続して行う

        引数に指定された行列の最初の行列を用いて代表値を計算する

        その代表値を用いて引数の行列を標準化する

        Parameters
        ----------
        *arrays: sequense of arrays
            標準化を行う行列の組

            １つ目の行列を基準に標準化を行う

        Returns
        -------
        transformed_arrays: list
            標準化された行列のリスト
        """

        self.fit(arrays[0], axis=axis)
        return self.transform(arrays)
