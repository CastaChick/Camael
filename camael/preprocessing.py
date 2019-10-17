import numpy as np


def train_test_split(*arrays, rate=0.2, random_state=None, shuffle=True):
    """トレーニングデータとテストデータを分割する関数

    Parameters
    ----------
    *arrays: 同じサンプル数を持ついくつかの行列

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
