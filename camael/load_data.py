import pickle
import os
import numpy as np


def _load_data(file_path):
    dir_path = \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    path = os.path.join(dir_path, file_path)
    with open(path, "rb") as f:
        dataset = pickle.load(f)

    return dataset


def load_mnist():
    """mnistデータセットを読み込む

    pickle形式のバイナリファイルからmnistデータセットを読み込む関数

    Returns
    -------
    dataset: tuple
        (X_train, y_train), (X_test, y_test)の要素を持つタプル

        * X : 手書きの数字が書かれた28＊28のグレースケール画像
        * y : 0~9のカテゴリラベル

    Examples
    --------
    >>> (X_train, y_train), (X_test, y_test) = load_mnist()
    """
    file_path = "mnist_data.binaryfile"

    return _load_data(file_path)


def load_cifar10():
    """cifar10データセットを読み込む

    pickle形式のバイナリファイルからcifar10データセットを読み込む関数

    Returns
    -------
    dataset: tuple
        (X_train, y_train), (X_test, y_test)の要素を持つタプル

        * X : 様々な物体を映した32＊32のRGB画像
        * y : 0~9のカテゴリラベル

    Examples
    --------
    >>> (X_train, y_train), (X_test, y_test) = load_cifar10()
    """
    dir_path = \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    path_li = [os.path.join(dir_path, "cifar10_data_0.binaryfile"),
               os.path.join(dir_path, "cifar10_data_1.binaryfile"),
               os.path.join(dir_path, "cifar10_data_2.binaryfile"),
               os.path.join(dir_path, "cifar10_data_3.binaryfile"),
               os.path.join(dir_path, "cifar10_data_4.binaryfile")]

    X_train = np.concatenate([_load_data(path) for path in path_li])
    X_test = _load_data(os.path.join(dir_path, "cifar10_test.binaryfile"))
    y_train, y_test = \
        _load_data(os.path.join(dir_path, "cifar10_target.binaryfile"))

    return (X_train, y_train.flatten()), (X_test, y_test.flatten())


def load_fashion_mnist():
    """fashion_mnistデータセットを読み込む

    pickle形式のバイナリファイルからfashion_mnistデータセットを読み込む関数

    Returns
    -------
    dataset: tuple
        (X_train, y_train), (X_test, y_test)の要素を持つタプル

        * X : 様々なファッショングッズを映した28＊28のグレースケール画像
        * y : 0~9のカテゴリラベル
    Examples
    --------
    >>> (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
    """
    file_path = "fashion_mnist_data.binaryfile"

    return _load_data(file_path)


def load_iris():
    """irisデータセットを読み込む

    pickle形式のバイナリファイルからirisデータセットを読み込む関数

    Returns
    -------
    dataset: tuple
        X, yの要素を持つタプル

        * X : 3種類のアヤメの花における4種類の特徴量
        * y : 0, 1, 2のカテゴリラベル
    Examples
    --------
    >>> (X, y) = load_iris()
    """
    file_path = "iris_data.binaryfile"

    return _load_data(file_path)


def load_boston():
    """bostonデータセットを読み込む

    pickle形式のバイナリファイルからbostonデータセットを読み込む関数

    Returns
    -------
    dataset: tuple
        (X_train, y_train), (X_test, y_test)の要素を持つタプル

        * X : bostonの各地域における住宅に関する13種類の特徴量
        * y : boostonのその地域の住宅価格の中央値
    Examples
    --------
    >>> (X_train, y_train), (X_test, y_test) = load_boston()
    """
    file_path = "boston_data.binaryfile"

    return _load_data(file_path)
