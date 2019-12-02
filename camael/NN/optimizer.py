class SGD:
    """
    確率的降下法による最適化

    Parameters
    ----------
    eta: float (default=0.1)
        学習率
    """
    def __init__(self, eta=0.1):
        self.eta = eta

    def _update(self, X, dx):
        return X - self.eta * dx


class Adam:
    """
    Adam法による最適化

    Parameters
    ----------
    eta: float (default=0.001)
        ステップ幅

    rho1: float[0, 1) (default=0.9)
        一次モーメントの指数減衰率

    rho2: float[0, 1) (default=0.999)
        二次モーメントの指数減衰率

    delta: float (default=1.0e-8)
        安定性のための小さな定数
    """
    def __init__(self,
                 eta=0.001,
                 rho1=0.9,
                 rho2=0.999,
                 delta=1.0e-8):
        self.eta = eta
        self.rho1 = rho1
        self.rho2 = rho2
        self.delta = delta
        self.s = 0.
        self.r = 0.
        self.t = 0

    def _update(self, X, dx):
        self.t += 1
        self.s = self.rho1*self.s + (1-self.rho1)*dx
        self.r = self.rho2*self.r + (1-self.rho2)*dx**2
        _s = self.s / (1 - self.rho1**self.t)
        _r = self.r / (1 - self.rho2**self.t)

        return X - self.eta*(_s / (_r**0.5 + self.delta))
