import numpy as np


class KF:
    def __init__(self,  initial_x: float,
                        initial_v: float,
                        accel_var: float) -> None:
        #mean of state GRV (gaussian random var)
        self._x = np.array([initial_x, initial_v])
        self._a_var = accel_var

        #covariance of state GRV
        self._P = np.eye(2)

    def predict(self, dt: float) -> None:
        # x_(k+1) = F * x_k
        # P = F P F^T + G a G^T
        F = np.array([[1, dt], [0, 1]])
        new_x = F.dot(self._x)

        G = np.array([.5 * dt**2, dt]).reshape(2, 1)
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._a_var

        self._P = new_P
        self._x = new_x

    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def pos(self) -> float:
        return self.x[0]

    @property
    def vel(self) -> float:
        return self.x[1]

