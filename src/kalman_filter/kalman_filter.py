import numpy
import numpy as np

# for future robustness if more varibles need to be added to state vector
# # index of each variable in the state vector
# iX = 0
# iV = 1
# NUMVARS = iV + 1

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
        # P = F P F^T + G G^T a
        F = np.array([[1, dt], [0, 1]])
        new_x = F.dot(self._x)

        G = np.array([.5 * dt**2, dt]).reshape(2, 1)
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._a_var

        self._P = new_P
        self._x = new_x

    def update(self, meas_value: float, meas_var: float) -> None:
        # y = z - H x
        # S = H P H^T + R
        # K = P H^T S^-1
        # x = x + K y
        # P = (I - K H) P

        z = np.array([meas_value])
        R = np.array([meas_var])

        H = np.array([1,0]).reshape((1, 2))

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self._P)

        self._x = new_x
        self._P = new_P

    def twoSidedConfidenceInterval(self, int) -> numpy.array:
        return numpy.array([self._x[0] - 1.96*np.sqrt(self._P[0, 0]), self._x[0] + 1.96*np.sqrt(self._P[0, 0])])

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

