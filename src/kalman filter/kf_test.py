from kalman_filter import KF
import unittest
import numpy as np

class TestKF(unittest.TestCase):
    def test_can_construct(self):
        x = 0.2
        v = 2.3

        kf = KF(initial_x=x, initial_v=v, accel_var = .2)
        self.assertEqual(x, kf.pos)
        self.assertEqual(v, kf.vel)

    def test_can_predict(self):
        x = 0.2
        v = 2.3

        kf = KF(initial_x=x, initial_v=v, accel_var=.2)
        kf.predict(.5)

    def test_mean_and_P_right_shape_after_predict(self):
        x = 0.2
        v = 2.3

        kf = KF(initial_x=x, initial_v=v, accel_var=.2)
        kf.predict(dt=.5)

        self.assertEqual(kf.cov.shape, (2, 2))
        self.assertEqual(kf.mean.shape, (2,))

    def test_after_calling_predict_increases_state_uncertainty(self):
        x = 0.2
        v = 2.3

        kf = KF(initial_x=x, initial_v=v, accel_var=.2)

        for i in range(10):
            det_before = np.linalg.det(kf.cov)
            kf.predict(dt=0.1)
            det_after = np.linalg.det(kf.cov)

            self.assertGreater(det_after, det_before)

