import numpy as np
import matplotlib.pyplot as plt

from kalman_filter import KF

plt.ion()
plt.figure()


real_x = 0
meas_var = 0.1**2
real_v = 0.5

kf = KF(initial_x=0.0, initial_v=1.0, accel_var=.1)

DT = 0.1
NUM_STEPS = 1000
MEAS_EVERY_STEPS = 20

mus = []
covs = []

for step in range(NUM_STEPS):

    if step > 500:
        real_v *= 0.9

    covs.append(kf.cov)
    mus.append(kf.mean)

    real_x = real_x + DT*real_v

    kf.predict(dt=DT)

    if step != 0 and step % MEAS_EVERY_STEPS == 0:
        kf.update(meas_value=real_x + np.random.randn()*np.sqrt(meas_var),
                  meas_var=meas_var)


plt.subplot(2, 1, 1)
plt.title("Position")
plt.plot([mu[0] for mu in mus], 'r')

#2-sided confidence interval for our estimation of position
plt.plot([mu[0] + 1.96*np.sqrt(covs[0,0]) for mu, covs in zip(mus, covs)], 'r--')
plt.plot([mu[0] - 1.96*np.sqrt(covs[0,0]) for mu, covs in zip(mus, covs)], 'r--')

plt.subplot(2, 1, 2)
plt.title("Velocity")
plt.plot([mu[1] for mu in mus], 'r')

#2-sided confidence interval for our estimation of velocity
plt.plot([mu[1] + 1.96*np.sqrt(covs[1,1]) for mu, covs in zip(mus, covs)], 'r--')
plt.plot([mu[1] - 1.96*np.sqrt(covs[1,1]) for mu, covs in zip(mus, covs)], 'r--')

plt.show(block=True)