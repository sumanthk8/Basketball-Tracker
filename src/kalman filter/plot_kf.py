import numpy as np
import matplotlib.pyplot as plt

from kalman_filter import KF

plt.ion()
plt.figure()

kf = KF(initial_x=0.0, initial_v=1.0, accel_var=0.1)

DT = 0.1
NUM_STEPS = 1000

mus = []
covs = []

for i in range(NUM_STEPS):
    covs.append(kf.cov)
    mus.append(kf.mean)

    kf.predict(DT)

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

plt.show()
plt.ginput(1)