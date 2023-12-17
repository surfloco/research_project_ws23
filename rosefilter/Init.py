
import numpy as np

# Constants and signals
D = 0.1
Ts = 0.1

# Velocity and curvature profiles
vv = np.concatenate((np.zeros(100), np.arange(0.2, 2.2, 0.2), np.full(350, 2), 
                     np.arange(2.1, 3.1, 0.1), np.full(100, 3), np.arange(2.8, 0.8, -0.2),
                     np.full(350, 1), np.arange(0.9, -0.1, -0.1), np.zeros(60)))

kk = np.concatenate((np.zeros(150), np.arange(0.05, 0.45, 0.05), np.full(30, 0.4),
                     np.arange(0.35, -0.05, -0.05), np.zeros(80), np.arange(-0.1, -0.9, -0.1),
                     np.full(11, -0.8), np.arange(-0.7, 0.1, 0.1), np.zeros(100),
                     np.arange(-0.02, -0.22, -0.02), np.full(55, -0.2), np.arange(-0.18, 0.02, 0.2),
                     np.zeros(35), np.arange(0.02, 0.22, 0.02), np.full(55, 0.2),
                     np.arange(0.18, -0.02, -0.2), np.zeros(145), np.arange(0.01, 0.11, 0.01),
                     np.full(165, 0.1), np.arange(0.09, -0.01, -0.01), np.zeros(100)))

# Calculated velocities
vl = vv / (1 - kk * 0.5 * D)
vr = vv / (1 + kk * 0.5 * D)

# Time array
t = Ts * np.arange(1, len(vv) + 1)

# Initialization of state variables
xx = np.zeros_like(vv)
yy = np.zeros_like(vv)
aa = np.full_like(vv, 0.3)
sl = np.full_like(vv, 0.0211)
sr = np.zeros_like(vv)

# Simulation loop to calculate the trajectory
for k in range(1, len(vv)):
    xx[k] = xx[k-1] - vv[k-1] * Ts * np.sin(aa[k-1])
    yy[k] = yy[k-1] + vv[k-1] * Ts * np.cos(aa[k-1])
    aa[k] = aa[k-1] + vv[k-1] * Ts * kk[k-1]
    sl[k] = sl[k-1] + vl[k-1] * Ts
    sr[k] = sr[k-1] + vr[k-1] * Ts

# Rounding sl and sr to two decimal places
sl = np.round(sl * 100) / 100
sr = np.round(sr * 100) / 100
