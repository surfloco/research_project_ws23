

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



def ekf_update_loop(y, Ts, Cj, P_dach, R, GQG, x_dach):
    # Define the length of the measurement vector y
    N = y.shape[1]

    # Initialize arrays to store the results
    xP = np.zeros(N)
    yP = np.zeros(N)
    alpha = np.zeros(N)
    Kr = np.zeros(N)
    v = np.zeros(N)

    # Start the loop over the measurements
    for k in range(N):
        dy = y[:, k] - np.dot(Cj, x_dach)
        M = np.dot(np.dot(Cj, P_dach), Cj.T) + R
        invM = np.linalg.inv(M)
        K = np.dot(np.dot(P_dach, Cj.T), invM)
        x_tilde = x_dach + np.dot(K, dy)
        P_tilde = (np.eye(len(x_dach)) - np.dot(K, Cj)) @ P_dach @ (np.eye(len(x_dach)) - np.dot(K, Cj)).T + np.dot(np.dot(K, R), K.T)
        
        xP[k] = x_tilde[0]
        yP[k] = x_tilde[1]
        alpha[k] = x_tilde[2]
        Kr[k] = x_tilde[3]
        v[k] = x_tilde[4]
        
        x_dach = np.array([
            xP[k] - v[k] * Ts * np.sin(alpha[k]),
            yP[k] + v[k] * Ts * np.cos(alpha[k]),
            alpha[k] + v[k] * Ts * Kr[k],
            Kr[k],
            v[k]
        ])
        
        Aj = np.array([
            [1, 0, -v[k] * Ts * np.cos(alpha[k]), 0, -Ts * np.sin(alpha[k])],
            [0, 1, -v[k] * Ts * np.sin(alpha[k]), 0, Ts * np.cos(alpha[k])],
            [0, 0, 1, v[k] * Ts, Kr[k] * Ts],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])
        
        P_dach = np.dot(np.dot(Aj, P_tilde + GQG), Aj.T)
    
    return xP, yP, alpha, Kr, v


# Assuming that the variables defined in the init script match the expected names in the EKF loop,
# otherwise you will need to map the initialized variable names to the names used in the ekf_update_loop

# Call the EKF update loop function with the initialized variables
xP, yP, alpha, Kr, v = ekf_update_loop(y, Ts, Cj, P_dach, R, GQG, x_dach)

# Plotting code
import matplotlib.pyplot as plt

# Assuming we have the ground truth data initialized as 'GT'
GT = {
    'x': xx,
    'y': yy,
    'alpha': aa,
    'Kr': kk,
    'v': vv
}

# Plotting the estimated path and the ground truth with higher resolution
plt.figure(figsize=(12, 10), dpi=150)

# Plot the measurements (assuming 'y' contains them), the estimated path, and the ground truth
plt.subplot(2, 1, 1)
plt.plot(GT['x'], GT['y'], 'r', label='Ground Truth')
plt.plot(xP, yP, 'g-*', label='Estimated Path')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Estimated Path vs Ground Truth')
plt.axis('equal')
plt.legend()

# Subplots for alpha, Kr, and v
plt.subplot(2, 3, 4)
plt.plot(t, alpha, 'g-', label='Estimated alpha')
plt.plot(t, GT['alpha'], 'r', label='True alpha')
plt.xlabel('Time')
plt.ylabel('Alpha')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(t, Kr, 'g-', label='Estimated Kr')
plt.plot(t, GT['Kr'], 'r', label='True Kr')
plt.xlabel('Time')
plt.ylabel('Kr')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(t, v, 'g-', label='Estimated v')
plt.plot(t, GT['v'], 'r', label='True v')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()

# Adjust layout
plt.tight_layout()

# Save the figure to a file with higher resolution
plt.savefig('ekf_estimation_plots.png', dpi=150)

# Show the plot
plt.show()
