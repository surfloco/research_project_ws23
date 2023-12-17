
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading data from the CSV file that was converted from 'xy.dat'
data = pd.read_csv('/mnt/data/xy.csv')

# Extracting data and assigning to variables
GT = {
    'x': data['GT_x'].values,
    'y': data['GT_y'].values,
    'alpha': data['GT_alpha'].values,
    'Kr': data['GT_Kr'].values,
    'v': data['GT_v'].values
}
t = data['time'].values
Ts = t[1] - t[0]
y = data[['Pos_x', 'Pos_y']].values

# EKF setup
# Measurement noise covariance matrix
R = np.cov(y[:100], rowvar=False)

# System noise covariance matrix
qxy = 2e-4
qa = 1e-4
qKr = 5e-4
qv = 5e-3
GQG = np.diag([qxy, qxy, qa, qKr, qv])

# Output matrix of the Kalman Filter
Cj = np.array([[1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0]])

# Initial state estimate and covariance matrix
x_dach = np.array([y[0, 0], y[0, 1], 0, 0, 0])
P_dach = np.diag([0.1, 0.1, 1e-2, 1e-3, 1e-2])

# Arrays to capture the position estimates and other variables
xP = np.zeros(len(y))
yP = np.zeros(len(y))
alpha = np.zeros(len(y))
Kr = np.zeros(len(y))
v = np.zeros(len(y))

# The filter update iterations
for k in range(len(y)):
    # Measurement update
    dy = y[k] - Cj @ x_dach
    M = Cj @ P_dach @ Cj.T + R
    invM = np.linalg.inv(M)
    K = P_dach @ Cj.T @ invM
    x_tilde = x_dach + K @ dy
    P_tilde = (np.eye(len(x_dach)) - K @ Cj) @ P_dach @ (np.eye(len(x_dach)) - K @ Cj).T + K @ R @ K.T
    
    # State transition (prediction step)
    xP[k] = x_tilde[0] - x_tilde[4] * Ts * np.sin(x_tilde[2])
    yP[k] = x_tilde[1] + x_tilde[4] * Ts * np.cos(x_tilde[2])
    alpha[k] = x_tilde[2] + x_tilde[4] * Ts * x_tilde[3]
    Kr[k] = x_tilde[3]
    v[k] = x_tilde[4]
    
    # Jacobian of the state transition
    Aj = np.array([
        [1, 0, -v[k]*Ts*np.cos(alpha[k]), 0, -Ts*np.sin(alpha[k])],
        [0, 1, -v[k]*Ts*np.sin(alpha[k]), 0, Ts*np.cos(alpha[k])],
        [0, 0, 1, v[k]*Ts, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ])
    
    # Predicted state estimate and covariance matrix update
    x_dach = np.array([xP[k], yP[k], alpha[k], Kr[k], v[k]])
    P_dach = Aj @ (P_tilde + GQG) @ Aj.T

# Plotting
plt.figure(1)
plt.clf()
plt.plot(y[:, 0], y[:, 1], 'bo', label='Measurements')
plt.plot(xP, yP, 'g-*', label='Estimated Path')
plt.plot(GT['x'], GT['y'], 'r', label='Ground Truth')
plt.axis('equal')
plt.legend()
plt.title('Measurements, Estimated Path, and Ground Truth with Prediction Step')
plt.show()
