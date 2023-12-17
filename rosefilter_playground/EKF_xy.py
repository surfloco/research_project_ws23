
import numpy as np
import pandas as pd

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

# Placeholder for the loop that will contain the filter update iterations
# ... (Loop logic will be added here)

# The filter update iterations
for k in range(len(y)):
    # Innovation vector
    dy = y[k] - Cj @ x_dach
    
    # Measurement update
    M = Cj @ P_dach @ Cj.T + R
    invM = np.linalg.inv(M)  # More stable inversion in Python
    K = P_dach @ Cj.T @ invM
    
    # State update
    x_dach = x_dach + K @ dy
    
    # Covariance update
    P_dach = P_dach - K @ Cj @ P_dach
