
import numpy as np
import pandas as pd

# Read data from '/mnt/data/xy.csv'
data = pd.read_csv('/mnt/data/xy.csv')

# Extracting data and transpose
GT = {
    'x': data['GT_x'].values,
    'y': data['GT_y'].values,
    'alpha': data['GT_alpha'].values,
    'Kr': data['GT_Kr'].values,
    'v': data['GT_v'].values
}
t = data['time'].values
Ts = t[1] - t[0]

# Adding noise to alpha
aa = GT['alpha'] + np.sqrt(1e-2) * np.random.randn(len(t))

# Measurement vector y
y = data[['Pos_x', 'Pos_y']].T.values

# Initial calculations for Kalman filter settings
R0 = 1
Q0 = 11
lambda_ = Ts * np.sqrt(Q0 / R0)
K1 = -1 / 8 * (lambda_**2 + 8 * lambda_ - (lambda_ + 4) * np.sqrt(lambda_**2 + 8 * lambda_))
K2 = 0.25 * (lambda_**2 + 4 * lambda_ - lambda_ * np.sqrt(lambda_**2 + 8 * lambda_)) / Ts

K0 = np.array([K1, K2])
H = np.array([[1 - K1, Ts - K1 * Ts], [-K2, 1 - K2 * Ts]])

# Constants for ROSE filter initialization
Gamma = 0.9          # Factor for measurement noise
Alpha_R = 0.08       # Smoothing factor measurement noise

# The rest of the EKF and ROSE filter initialization and update steps will be translated accordingly.

# Calculate covariance matrix R using the initial part of the measurement vector y
R = Gamma * np.cov(y[:, :100])

# Noise covariance values
qxy = 2E-5
qa = 4E-4
qKr = 3E-4
qv = 5E-3

# Construction of the GQG matrix using the noise covariances
GQG = np.array([
    [qxy, 0, 0, 0, 0],
    [0, qxy, 0, 0, 0],
    [0, 0, qa, 0, 0],
    [0, 0, 0, qKr, 0],
    [0, 0, 0, 0, qv]
])

# Definition of the output matrix Cj for the Kalman filter
Cj = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0]
])

# Initial state estimate
x_dach = np.array([y[0, 0], y[1, 0], 0, 0, 0])

# Initial estimate covariance matrix
P_dach = np.array([
    [0.1, 0, 0, 0, 0],
    [0, 0.1, 0, 0, 0],
    [0, 0, 1e-2, 0, 0],
    [0, 0, 0, 1e-3, 0],
    [0, 0, 0, 0, 1e-2]
])

# Initial measurement noise covariance matrix (commented out in MATLAB)
# MM = np.dot(Cj, np.dot(P_dach, Cj.T))

# Initial state estimates for position and velocity in each dimension
x1 = np.array([y[0, 0], 0])
x2 = np.array([y[1, 0], 0])

# Initialization for arrays to store the estimated states and R values
xR = np.zeros(len(y[0]))
yR = np.zeros(len(y[0]))
r1 = np.zeros(len(y[0]))
r2 = np.zeros(len(y[0]))

# Loop through each measurement
for k in range(len(y[0])):
    # Update state estimates
    x1 = H @ x1 + K0 * y[0, k]
    xR[k] = x1[0]
    x2 = H @ x2 + K0 * y[1, k]
    yR[k] = x2[0]
    
    # Update R with exponential smoothing
    measurement_error = y[:, k] - np.array([x1[0], x2[0]])
    R = Gamma * Alpha_R * np.outer(measurement_error, measurement_error) + (1 - Alpha_R) * R
    r1[k] = R[0, 0]
    r2[k] = R[1, 1]
    
    # Measurement residual
    dy = y[:, k] - Cj @ x_dach
    
    # Calculate M matrix
    M = Cj @ P_dach @ Cj.T + R

# Smoothing factor for process noise (commented out in MATLAB)
# Alpha_M = 0.001

# Process noise covariance matrix update (commented out in MATLAB)
# MM = Alpha_M * np.outer(dy, dy) + (1 - Alpha_M) * MM

# Inverse of the M matrix using numpy.linalg.inv for a more general approach
# Here we use the more robust pinv (Moore-Penrose pseudo-inverse) which can handle singular matrices
invM = np.linalg.pinv(M)

# Kalman gain
K = P_dach @ Cj.T @ invM

# State estimate update
x_tilde = x_dach + K @ dy

# Estimate covariance matrix update (commented out in MATLAB)
# P_tilde = (np.eye(len(x_dach)) - K @ Cj) @ P_dach

# Constants for ROSE filter and EKF initialization
Gamma = 0.9          # Factor for measurement noise
Alpha_R = 0.08       # Smoothing factor measurement noise

# Measurement noise covariance matrix
R = Gamma * np.cov(y[:, :100])

# System noise covariance matrix and other noise related variables
qxy = 2e-5
qa = 4e-4
qKr = 3e-4
qv = 5e-3
GQG = np.diag([qxy, qxy, qa, qKr, qv])

# Output matrix of the Kalman Filter
Cj = np.array([[1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0]])

# Initial state estimate and covariance matrix
x_dach = np.array([y[0, 0], y[1, 0], 0, 0, 0])
P_dach = np.diag([0.1, 0.1, 1e-2, 1e-3, 1e-2])

# Initial states for the filter iterations
x1 = np.array([y[0, 0], 0])
x2 = np.array([y[1, 0], 0])

# Placeholder for the loop that will contain the filter update iterations
# ... (Loop logic will be added here)


# The filter update iterations
length_y = len(y[0])  # Assuming y is a 2D array with shape (2, N)
xR = np.zeros(length_y)
yR = np.zeros(length_y)

for k in range(length_y):
    # Update state vectors with the system matrix H and Kalman gain K0
    x1 = H @ x1 + K0 * y[0, k]
    xR[k] = x1[0]
    x2 = H @ x2 + K0 * y[1, k]
    yR[k] = x2[0]
    
    # Update the measurement noise covariance matrix R
    innovation = y[:, k] - np.array([x1[0], x2[0]])
    R = Gamma * Alpha_R * np.outer(innovation, innovation) + (1 - Alpha_R) * R

# (Any further processing will be added here)
