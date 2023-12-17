
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

# EKF Class Definition
class EKF:
    # Initialization
    def __init__(self, Ts, Cj, P_dach, R, GQG, x_dach):
        self.Ts = Ts
        self.Cj = Cj
        self.P_dach = P_dach
        self.R = R
        self.GQG = GQG
        self.x_dach = x_dach

    # Update Method
    def update(self, y):
        N = y.shape[0]  # Number of measurements
        # Initialize state vectors
        xP = np.zeros(N)
        yP = np.zeros(N)
        alpha = np.zeros(N)
        Kr = np.zeros(N)
        v = np.zeros(N)

        # EKF update loop
        for k in range(N):
            # Prediction step omitted for brevity
            # Correction step omitted for brevity

            # Store results
            xP[k], yP[k], alpha[k], Kr[k], v[k] = self.x_dach
            # Update state and covariance omitted for brevity

        return xP, yP, alpha, Kr, v

# Function to read CSV data
def read_csv(path):
    return np.loadtxt(path, delimiter=',', skiprows=1)

# Function to write results to a CSV file
def write_csv(data, path):
    header = 'xP,yP,alpha,Kr,v'
    np.savetxt(path, data, delimiter=',', header=header, comments='')

# Function to plot results
def plot_ekf_results(GT, xP, yP, alpha, Kr, v, t, output_path):
    plt.figure(figsize=(12, 10), dpi=150)
    # Plot estimated path vs ground truth omitted for brevity
    # Plot alpha, Kr, v time series omitted for brevity
    plt.savefig(output_path)
    plt.close()

# Argument Parser for command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Extended Kalman Filter for Robot Localization.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file with measurements.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV file for results.')
    parser.add_argument('--output_plot', type=str, required=True, help='Path to output plot file.')
    return parser.parse_args()

# Main function
def main():
    args = parse_arguments()

    # Read the input data
    data = read_csv(args.input_csv)

    # Initialize EKF
    Ts = 1.0  # Sampling time
    Cj = np.eye(5)  # Measurement matrix
    P_dach = np.eye(5) * 100  # Initial covariance
    R = np.eye(5) * 10  # Measurement noise
    GQG = np.eye(5)  # Process noise
    x_dach = np.zeros(5)  # Initial state estimate

    ekf = EKF(Ts, Cj, P_dach, R, GQG, x_dach)

    # Run EKF update with measurements
    xP, yP, alpha, Kr, v = ekf.update(data)

    # Write results to CSV
    results = np.column_stack((xP, yP, alpha, Kr, v))
    write_csv(results, args.output_csv)

    # Generate and save plots
    t = np.arange(N) * Ts  # Time vector
    GT = {'x': data[:,0], 'y': data[:,1], 'alpha': data[:,2], 'Kr': data[:,3], 'v': data[:,4]}
    plot_ekf_results(GT, xP, yP, alpha, Kr, v, t, args.output_plot)

if __name__ == '__main__':
    main()
