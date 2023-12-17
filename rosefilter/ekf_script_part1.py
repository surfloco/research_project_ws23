
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

class EKF:
    def __init__(self, Ts, Cj, P_dach, R, GQG, x_dach):
        self.Ts = Ts
        self.Cj = Cj
        self.P_dach = P_dach
        self.R = R
        self.GQG = GQG
        self.x_dach = x_dach

    def update(self, y):
        N = y.shape[1]  # Number of measurements
        xP = np.zeros(N)
        yP = np.zeros(N)
        alpha = np.zeros(N)
        Kr = np.zeros(N)
        v = np.zeros(N)

        for k in range(N):
            dy = y[:, k] - np.dot(self.Cj, self.x_dach)
            M = np.dot(np.dot(self.Cj, self.P_dach), self.Cj.T) + self.R
            invM = np.linalg.inv(M)
            K = np.dot(np.dot(self.P_dach, self.Cj.T), invM)
            x_tilde = self.x_dach + np.dot(K, dy)
            P_tilde = np.dot(np.dot(np.eye(len(self.x_dach)) - np.dot(K, self.Cj), self.P_dach), np.eye(len(self.x_dach)) - np.dot(K, self.Cj).T) + np.dot(K, self.R).dot(K.T)
            self.x_dach = x_tilde
            Aj = np.array([[1, 0, -self.v[k] * self.Ts * np.cos(self.alpha[k]), 0, -self.Ts * np.sin(self.alpha[k])],
                           [0, 1, -self.v[k] * self.Ts * np.sin(self.alpha[k]), 0, self.Ts * np.cos(self.alpha[k])],
                           [0, 0, 1, self.v[k] * self.Ts, self.Kr[k] * self.Ts],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]])
            self.P_dach = np.dot(np.dot(Aj, P_tilde), Aj.T) + self.GQG
            xP[k], yP[k], alpha[k], Kr[k], v[k] = self.x_dach
        return xP, yP, alpha, Kr, v

def read_csv(path):
    return np.loadtxt(path, delimiter=',', skiprows=1)

def write_csv(data, path):
    header = 'x,y,alpha,Kr,v'
    np.savetxt(path, data, delimiter=',', header=header, comments='')

def plot_ekf_results(xP, yP, alpha, Kr, v, output_path):
    plt.figure(figsize=(12, 10), dpi=150)
    plt.subplot(311)
    plt.plot(xP, yP, 'g-*')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('EKF Estimated Path')
    plt.subplot(312)
    plt.plot(alpha, 'g-')
    plt.title('EKF Estimated Alpha')
    plt.subplot(313)
    plt.plot(Kr, 'g-')
    plt.title('EKF Estimated Kr')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
