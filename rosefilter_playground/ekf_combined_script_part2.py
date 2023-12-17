
def parse_arguments():
    parser = argparse.ArgumentParser(description='Extended Kalman Filter for Robot Localization.')
    parser.add_argument('mode', choices=['xy', 'rose'], help='EKF mode: "xy" or "rose"')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file with measurements.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV file for results.')
    parser.add_argument('--output_plot', type=str, required=True, help='Path to output plot file.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    data = read_csv(args.input_csv)
    Ts = 1.0  # Time step
    Cj = np.eye(5)  # Measurement matrix
    P_dach = np.eye(5) * 0.1  # Initial state covariance
    R = np.eye(5) * 10  # Measurement noise
    GQG = np.eye(5) * 0.1  # Process noise
    x_dach = np.zeros(5)  # Initial state estimate

    ekf = EKF(Ts, Cj, P_dach, R, GQG, x_dach, ekf_type=args.mode)
    xP, yP, alpha, Kr, v = ekf.update(data)

    results = np.column_stack((xP, yP, alpha, Kr, v))
    write_csv(results, args.output_csv)
    plot_ekf_results(xP, yP, alpha, Kr, v, args.output_plot, ekf_type=args.mode)

if __name__ == '__main__':
    main()
