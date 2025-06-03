from kfplusplus import (
    ExtendedKalmanFilter5x0,
    Vector2, Vector5,
    Matrix2x2, Matrix5x5, Matrix2x5
)
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Parameters
random.seed(42)
STATE_DIM = 5
MEASUREMENT_DIM = 2
CONTROL_DIM = 0
dt = 1
steps = 100
ENABLE_OPTIMIZATION = False  # Toggle optimization for Q

# Measurement noise (R) - typically known from sensor specs
R_VALUE = 0.1

# Generate IMU data
def generate_imu_data(steps, dt):
    ground_truth = []
    imu_measurements = []
    x, y = 0.0, 0.0      # Start at origin
    base_theta = 0.0     # Nominal heading straight (along x)
    v = 1.0
    # Use a slow sinusoidal modulation on heading to simulate listing
    amplitude = 0.21      # maximum deviation in radians
    frequency = 0.15      # slower oscillation frequency

    for i in range(steps):
        t = i * dt
        # Oscillate heading: listing left then right.
        theta = base_theta + amplitude * math.sin(frequency * t)
        # Update position using the current heading
        x += v * dt * math.cos(theta)
        y += v * dt * math.sin(theta)
        ground_truth.append([x, y, theta])
        # Compute the true angular rate (derivative of theta)
        true_omega = amplitude * frequency * math.cos(frequency * t)
        # Add noise to the measurements
        noisy_v = v + random.gauss(0.0, 0.1)
        noisy_omega = true_omega + random.gauss(0.0, 0.1)
        meas = Vector2()
        meas[0] = noisy_v
        meas[1] = noisy_omega
        imu_measurements.append(meas)
        
    return ground_truth, imu_measurements

# State transition model
def state_transition(state, control):
    x, y, theta, v, omega = [state[i] for i in range(5)]
    new_state = Vector5()
    new_state[0] = x + v * dt * math.cos(theta)
    new_state[1] = y + v * dt * math.sin(theta)
    new_state[2] = theta + omega * dt
    new_state[3] = v
    new_state[4] = omega
    return new_state

# Jacobian of the state transition model
def jacobian_transition(state, control):
    theta = state[2]
    v = state[3]
    jacobian = Matrix5x5.identity()
    jacobian[0, 2] = -v * dt * math.sin(theta)
    jacobian[0, 3] = dt * math.cos(theta)
    jacobian[1, 2] = v * dt * math.cos(theta)
    jacobian[1, 3] = dt * math.sin(theta)
    jacobian[2, 4] = dt
    return jacobian

# Measurement function
def measurement_function(state):
    meas = Vector2()
    meas[0] = state[3]  # velocity
    meas[1] = state[4]  # angular velocity
    return meas

# Jacobian of the measurement function
def jacobian_measurement(state):
    H = Matrix2x5()
    H[0, 3] = 1.0  # d(measurement)/d(velocity)
    H[1, 4] = 1.0  # d(measurement)/d(angular velocity)
    return H

# Run EKF with given Q and R
def run_ekf(q_value):
    ekf = ExtendedKalmanFilter5x0()
    initial_state = Vector5()
    initial_state[0] = 0.0
    initial_state[1] = 0.0
    initial_state[2] = 0.0
    initial_state[3] = 1.0
    initial_state[4] = 0.0
    ekf.set_state(initial_state)
    ekf.set_covariance(Matrix5x5.identity() * 1.0)
    ekf.set_process_noise(Matrix5x5.identity() * q_value)

    ekf_x, ekf_y, innovations, covariances = [], [], [], []
    for i in range(steps):
        ekf.predict(state_transition, jacobian_transition)
        ekf.update_imu(
            imu_measurements[i],
            Matrix2x2.identity() * R_VALUE,
            measurement_function,
            jacobian_measurement
        )
        state = ekf.get_state()
        ekf_x.append(state[0])
        ekf_y.append(state[1])

        # Compute innovation (measurement residual)
        pred_meas = measurement_function(state)
        innovation = np.sqrt((imu_measurements[i][0] - pred_meas[0])**2 +
                             (imu_measurements[i][1] - pred_meas[1])**2)
        innovations.append(innovation)

        # Store covariance (diagonal elements)
        covariances.append([ekf.get_covariance()[i, i] for i in range(STATE_DIM)])

    return np.array(ekf_x), np.array(ekf_y), np.array(innovations), np.array(covariances)

# Compute MSE
def compute_mse(ekf_x, ekf_y, truth_x, truth_y):
    return np.mean((ekf_x - truth_x) ** 2 + (ekf_y - truth_y) ** 2)

# Optimization for Q
def optimize_q():
    best_q = None
    best_mse = float('inf')
    for q in np.linspace(1e-5, 0.01, 100):
        ekf_x, ekf_y, _, _ = run_ekf(q)
        mse = compute_mse(ekf_x, ekf_y, truth_x, truth_y)
        if mse < best_mse:
            best_mse = mse
            best_q = q
    return best_q

# Optimization for R
def optimize_r():
    best_r = None
    best_mse = float('inf')
    for r in np.linspace(0.01, 1.0, 100):  # Sweep R values
        global R_VALUE  # Update the global R_VALUE for each run
        R_VALUE = r
        ekf_x, ekf_y, _, _ = run_ekf(Q_VALUE)  # Use the best Q_VALUE
        mse = compute_mse(ekf_x, ekf_y, truth_x, truth_y)
        if mse < best_mse:
            best_mse = mse
            best_r = r
    return best_r

# Optimization for Initial Covariance
def optimize_initial_covariance():
    best_cov = None
    best_mse = float('inf')
    for cov in np.linspace(0.01, 10.0, 100):  # Sweep initial covariance values
        ekf = ExtendedKalmanFilter5x0()
        initial_state = Vector5()
        initial_state[0] = 0.0
        initial_state[1] = 0.0
        initial_state[2] = 0.0
        initial_state[3] = 1.0
        initial_state[4] = 0.1
        ekf.set_state(initial_state)
        ekf.set_covariance(Matrix5x5.identity() * cov)  # Set initial covariance
        ekf.set_process_noise(Matrix5x5.identity() * Q_VALUE)

        ekf_x, ekf_y, _, _ = run_ekf(Q_VALUE)
        mse = compute_mse(ekf_x, ekf_y, truth_x, truth_y)
        if mse < best_mse:
            best_mse = mse
            best_cov = cov
    return best_cov

# Generate data
ground_truth, imu_measurements = generate_imu_data(steps, dt)
truth_x = np.array([gt[0] for gt in ground_truth])
truth_y = np.array([gt[1] for gt in ground_truth])

# Optimize Q, R, and Initial Covariance if enabled
if ENABLE_OPTIMIZATION:
    Q_VALUE = optimize_q()
    R_VALUE = optimize_r()
    INITIAL_COVARIANCE = optimize_initial_covariance()
else:
    Q_VALUE = 1e-5  # Default Q value
    R_VALUE = 0.1   # Default R value
    INITIAL_COVARIANCE = 1.0  # Default initial covariance

# Run EKF with the best Q, R, and Initial Covariance
def run_ekf_with_cov(q_value, r_value, initial_cov):
    ekf = ExtendedKalmanFilter5x0()
    initial_state = Vector5()
    initial_state[0] = 0.0
    initial_state[1] = 0.0
    initial_state[2] = 0.0
    initial_state[3] = 1.0
    initial_state[4] = 0.1
    ekf.set_state(initial_state)
    ekf.set_covariance(Matrix5x5.identity() * initial_cov)
    ekf.set_process_noise(Matrix5x5.identity() * q_value)

    ekf_x, ekf_y, innovations, covariances = [], [], [], []
    for i in range(steps):
        ekf.predict(state_transition, jacobian_transition)
        ekf.update_imu(
            imu_measurements[i],
            Matrix2x2.identity() * r_value,
            measurement_function,
            jacobian_measurement
        )
        state = ekf.get_state()
        ekf_x.append(state[0])
        ekf_y.append(state[1])

        # Compute innovation (measurement residual)
        pred_meas = measurement_function(state)
        innovation = np.sqrt((imu_measurements[i][0] - pred_meas[0])**2 +
                             (imu_measurements[i][1] - pred_meas[1])**2)
        innovations.append(innovation)

        # Store covariance (diagonal elements)
        covariances.append([ekf.get_covariance()[i, i] for i in range(STATE_DIM)])

    return np.array(ekf_x), np.array(ekf_y), np.array(innovations), np.array(covariances)

ekf_x, ekf_y, innovations, covariances = run_ekf_with_cov(Q_VALUE, R_VALUE, INITIAL_COVARIANCE)

# Plot results
plt.figure()
plt.plot(truth_x, truth_y, label="Ground Truth", linewidth=2, color='blue')
plt.plot(ekf_x, ekf_y, label=f"EKF (Q={Q_VALUE:.4e}, R={R_VALUE:.2f}, Cov={INITIAL_COVARIANCE:.2f})", linestyle='--', color='orange')

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("EKF Trajectory vs Ground Truth")
plt.show()

# Plot innovations
plt.figure()
plt.plot(innovations, label="Innovations")
plt.xlabel("Step")
plt.ylabel("Innovation")
plt.title("Measurement Residuals (Innovations)")
plt.legend()
plt.show()

# Plot covariance
plt.figure()
for i in range(STATE_DIM):
    plt.plot(covariances[:, i], label=f"Covariance of State {i}")
plt.xlabel("Step")
plt.ylabel("Covariance")
plt.title("EKF State Covariances")
plt.legend()
plt.show()

def plot_covariance_ellipse(ax, mean, cov, n_std=1.0, facecolor='none',  point_color='black', **kwargs):
    """Plot a covariance ellipse centered at mean with covariance matrix cov."""
    # Eigen decomposition of the 2x2 covariance submatrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:,order]

    # angle of ellipse rotation in degrees
    angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
    
    # Width and height of ellipse = 2*n_std*sqrt(eigenvalue)
    width, height = 2 * n_std * np.sqrt(eigvals)
    
    ellipse = patches.Ellipse(xy=mean, width=width, height=height,
                              angle=angle, facecolor=facecolor, **kwargs)
    ax.add_patch(ellipse)
    ax.scatter(mean[0], mean[1], color=point_color, s=15, zorder=10)

# After running the EKF, plotting trajectory and ellipses:
fig, ax = plt.subplots()

# Plot ground truth and estimated trajectory
ax.plot(truth_x, truth_y, label="Ground Truth", linewidth=2, color='blue')
ax.plot(ekf_x, ekf_y, label=f"EKF (Q={Q_VALUE:.4e}, R={R_VALUE:.2f})", linestyle='--', color='orange')

# Plot covariance ellipses every 10 steps
for idx in range(0, steps, 10):
    # Get the 2x2 submatrix for positions (assumes state order [x, y, theta, v, omega])
    cov_matrix = np.array([[covariances[idx][0], covariances[idx][1]*0.0],   # Off-diagonals not computed here
                            [covariances[idx][1]*0.0, covariances[idx][1]]])
    # Note: If you stored the full 5x5 covariance matrices properly,
    # you should extract the (0,0), (0,1), (1,0), (1,1) components.
    # Here we use the diagonal approximations for demonstration.
    mean = (ekf_x[idx], ekf_y[idx])
    plot_covariance_ellipse(ax, mean, cov_matrix, n_std=1, edgecolor='red', alpha=0.5)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.set_title("EKF Trajectory vs Ground Truth with Covariance Ellipses")
plt.show()
