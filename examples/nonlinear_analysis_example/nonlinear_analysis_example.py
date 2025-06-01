from kfplusplus import (
    ExtendedKalmanFilter5x0,
    Vector2, Vector5,
    Matrix2x2, Matrix5x5, Matrix2x5
)
import math
import random

def generate_imu_data(steps, dt):
    ground_truth = []
    imu_measurements = []
    x, y, theta = 0.0, 0.0, 0.0
    v, omega = 1.0, 0.1

    for _ in range(steps):
        x += v * dt * math.cos(theta)
        y += v * dt * math.sin(theta)
        theta += omega * dt
        ground_truth.append([x, y, theta])

        noisy_v = v + random.gauss(0.0, 0.1)
        noisy_omega = omega + random.gauss(0.0, 0.1)
        meas = Vector2()
        meas[0] = noisy_v
        meas[1] = noisy_omega
        imu_measurements.append(meas)
    return ground_truth, imu_measurements

# Parameters
STATE_DIM = 5
MEASUREMENT_DIM = 2
CONTROL_DIM = 0
dt = 1.0
steps = 100

ekf = ExtendedKalmanFilter5x0()
initial_state = Vector5()
initial_state[0] = 0.0  # x position
initial_state[1] = 0.0  # y position
initial_state[2] = 0.0  # orientation (theta)
initial_state[3] = 1.0  # velocity (v)
initial_state[4] = 0.1  # angular velocity (omega)
ekf.set_state(initial_state)
ekf.set_covariance(Matrix5x5.identity() * 0.1)
ekf.set_process_noise(Matrix5x5.identity() * 0.01)

# Nonlinear state transition (bicycle model)
def state_transition(state, control):
    x, y, theta, v, omega = [state[i] for i in range(5)]
    new_state = Vector5()
    new_state[0] = x + v * dt * math.cos(theta)
    new_state[1] = y + v * dt * math.sin(theta)
    new_state[2] = theta + omega * dt
    new_state[3] = v
    new_state[4] = omega

    return new_state

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

def measurement_function(state):
    meas = Vector2()
    meas[0] = state[3]  # velocity
    meas[1] = state[4]  # angular velocity
    return meas

def jacobian_measurement(state):
    H = Matrix2x5()
    H[0, 3] = 1.0  # d(measurement)/d(velocity)
    H[1, 4] = 1.0  # d(measurement)/d(angular velocity)
    return H

# Generate data
ground_truth, imu_measurements = generate_imu_data(steps, dt)

for i in range(steps):
    ekf.predict(state_transition, jacobian_transition)
    ekf.update_imu(
        imu_measurements[i],
        Matrix2x2.identity() * 0.1,
        measurement_function,
        jacobian_measurement
    )
    state = ekf.get_state()
    print(f"Step {i+1}:")
    print(f"Ground Truth: x = {ground_truth[i][0]}, y = {ground_truth[i][1]}, theta = {ground_truth[i][2]}")
    print(f"Estimated: x = {state[0]}, y = {state[1]}, theta = {state[2]}")
    print("-----------------------------------")

