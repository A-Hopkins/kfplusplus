/**
 * @file nonlinear_analysis_example.cpp
 * @brief Example program demonstrating the use of the Extended Kalman Filter
 * (EKF) for analyzing non-linear systems.
 */
#include "kfplusplus.h"
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

void generate_imu_data(std::vector<linalg::Vector<3>>& ground_truth,
                       std::vector<linalg::Vector<2>>& imu_measurements, size_t steps, double dt)
{
  std::default_random_engine       generator;
  std::normal_distribution<double> noise_dist(0.0, 0.1); // IMU noise

  double x = 0.0, y = 0.0, theta = 0.0; // Initial state
  double v = 1.0, omega = 0.1;          // Constant velocity and angular velocity

  for (size_t i = 0; i < steps; ++i)
  {
    // Ground truth state update
    x += v * dt * std::cos(theta);
    y += v * dt * std::sin(theta);
    theta += omega * dt;

    // Store ground truth
    ground_truth.push_back(linalg::Vector<3>({x, y, theta}));

    // Generate noisy IMU measurements (linear velocity and angular velocity)
    double noisy_v     = v + noise_dist(generator);
    double noisy_omega = omega + noise_dist(generator);
    imu_measurements.push_back(linalg::Vector<2>({noisy_v, noisy_omega}));
  }
}

int main()
{
  // Define dimensions
  constexpr size_t STATE_DIM       = 5; // [x, y, theta, v, omega]
  constexpr size_t MEASUREMENT_DIM = 2; // [v, omega]
  constexpr size_t CONTROL_DIM     = 0; // No control input

  const double dt    = 1.0; // 1 hz IMU update rate
  const size_t steps = 100; // Number of time steps

  kfplusplus::ExtendedKalmanFilter<STATE_DIM, CONTROL_DIM> ekf;

  linalg::Vector<STATE_DIM> initial_state({0.0, 0.0, 0.0, 1.0, 0.1});
  ekf.set_state(initial_state);

  // Set initial covariance and process noise
  linalg::Matrix<STATE_DIM, STATE_DIM> initial_covariance =
      linalg::Matrix<STATE_DIM, STATE_DIM>::identity() * 0.1;
  linalg::Matrix<STATE_DIM, STATE_DIM> process_noise =
      linalg::Matrix<STATE_DIM, STATE_DIM>::identity() * 0.01;

  ekf.set_covariance(initial_covariance);
  ekf.set_process_noise(process_noise);

  // Define the non-linear state transition function (bicycle model)
  auto state_transition =
      [dt](const linalg::Vector<STATE_DIM>& state, const linalg::Vector<CONTROL_DIM>& /*control*/)
  {
    double x     = state(0);
    double y     = state(1);
    double theta = state(2);
    double v     = state(3);
    double omega = state(4);

    linalg::Vector<STATE_DIM> new_state;
    new_state(0) = x + v * dt * std::cos(theta);
    new_state(1) = y + v * dt * std::sin(theta);
    new_state(2) = theta + omega * dt;
    new_state(3) = v;
    new_state(4) = omega;
    return new_state;
  };

  // Define the Jacobian of the state transition function
  auto jacobian_transition =
      [dt](const linalg::Vector<STATE_DIM>& state, const linalg::Vector<CONTROL_DIM>& /*control*/)
  {
    double theta = state(2);
    double v     = state(3);

    return linalg::Matrix<STATE_DIM, STATE_DIM>(
        {{1.0, 0.0, -v * dt * std::sin(theta), dt * std::cos(theta), 0.0},
         {0.0, 1.0, v * dt * std::cos(theta), dt * std::sin(theta), 0.0},
         {0.0, 0.0, 1.0, 0.0, dt},
         {0.0, 0.0, 0.0, 1.0, 0.0},
         {0.0, 0.0, 0.0, 0.0, 1.0}});
  };

  // Define the measurement function (IMU provides noisy v and omega)
  auto measurement_function = [](const linalg::Vector<STATE_DIM>& state) {
    return linalg::Vector<MEASUREMENT_DIM>({state(3), state(4)});
  };

  // Define the Jacobian of the measurement function
  auto jacobian_measurement = [](const linalg::Vector<STATE_DIM>& /*state*/)
  {
    return linalg::Matrix<MEASUREMENT_DIM, STATE_DIM>(
        {{0.0, 0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 1.0}});
  };

  // Generate ground truth and noisy IMU measurements
  std::vector<linalg::Vector<3>> ground_truth;
  std::vector<linalg::Vector<2>> imu_measurements;
  generate_imu_data(ground_truth, imu_measurements, steps, dt);

  // Process each measurement
  for (size_t i = 0; i < steps; ++i)
  {
    // Predict step
    ekf.predict(state_transition, jacobian_transition);

    // Update step
    ekf.update<MEASUREMENT_DIM>(imu_measurements[i],
                                linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM>::identity() * 0.1,
                                measurement_function, jacobian_measurement);

    // Get the updated state
    const linalg::Vector<STATE_DIM>& state = ekf.get_state();

    // Output the results
    std::cout << "Step " << i + 1 << ":" << std::endl;
    std::cout << "Ground Truth: x = " << ground_truth[i](0) << ", y = " << ground_truth[i](1)
              << ", theta = " << ground_truth[i](2) << std::endl;
    std::cout << "Estimated: x = " << state(0) << ", y = " << state(1) << ", theta = " << state(2)
              << std::endl;
    std::cout << "-----------------------------------" << std::endl;
  }

  return 0;
}
