/**
 * @file nonlinear_kf_example.cpp
 * @brief Example program demonstrating the use of the Extended Kalman Filter (EKF) for tracking an airplane.
 *
 * This program illustrates the use of the Extended Kalman Filter implemented in the `kfplusplus` library
 * for estimating the position and velocity of an airplane in 2D space based on radar measurements.
 * The EKF operates on a state vector consisting of:
 *   - Position: x, y
 *   - Velocity: vx, vy
 *
 * The program uses a constant velocity motion model and simulates radar measurements consisting of:
 *   - Range: Distance from the radar to the airplane.
 *   - Bearing: Angle of the airplane relative to the radar.
 *
 * Steps:
 * 1. Initializes the EKF with a state transition matrix, process noise covariance, and measurement noise covariance.
 * 2. Defines a non-linear measurement function and its Jacobian.
 * 3. Simulates a series of noisy radar measurements (range and bearing).
 * 4. Iteratively performs prediction and update steps of the EKF.
 * 5. Outputs the estimated state and covariance matrix at each step.
 *
 * Example output demonstrates how the EKF handles non-linear measurements to accurately estimate
 * the airplane's position and velocity.
 */

#include <iostream>
#include <cmath>
#include "kfplusplus.h"

int main()
{
  const double dt = 1.0; // Time step in seconds

  // Initialize the EKF with 4D state and 2D measurement
  kfplusplus::ExtendedKalmanFilter ekf(4, 2);

  // Set the initial state: [x, y, vx, vy]
  ekf.set_state(linalg::Vector({1000.0, 1000.0, 100.0, 50.0}));

  // Set the state transition matrix (constant velocity model)
  ekf.set_transition(linalg::Matrix({
    {1.0, 0.0, dt, 0.0},
    {0.0, 1.0, 0.0, dt},
    {0.0, 0.0, 1.0, 0.0},
    {0.0, 0.0, 0.0, 1.0},
  }));

  // Set initial covariance and noise
  ekf.set_covariance(linalg::Matrix::identity(4) * 100.0);  // High initial uncertainty
  ekf.set_process_noise(linalg::Matrix::identity(4) * 1.0);  // Process noise
  ekf.set_measurement_noise(linalg::Matrix::identity(2) * 10.0);  // Measurement noise

  // Define the non-linear measurement function
  auto measurement_function = [](const linalg::Vector& state) {
    double x = state(0);
    double y = state(1);
    double range = std::sqrt(x * x + y * y);
    double bearing = std::atan2(y, x);
    return linalg::Vector({range, bearing});
  };

  // Define the Jacobian of the measurement function
  auto jacobian_measurement = [](const linalg::Vector& state) {
    double x = state(0);
    double y = state(1);
    double range = std::sqrt(x * x + y * y);
    if (range < 1e-5) range = 1e-5;  // Avoid division by zero
    return linalg::Matrix({
      {x / range, y / range, 0.0, 0.0},
      {-y / (range * range), x / (range * range), 0.0, 0.0}
    });
  };

  // Simulated radar measurements (range, bearing)
  std::vector<linalg::Vector> measurements = {
    linalg::Vector({1414.0, 0.7854}), // sqrt(1000^2 + 1000^2), atan2(1000, 1000)
    linalg::Vector({1500.0, 0.8321}),
    linalg::Vector({1600.0, 0.8761}),
    linalg::Vector({1700.0, 0.9273}),
  };

  // Perform prediction and update for each measurement
  for (const auto& measurement : measurements) {
    // Predict the next state
    ekf.predict();

    // Update the state with the measurement
    ekf.update(measurement, measurement_function, jacobian_measurement);

    // Get the updated state
    const linalg::Vector& updated_state = ekf.get_state();
    const linalg::Matrix& updated_covariance = ekf.get_covariance();

    // Output the results
    std::cout << "Updated State: ";
    updated_state.print();
    std::cout << "Updated Covariance: ";
    updated_covariance.print();
    std::cout << "-----------------------------------" << std::endl;
  }

  return 0;
}
