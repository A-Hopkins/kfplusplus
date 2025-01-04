/**
 * @file linear_kf_no_control_example.cpp
 * @brief Example program demonstrating the use of the Kalman Filter for state estimation.
 *
 * This program illustrates the use of the Kalman Filter implemented in the `kfplusplus` library
 * for estimating the position and velocity of a moving object in 2D space. The Kalman Filter
 * operates on a state vector consisting of:
 *   - Position: x, y
 *   - Velocity: vx, vy
 *
 * The program assumes a constant velocity model with no control input.
 * Measurements consist of position data (x, y) with added noise.
 *
 * Steps:
 * 1. Initializes the Kalman Filter with a state transition matrix, measurement matrix,
 *    process noise covariance, and measurement noise covariance.
 * 2. Simulates a series of noisy position measurements.
 * 3. Iteratively performs prediction and update steps of the Kalman Filter.
 * 4. Outputs the estimated state and covariance matrix at each step.
 *
 * Example output demonstrates how the Kalman Filter smooths noisy measurements to
 * provide accurate position and velocity estimates.
 */

#include <iostream>
#include "kfplusplus.h"

int main()
{
  // Define dimensions: state (4), measurement (2)
  const unsigned int state_dim = 4; // x, y, vx, vy
  const unsigned int measurement_dim = 2; // x, y

  // Initialize Kalman Filter
  kfplusplus::KalmanFilter kf(state_dim, measurement_dim);

  // Set transition matrix (assuming constant velocity model)
  linalg::Matrix transition_matrix({ {1.0, 0.0, 1.0, 0.0},
                                     {0.0, 1.0, 0.0, 1.0},
                                     {0.0, 0.0, 1.0, 0.0},
                                     {0.0, 0.0, 0.0, 1.0} });
  kf.set_transition(transition_matrix);

  // Set measurement matrix (we only observe position)
  linalg::Matrix measurement_matrix({ {1.0, 0.0, 0.0, 0.0},
                                      {0.0, 1.0, 0.0, 0.0} });
  kf.set_measurement_matrix(measurement_matrix);

  // Set process noise covariance (tune as needed)
  linalg::Matrix process_noise = linalg::Matrix::identity(state_dim) * 0.01;
  kf.set_process_noise(process_noise);

  // Set measurement noise covariance (tune as needed)
  linalg::Matrix measurement_noise = linalg::Matrix::identity(measurement_dim) * 0.1;
  kf.set_measurement_noise(measurement_noise);

  // Initial state (x=0, y=0, vx=1, vy=1)
  linalg::Vector initial_state({0.0, 0.0, 1.0, 1.0});
  linalg::Matrix initial_covariance = linalg::Matrix::identity(state_dim);

  // Simulated measurements (x, y positions)
  std::vector<linalg::Vector> measurements = { {1.0, 1.0},
                                               {2.1, 2.2},
                                               {3.3, 3.2},
                                               {4.1, 4.0} };

  std::cout << "Kalman Filter Example: Position and Velocity Estimation" << std::endl;

  // Main loop
  for (const auto& measurement : measurements) {
    // Prediction step
    kf.predict();

    // Update step with the new measurement
    kf.update(measurement);

    // Get and print the updated state
    linalg::Vector state = kf.get_state();
    linalg::Matrix cov = kf.get_covariance();
    std::cout << "State: ";
    state.print();
    std::cout << "Cov: ";
    cov.print();
  }

  return 0;
}
