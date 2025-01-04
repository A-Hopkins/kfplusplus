/**
 * @file kalman_filter_control_example.cpp
 * @brief Example program demonstrating the use of the Kalman Filter with a control vector.
 *
 * This program shows how the Kalman Filter from the `kfplusplus` library can incorporate
 * a control vector to estimate the position and velocity of a moving object in 2D space.
 * The control vector represents acceleration inputs.
 *
 * The Kalman Filter operates on a state vector consisting of:
 *   - Position: x, y
 *   - Velocity: vx, vy
 *
 * The program assumes a constant acceleration model.
 *
 * Steps:
 * 1. Initializes the Kalman Filter with state transition, control, measurement matrices,
 *    and noise covariances.
 * 2. Simulates a series of noisy position measurements and control inputs (accelerations).
 * 3. Iteratively performs prediction and update steps of the Kalman Filter.
 * 4. Outputs the estimated state and covariance matrix at each step.
 */

#include <iostream>
#include "kfplusplus.h"

int main()
{
  // Define dimensions: state (4), measurement (2), control (2)
  const unsigned int state_dim = 4; // x, y, vx, vy
  const unsigned int measurement_dim = 2; // x, y
  const unsigned int control_dim = 2; // ax, ay

  // Initialize Kalman Filter
  kfplusplus::KalmanFilter kf(state_dim, measurement_dim, control_dim);

  // Set transition matrix (assuming constant acceleration model)
  linalg::Matrix transition_matrix({ {1.0, 0.0, 1.0, 0.0},
                                     {0.0, 1.0, 0.0, 1.0},
                                     {0.0, 0.0, 1.0, 0.0},
                                     {0.0, 0.0, 0.0, 1.0} });
  kf.set_transition(transition_matrix);

  // Set control matrix
  linalg::Matrix control_matrix({ {0.5, 0.0},
                                  {0.0, 0.5},
                                  {1.0, 0.0},
                                  {0.0, 1.0} });
  kf.set_control_matrix(control_matrix);

  // Set measurement matrix (we only observe position)
  linalg::Matrix measurement_matrix({ {1.0, 0.0, 0.0, 0.0},
                                      {0.0, 1.0, 0.0, 0.0} });
  kf.set_measurement_matrix(measurement_matrix);

  // Set process noise covariance
  linalg::Matrix process_noise = linalg::Matrix::identity(state_dim) * 0.01;
  kf.set_process_noise(process_noise);

  // Set measurement noise covariance
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

  // Simulated control inputs (ax, ay accelerations)
  std::vector<linalg::Vector> controls = { {0.2, 0.2},
                                           {0.1, 0.1},
                                           {0.0, 0.0},
                                           {-0.1, -0.1} };

  std::cout << "Kalman Filter Example with Control Vector" << std::endl;

  // Main loop
  for (size_t i = 0; i < measurements.size(); ++i) {
    // Prediction step with control input
    kf.predict(controls[i]);

    // Update step with the new measurement
    kf.update(measurements[i]);

    // Get and print the updated state
    linalg::Vector state = kf.get_state();
    linalg::Matrix cov = kf.get_covariance();
    std::cout << "Step " << i + 1 << ":" << std::endl;
    std::cout << "State: ";
    state.print();
    std::cout << "Cov: ";
    cov.print();
    std::cout << std::endl;
  }

  return 0;
}
