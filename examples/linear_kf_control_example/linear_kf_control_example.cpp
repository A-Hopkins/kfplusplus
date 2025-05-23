/**
 * @file kalman_filter_control_example.cpp
 * @brief Example program demonstrating the use of the Kalman Filter with a
 * control vector.
 *
 * This program shows how the Kalman Filter from the `kfplusplus` library can
 * incorporate a control vector to estimate the position and velocity of a
 * moving object in 2D space. The control vector represents acceleration inputs.
 *
 * The Kalman Filter operates on a state vector consisting of:
 *   - Position: x, y
 *   - Velocity: vx, vy
 *
 * The program assumes a constant acceleration model.
 *
 * Steps:
 * 1. Initializes the Kalman Filter with state transition, control, measurement
 * matrices, and noise covariances.
 * 2. Simulates a series of noisy position measurements and control inputs
 * (accelerations).
 * 3. Iteratively performs prediction and update steps of the Kalman Filter.
 * 4. Outputs the estimated state and covariance matrix at each step.
 */

#include "kfplusplus.h"
#include <iostream>
#include <vector>

int main()
{
  // Define dimensions as compile-time constants
  constexpr size_t STATE_DIM       = 4; // x, y, vx, vy
  constexpr size_t MEASUREMENT_DIM = 2; // x, y
  constexpr size_t CONTROL_DIM     = 2; // ax, ay

  // Initialize Kalman Filter with template parameters
  kfplusplus::KalmanFilter<STATE_DIM, CONTROL_DIM> kf;

  // Set transition matrix (assuming constant acceleration model)
  linalg::Matrix<STATE_DIM, STATE_DIM> transition_matrix(
      {{1.0, 0.0, 1.0, 0.0}, {0.0, 1.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0}});
  kf.set_transition(transition_matrix);

  // Set control matrix
  linalg::Matrix<STATE_DIM, CONTROL_DIM> control_matrix(
      {{0.5, 0.0}, {0.0, 0.5}, {1.0, 0.0}, {0.0, 1.0}});
  kf.set_control_matrix(control_matrix);

  // Set measurement matrix (we only observe position)
  linalg::Matrix<MEASUREMENT_DIM, STATE_DIM> measurement_matrix(
      {{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}});

  // Set process noise covariance
  linalg::Matrix<STATE_DIM, STATE_DIM> process_noise =
      linalg::Matrix<STATE_DIM, STATE_DIM>::identity() * 0.01;
  kf.set_process_noise(process_noise);

  // Set measurement noise covariance
  linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM> measurement_noise =
      linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM>::identity() * 0.1;

  // Initial state (x=0, y=0, vx=1, vy=1)
  linalg::Vector<STATE_DIM>            initial_state({0.0, 0.0, 1.0, 1.0});
  linalg::Matrix<STATE_DIM, STATE_DIM> initial_covariance =
      linalg::Matrix<STATE_DIM, STATE_DIM>::identity();
  kf.set_state(initial_state);
  kf.set_covariance(initial_covariance);

  // Simulated measurements (x, y positions)
  std::vector<linalg::Vector<MEASUREMENT_DIM>> measurements = {
      linalg::Vector<MEASUREMENT_DIM>({1.0, 1.0}), linalg::Vector<MEASUREMENT_DIM>({2.1, 2.2}),
      linalg::Vector<MEASUREMENT_DIM>({3.3, 3.2}), linalg::Vector<MEASUREMENT_DIM>({4.1, 4.0})};

  // Simulated control inputs (ax, ay accelerations)
  std::vector<linalg::Vector<CONTROL_DIM>> controls = {
      linalg::Vector<CONTROL_DIM>({0.2, 0.2}), linalg::Vector<CONTROL_DIM>({0.1, 0.1}),
      linalg::Vector<CONTROL_DIM>({0.0, 0.0}), linalg::Vector<CONTROL_DIM>({-0.1, -0.1})};

  std::cout << "Kalman Filter Example with Control Vector" << std::endl;

  // Main loop
  for (size_t i = 0; i < measurements.size(); ++i)
  {
    // Prediction step with control input
    kf.predict(controls[i]);

    // Update step with the new measurement
    // Pass measurement_matrix and measurement_noise as arguments
    kf.update<MEASUREMENT_DIM>(measurements[i], measurement_matrix, measurement_noise);

    // Get and print the updated state
    const linalg::Vector<STATE_DIM>&            state = kf.get_state();
    const linalg::Matrix<STATE_DIM, STATE_DIM>& cov   = kf.get_covariance();

    std::cout << "Step " << i + 1 << ":" << std::endl;
    std::cout << "State: Position (x,y) = (" << state(0) << ", " << state(1) << "), ";
    std::cout << "Velocity (vx,vy) = (" << state(2) << ", " << state(3) << ")" << std::endl;

    // Print diagonal elements of covariance (uncertainty)
    std::cout << "Uncertainty: ";
    for (size_t j = 0; j < STATE_DIM; ++j)
    {
      std::cout << std::sqrt(cov(j, j)) << " ";
    }
    std::cout << std::endl << std::endl;
  }

  return 0;
}
