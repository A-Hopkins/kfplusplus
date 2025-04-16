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
#include <vector>
#include <cmath>

#include "kfplusplus.h"

int main()
{
  // Define dimensions as compile-time constants
  constexpr size_t STATE_DIM = 4;        // x, y, vx, vy
  constexpr size_t MEASUREMENT_DIM = 2;  // range, bearing
  
  const double dt = 1.0; // Time step in seconds

  // Initialize the EKF with template parameters
  kfplusplus::ExtendedKalmanFilter<STATE_DIM> ekf;

  // Set the initial state: [x, y, vx, vy]
  linalg::Vector<STATE_DIM> initial_state({1000.0, 1000.0, 100.0, 50.0});
  ekf.set_state(initial_state);

  // Set the state transition matrix (constant velocity model)
  linalg::Matrix<STATE_DIM, STATE_DIM> transition_matrix({
    {1.0, 0.0, dt, 0.0},
    {0.0, 1.0, 0.0, dt},
    {0.0, 0.0, 1.0, 0.0},
    {0.0, 0.0, 0.0, 1.0}
  });
  ekf.set_transition(transition_matrix);

  // Set initial covariance and noise
  linalg::Matrix<STATE_DIM, STATE_DIM> initial_covariance = 
      linalg::Matrix<STATE_DIM, STATE_DIM>::identity() * 100.0;  // High initial uncertainty
  
  linalg::Matrix<STATE_DIM, STATE_DIM> process_noise = 
      linalg::Matrix<STATE_DIM, STATE_DIM>::identity() * 1.0;    // Process noise
  
  linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM> measurement_noise = 
      linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM>::identity() * 10.0;  // Measurement noise
  
  ekf.set_covariance(initial_covariance);
  ekf.set_process_noise(process_noise);

  // Define the non-linear measurement function
  auto measurement_function = [](const linalg::Vector<STATE_DIM>& state) {
    double x = state(0);
    double y = state(1);
    double range = std::sqrt(x * x + y * y);
    double bearing = std::atan2(y, x);
    return linalg::Vector<MEASUREMENT_DIM>({range, bearing});
  };

  // Define the Jacobian of the measurement function
  auto jacobian_measurement = [](const linalg::Vector<STATE_DIM>& state) {
    double x = state(0);
    double y = state(1);
    double range = std::sqrt(x * x + y * y);
    if (range < 1e-5) range = 1e-5;  // Avoid division by zero
    
    // Partial derivatives for range and bearing with respect to x, y, vx, vy
    return linalg::Matrix<MEASUREMENT_DIM, STATE_DIM>({
      {x / range, y / range, 0.0, 0.0},
      {-y / (x*x + y*y), x / (x*x + y*y), 0.0, 0.0}
    });
  };

  // Simulated radar measurements (range, bearing)
  std::vector<linalg::Vector<MEASUREMENT_DIM>> measurements = {
    linalg::Vector<MEASUREMENT_DIM>({1414.0, 0.7854}), // sqrt(1000^2 + 1000^2), atan2(1000, 1000)
    linalg::Vector<MEASUREMENT_DIM>({1500.0, 0.8321}),
    linalg::Vector<MEASUREMENT_DIM>({1600.0, 0.8761}),
    linalg::Vector<MEASUREMENT_DIM>({1700.0, 0.9273})
  };

  std::cout << "Extended Kalman Filter Example: Airplane Tracking with Radar" << std::endl;
  std::cout << "------------------------------------------------------------ " << std::endl;

  // Perform prediction and update for each measurement
  for (size_t i = 0; i < measurements.size(); ++i)
  {
    // Predict the next state
    ekf.predict();

    // Update the state with the measurement
    ekf.update<MEASUREMENT_DIM>(measurements[i], measurement_noise, measurement_function, jacobian_measurement);

    // Get the updated state
    const linalg::Vector<STATE_DIM>& state = ekf.get_state();
    const linalg::Matrix<STATE_DIM, STATE_DIM>& cov = ekf.get_covariance();

    // Output the results
    std::cout << "Step " << i + 1 << ":" << std::endl;
    std::cout << "Measurement (range, bearing): (" 
              << measurements[i](0) << ", " << measurements[i](1) << ")" << std::endl;
              
    std::cout << "State: Position (x,y) = (" 
              << state(0) << ", " << state(1) << "), ";
    std::cout << "Velocity (vx,vy) = (" 
              << state(2) << ", " << state(3) << ")" << std::endl;
    
    // Print diagonal elements of covariance (uncertainty)
    std::cout << "Position uncertainty: " 
              << std::sqrt(cov(0, 0)) << ", " << std::sqrt(cov(1, 1)) << std::endl;
    std::cout << "Velocity uncertainty: " 
              << std::sqrt(cov(2, 2)) << ", " << std::sqrt(cov(3, 3)) << std::endl;
              
    std::cout << "-----------------------------------" << std::endl;
  }

  return 0;
}
