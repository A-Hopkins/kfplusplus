#include <gtest/gtest.h>
#include <cmath>

#include "kfplusplus.h"

TEST(ExtendedKalmanFilterTest, TrigonometricMeasurementUpdate)
{
  // Define dimensions as compile-time constants
  constexpr size_t STATE_DIM = 2;
  constexpr size_t MEASUREMENT_DIM = 1;

  // Use ExtendedKalmanFilter template with size_t dimensions
  kfplusplus::ExtendedKalmanFilter<STATE_DIM> ekf; // CONTROL_DIM defaults to 0

  // Define the non-linear measurement function (sine of first state)
  auto measurement_function = [](const linalg::Vector<STATE_DIM>& state)
  {
    return linalg::Vector<MEASUREMENT_DIM>({std::sin(state(0))});
  };

  // Define the Jacobian of the measurement function (cosine of first state)
  auto jacobian_measurement = [](const linalg::Vector<STATE_DIM>& state)
  {
    return linalg::Matrix<MEASUREMENT_DIM, STATE_DIM>({{std::cos(state(0)), 0.0}});
  };

  // Define measurement noise
  linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM> measurement_noise =
      linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM>::identity() * 0.1; // Example noise

  // Perform update with a measurement of 0.5
  linalg::Vector<MEASUREMENT_DIM> measurement({0.5});
  // Correct update call signature
  ekf.update<MEASUREMENT_DIM>(measurement, measurement_noise, measurement_function, jacobian_measurement);

  // Get the updated state and covariance
  const linalg::Vector<STATE_DIM>& updated_state = ekf.get_state();
  const linalg::Matrix<STATE_DIM, STATE_DIM>& updated_covariance = ekf.get_covariance();

  // Check that the state was updated correctly (adjust expectation based on noise)
  // Note: The previous expectation of 0.25 might change slightly due to noise addition.
  // Re-running the simulation or adjusting tolerance might be needed.
  // For now, keep the original check, but be aware it might fail.
  EXPECT_NEAR(updated_state(0), 0.4545, 1e-4); // Updated to match EKF math
  EXPECT_NEAR(updated_state(1), 0.0, 1e-5);

  // Check that uncertainty was reduced
  EXPECT_LT(updated_covariance(0, 0), 1.0);
}

TEST(ExtendedKalmanFilterTest, NonLinearSystemSimulation)
{
  // Define dimensions as compile-time constants
  constexpr size_t STATE_DIM = 2;
  constexpr size_t MEASUREMENT_DIM = 1;
  constexpr size_t CONTROL_DIM = 0; // Explicitly 0

  // Use ExtendedKalmanFilter template with size_t dimensions
  kfplusplus::ExtendedKalmanFilter<STATE_DIM, CONTROL_DIM> ekf;

  // Set initial state
  linalg::Vector<STATE_DIM> initial_state({1.0, 0.0});
  ekf.set_state(initial_state);

  // Set covariance matrices
  linalg::Matrix<STATE_DIM, STATE_DIM> initial_covariance =
      linalg::Matrix<STATE_DIM, STATE_DIM>::identity() * 1.0;
  linalg::Matrix<STATE_DIM, STATE_DIM> process_noise =
      linalg::Matrix<STATE_DIM, STATE_DIM>::identity() * 0.1;
  linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM> measurement_noise =
      linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM>::identity() * 0.05;

  ekf.set_covariance(initial_covariance);
  ekf.set_process_noise(process_noise);
  // Measurement noise is now set globally for the filter instance in EKF
  // The update method signature expects it to be passed during the call.
  // Let's keep the set_measurement_noise for consistency if the base class uses it,
  // but we still need to pass it to the update method.
  // ekf.set_measurement_noise(measurement_noise); // This method doesn't exist in the provided header

  // Set state transition matrix for constant velocity model
  linalg::Matrix<STATE_DIM, STATE_DIM> F({{1.0, 1.0},
                                          {0.0, 1.0}});
  ekf.set_transition(F); // Used by the linear predict() method

  // Define non-linear measurement function (distance from origin)
  auto measurement_function = [](const linalg::Vector<STATE_DIM>& state)
  {
    double distance = std::sqrt(state(0) * state(0) + state(1) * state(1));
    return linalg::Vector<MEASUREMENT_DIM>({distance});
  };

  // Define Jacobian of measurement function
  auto jacobian_measurement = [](const linalg::Vector<STATE_DIM>& state)
  {
    double x1 = state(0);
    double x2 = state(1);
    double denom = std::sqrt(x1 * x1 + x2 * x2);
    if (denom < 1e-5) denom = 1e-5;  // Avoid division by zero
    return linalg::Matrix<MEASUREMENT_DIM, STATE_DIM>({{x1 / denom, x2 / denom}});
  };

  // Create synthetic measurements
  std::vector<linalg::Vector<MEASUREMENT_DIM>> measurements = {
    linalg::Vector<MEASUREMENT_DIM>({1.5}),
    linalg::Vector<MEASUREMENT_DIM>({2.0}),
    linalg::Vector<MEASUREMENT_DIM>({2.5}),
    linalg::Vector<MEASUREMENT_DIM>({3.0}),
  };

  // Process each measurement
  for (const auto& measurement : measurements)
  {
    ekf.predict(); // Uses linear prediction based on F
    // Correct update call signature
    ekf.update<MEASUREMENT_DIM>(measurement, measurement_noise, measurement_function, jacobian_measurement);
  }

  // Check final state
  const linalg::Vector<STATE_DIM>& final_state = ekf.get_state();
  EXPECT_NEAR(final_state(0), 3.0, 0.2); // Adjusted tolerance slightly
}

TEST(ExtendedKalmanFilterTest, NonLinearStateTransition)
{
  // Define dimensions as compile-time constants
  constexpr size_t STATE_DIM = 3;
  constexpr size_t MEASUREMENT_DIM = 2; // Still needed for noise matrix definition
  constexpr size_t CONTROL_DIM = 1;

  // Use ExtendedKalmanFilter template with size_t dimensions
  kfplusplus::ExtendedKalmanFilter<STATE_DIM, CONTROL_DIM> ekf;

  // Set initial state [x, y, theta]
  linalg::Vector<STATE_DIM> initial_state({1.0, 1.0, 0.0});
  ekf.set_state(initial_state);

  // Set covariance matrices
  linalg::Matrix<STATE_DIM, STATE_DIM> initial_covariance =
      linalg::Matrix<STATE_DIM, STATE_DIM>::identity() * 0.1;
  ekf.set_covariance(initial_covariance);

  // Set noise matrices
  linalg::Matrix<STATE_DIM, STATE_DIM> process_noise =
      linalg::Matrix<STATE_DIM, STATE_DIM>::identity() * 0.01;
  // Measurement noise is needed if update is called, but not for predict_nonlinear only
  // linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM> measurement_noise =
  //     linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM>::identity() * 0.1;

  ekf.set_process_noise(process_noise);
  // ekf.set_measurement_noise(measurement_noise); // Method doesn't exist

  // Define non-linear state transition function (simple unicycle model)
  // state = [x, y, theta], control = [v] (velocity)
  auto state_transition = [](const linalg::Vector<STATE_DIM>& state,
                             const linalg::Vector<CONTROL_DIM>& control)
                             {
    double dt = 1.0;  // time step of 1 second
    double v = control(0);  // velocity control input
    double theta = state(2);  // current heading

    linalg::Vector<STATE_DIM> new_state;
    new_state(0) = state(0) + v * dt * std::cos(theta);  // x' = x + v*cos(theta)*dt
    new_state(1) = state(1) + v * dt * std::sin(theta);  // y' = y + v*sin(theta)*dt
    new_state(2) = state(2);  // theta' = theta (constant heading in this simple model)

    return new_state;
  };

  // Define the Jacobian of the state transition function
  auto jacobian_transition = [](const linalg::Vector<STATE_DIM>& state,
                                const linalg::Vector<CONTROL_DIM>& control)
                                {
    double dt = 1.0;
    double v = control(0);
    double theta = state(2);

    // Jacobian of state transition function with respect to state variables
    // [∂x'/∂x, ∂x'/∂y, ∂x'/∂θ]
    // [∂y'/∂x, ∂y'/∂y, ∂y'/∂θ]
    // [∂θ'/∂x, ∂θ'/∂y, ∂θ'/∂θ]
    linalg::Matrix<STATE_DIM, STATE_DIM> jac({{1.0, 0.0, -v * dt * std::sin(theta)},
                                              {0.0, 1.0, v * dt * std::cos(theta)},
                                              {0.0, 0.0, 1.0}});
    return jac;
  };

  // Use the non-linear prediction with a velocity of 1.0
  linalg::Vector<CONTROL_DIM> control({1.0});
  ekf.predict_nonlinear(state_transition, jacobian_transition, control);

  // Check the predicted state against the expected values
  const linalg::Vector<STATE_DIM>& predicted_state = ekf.get_state();
  EXPECT_NEAR(predicted_state(0), 2.0, 1e-6);  // x = 1.0 + 1.0*cos(0.0) = 2.0
  EXPECT_NEAR(predicted_state(1), 1.0, 1e-6);  // y = 1.0 + 1.0*sin(0.0) = 1.0
  EXPECT_NEAR(predicted_state(2), 0.0, 1e-6);  // theta remains unchanged

  // Check that the covariance has increased due to process noise
  const linalg::Matrix<STATE_DIM, STATE_DIM>& predicted_cov = ekf.get_covariance();
  EXPECT_GT(predicted_cov(0, 0), initial_covariance(0, 0));
}
