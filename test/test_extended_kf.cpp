#include <gtest/gtest.h>
#include <cmath>

#include "kfplusplus.h"

TEST(ExtendedKalmanFilterTest, TrigonometricMeasurementUpdate) {
  // Initialize EKF with zero state
  kfplusplus::ExtendedKalmanFilter ekf(2, 1);

  // Define the measurement function h(x) = sin(x1)
  auto measurement_function = [](const linalg::Vector& state) { return linalg::Vector({std::sin(state(0))}); };

  // Define the Jacobian of the measurement function: H = [cos(x1), 0]
  auto jacobian_measurement = [](const linalg::Vector& state) { return linalg::Matrix({{std::cos(state(0)), 0.0}}); };

  // Update the EKF with a measurement
  ekf.update(linalg::Vector({0.5}), measurement_function, jacobian_measurement);

  // Get updated state and covariance
  const linalg::Vector& updated_state = ekf.get_state();
  const linalg::Matrix& updated_covariance = ekf.get_covariance();

  // Validate updated state
  EXPECT_NEAR(updated_state(0), 0.25, 1e-5); // Example: state moves toward the measurement
  EXPECT_NEAR(updated_state(1), 0.0, 1e-5);

  // Validate covariance is reduced
  EXPECT_LT(updated_covariance(0, 0), 1.0); // Example: uncertainty decreases
}
