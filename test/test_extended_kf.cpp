#include <gtest/gtest.h>
#include <cmath>

#include "kfplusplus.h"

TEST(ExtendedKalmanFilterTest, TrigonometricMeasurementUpdate) {
  kfplusplus::ExtendedKalmanFilter ekf(2, 1);

  auto measurement_function = [](const linalg::Vector& state) { return linalg::Vector({std::sin(state(0))}); };

  auto jacobian_measurement = [](const linalg::Vector& state) { return linalg::Matrix({{std::cos(state(0)), 0.0}}); };

  ekf.update(linalg::Vector({0.5}), measurement_function, jacobian_measurement);

  const linalg::Vector& updated_state = ekf.get_state();
  const linalg::Matrix& updated_covariance = ekf.get_covariance();

  EXPECT_NEAR(updated_state(0), 0.25, 1e-5);
  EXPECT_NEAR(updated_state(1), 0.0, 1e-5);

  EXPECT_LT(updated_covariance(0, 0), 1.0);
}

TEST(ExtendedKalmanFilterTest, NonLinearSystemSimulation) {
  kfplusplus::ExtendedKalmanFilter ekf(2, 1);

  ekf.set_state(linalg::Vector({1.0, 0.0}));

  ekf.set_covariance(linalg::Matrix::identity(2) * 1.0);
  ekf.set_process_noise(linalg::Matrix::identity(2) * 0.1);
  ekf.set_measurement_noise(linalg::Matrix::identity(1) * 0.05);

  ekf.set_transition(linalg::Matrix({{1.0, 1.0},
                                      {0.0, 1.0}}));

  auto measurement_function = [](const linalg::Vector& state) {
    double distance = std::sqrt(state(0) * state(0) + state(1) * state(1));
    return linalg::Vector({distance});
  };

  auto jacobian_measurement = [](const linalg::Vector& state) {
    double x1 = state(0);
    double x2 = state(1);
    double denom = std::sqrt(x1 * x1 + x2 * x2);
    if (denom < 1e-5) denom = 1e-5;
    return linalg::Matrix({{x1 / denom, x2 / denom}});
  };

  std::vector<linalg::Vector> measurements = {
    linalg::Vector({1.5}),
    linalg::Vector({2.0}),
    linalg::Vector({2.5}),
    linalg::Vector({3.0}),
  };

  for (const auto& measurement : measurements)
  {
    ekf.predict();
    ekf.update(measurement, measurement_function, jacobian_measurement);
  }

  const linalg::Vector& final_state = ekf.get_state();
  EXPECT_NEAR(final_state(0), 3.0, 0.1);
}
