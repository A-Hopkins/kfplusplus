#include "kfplusplus.h"

#include <iostream>

namespace kfplusplus
{
  void ExtendedKalmanFilter::update(const linalg::Vector& measurement,
                                    const std::function<linalg::Vector(const linalg::Vector&)>& measurement_function,
                                    const std::function<linalg::Matrix(const linalg::Vector&)>& jacobian_measurement)
  {
    // Compute the predicted measurement: h(x)
    linalg::Vector h_x = measurement_function(state);

    // Compute the innovation (residual): y = z - h(x)
    linalg::Vector y = measurement - h_x;

    // Compute the Jacobian of the measurement function: H
    linalg::Matrix H = jacobian_measurement(state);

    // Predictied covariance mapped to measurement space: P * H^T
    linalg::Matrix predicted_covariance = covariance * H.transpose();

    // Innovation covariance S = H * P * H^T + R
    linalg::Matrix innovation_covariance = H * predicted_covariance + measurement_noise;

    // Kalman gain: K = P * H^T * (H * P * H^T + R)^-1
    linalg::Matrix gain = predicted_covariance * innovation_covariance.invert();

    // Update the state estimate: x = x + K * y
    state = state + gain * y;

    // Update the covariance: P = (I - K * H) * P
    linalg::Matrix I = linalg::Matrix::identity(state_dim);
    covariance = (I - gain * H) * covariance;
  }
}
