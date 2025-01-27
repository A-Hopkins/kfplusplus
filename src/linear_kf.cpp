#include "kfplusplus.h"

namespace kfplusplus
{
  void KalmanFilter::predict(const linalg::Vector &control)
  {
    // Predict state: x = F * x + B * u
    if (control.size() == state_dim)
    {
      state = transition_matrix * state + control_matrix * control;
    }
    else
    {
      state = transition_matrix * state;
    }

    // Predict covariance: P = F * P * F^T + Q
    covariance = transition_matrix * covariance * transition_matrix.transpose() + process_noise;
  }

  void KalmanFilter::update(const linalg::Vector &measurement)
  {
    // Innovation y = z - Hx
    linalg::Vector y = measurement - measurement_matrix * state;
    linalg::Matrix ht = measurement_matrix.transpose();

    // Predictied covariance mapped to measurement space: P * H^T
    linalg::Matrix predicted_covariance = covariance * ht;

    // Innovation covariance S = H * P * H^T + R
    linalg::Matrix innovation_covariance = measurement_matrix * predicted_covariance + measurement_noise;

    // Kalman gain: K = P * H^T * (H * P * H^T + R)^-1
    linalg::Matrix gain = predicted_covariance * innovation_covariance.invert();

    // Update state: x = x + K * y
    state = state + (gain * y);

    // Update covariance: P = (I - K * H) * P
    linalg::Matrix I = linalg::Matrix::identity(state_dim);
    covariance = (I - gain * measurement_matrix) * covariance;
  }
}
