/**
 * @file kfplusplus.h
 * @brief Kalman Filter and Extended Kalman Filter c++ Implementation
 *
 * This file defines the KalmanFilter class, which models a linear Kalman filter
 * for state estimation. It also defines the ExtendedKalmanFilter class, which
 * extends the functionality to handle non-linear state transitions and
 * measurements by incorporating Jacobians and non-linear functions.
 */

#pragma once

#include "linalg.h"
#include <cassert>
#include <functional>

namespace kfplusplus
{
  /**
   * @class KalmanFilter
   * @brief Implements a linear Kalman filter for state estimation with
   * compile-time dimensions.
   * @tparam STATE_DIM Dimension of the state vector.
   * @tparam CONTROL_DIM Dimension of the control vector (defaults to 0).
   *
   * The update method is templated on MEASUREMENT_DIM, allowing measurements of
   * arbitrary dimension.
   */
  template <size_t STATE_DIM, size_t CONTROL_DIM = 0>
  class KalmanFilter
  {
  public:
    /**
     * @brief Constructor to initialize the Kalman filter with identity matrices.
     */
    KalmanFilter()
      : state(), // Zero-initialized by default
        covariance(linalg::Matrix<STATE_DIM, STATE_DIM>::identity()),
        transition_matrix(linalg::Matrix<STATE_DIM, STATE_DIM>::identity()),
        control_matrix(), // Zero-initialized by default
        process_noise(linalg::Matrix<STATE_DIM, STATE_DIM>::identity())
    {
    }

    /**
     * @brief Performs the prediction step of the Kalman filter.
     *
     * The predict step estimates the next state and covariance based on the
     * state transition model. It applies the linear state transition matrix (F)
     * and optionally includes the influence of a control vector (u) through the
     * control matrix (B).
     *
     * Equations:
     * - Predicted state: x' = F * x + B * u
     * - Predicted covariance: P' = F * P * F^T + Q
     *
     * @param control The control vector, u (optional). If not provided, it
     * defaults to a zero vector, and the control matrix (B) is ignored.
     */
    void predict(const linalg::Vector<CONTROL_DIM>& control = linalg::Vector<CONTROL_DIM>())
    {
      // State prediction: x = F * x + B * u
      if constexpr (CONTROL_DIM > 0)
      {
        static_assert(CONTROL_DIM == CONTROL_DIM, "Control vector size must match CONTROL_DIM");
        state = transition_matrix * state + control_matrix * control;
      }
      else
      {
        state = transition_matrix * state;
      }

      // Covariance prediction: P = F * P * F^T + Q
      covariance = transition_matrix * covariance * transition_matrix.transpose() + process_noise;
    }

    /**
     * @brief Performs the update step of the Kalman filter.
     *
     * The update step incorporates a new measurement to correct the predicted
     * state and covariance. It calculates the innovation (difference between the
     * predicted and actual measurement), the Kalman gain, and updates the state
     * and covariance accordingly.
     *
     * This method is templated on MEASUREMENT_DIM, allowing measurements of
     * arbitrary dimension.
     *
     * Equations:
     * - Innovation: y = z - H * x
     * - Innovation covariance: S = H * P * H^T + R
     * - Kalman gain: K = P * H^T * S^(-1)
     * - Updated state: x = x + K * y
     * - Updated covariance: P = (I - K * H) * P
     *
     * @param measurement The measurement vector, z, representing the observed
     *                    data to incorporate into the state estimate.
     * @param measurement_matrix The measurement matrix, H.
     * @param measurement_noise The measurement noise covariance matrix, R.
     */
    template <size_t MEASUREMENT_DIM>
    void update(const linalg::Vector<MEASUREMENT_DIM>&                  measurement,
                const linalg::Matrix<MEASUREMENT_DIM, STATE_DIM>&       measurement_matrix,
                const linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM>& measurement_noise)
    {
      static_assert(MEASUREMENT_DIM > 0, "Measurement dimension must be greater than 0");

      // Innovation y = z - Hx
      linalg::Vector<MEASUREMENT_DIM> y = measurement - measurement_matrix * state;

      // Transpose of measurement matrix
      linalg::Matrix<STATE_DIM, MEASUREMENT_DIM> Ht = measurement_matrix.transpose();

      // Predictied covariance mapped to measurement space: P * H^T
      linalg::Matrix<STATE_DIM, MEASUREMENT_DIM> PHt = covariance * Ht;

      // Innovation covariance S = H * P * H^T + R
      linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM> S =
          measurement_matrix * PHt + measurement_noise;

      // Kalman gain: K = P * H^T * S^(-1)
      linalg::Matrix<STATE_DIM, MEASUREMENT_DIM> K = PHt * S.invert();

      // Update state: x = x + K * y
      state = state + K * y;

      // Update covariance: P = (I - K * H) * P
      linalg::Matrix<STATE_DIM, STATE_DIM> I = linalg::Matrix<STATE_DIM, STATE_DIM>::identity();
      covariance                             = (I - K * measurement_matrix) * covariance;
    }

    /**
     * @brief Returns the current state vector.
     * @return The current state vector.
     */
    const linalg::Vector<STATE_DIM>& get_state() const { return state; }

    /**
     * @brief Returns the current covariance matrix.
     * @return The current covariance matrix.
     */
    const linalg::Matrix<STATE_DIM, STATE_DIM>& get_covariance() const { return covariance; }

    /**
     * @brief Sets the state vector.
     * @param state_vector The new state vector
     */
    void set_state(linalg::Vector<STATE_DIM>& state_vector) { this->state = state_vector; }

    /**
     * @brief Sets the covariance matrix.
     * @param covariance_matrix The new covariance matrix.
     */
    void set_covariance(const linalg::Matrix<STATE_DIM, STATE_DIM>& covariance_matrix)
    {
      this->covariance = covariance_matrix;
    }

    /**
     * @brief Sets the state transition matrix.
     * @param transition_matrix The new state transition matrix.
     */
    void set_transition(const linalg::Matrix<STATE_DIM, STATE_DIM>& transition_matrix)
    {
      this->transition_matrix = transition_matrix;
    }

    /**
     * @brief Sets the control matrix.
     * @param control_matrix The new control matrix.
     */
    void set_control_matrix(const linalg::Matrix<STATE_DIM, CONTROL_DIM>& control_matrix)
    {
      this->control_matrix = control_matrix;
    }

    /**
     * @brief Sets the process noise covariance matrix.
     * @param process_noise_matrix The new process noise covariance matrix.
     */
    void set_process_noise(const linalg::Matrix<STATE_DIM, STATE_DIM>& process_noise_matrix)
    {
      this->process_noise = process_noise_matrix;
    }

  protected:
    linalg::Vector<STATE_DIM>              state;             ///< State vector, x
    linalg::Matrix<STATE_DIM, STATE_DIM>   covariance;        ///< Uncertainty covariance matrix, P
    linalg::Matrix<STATE_DIM, STATE_DIM>   transition_matrix; ///< State transition matrix, F
    linalg::Matrix<STATE_DIM, CONTROL_DIM> control_matrix;    ///< Control transition matrix, B
    linalg::Matrix<STATE_DIM, STATE_DIM>   process_noise; ///< Process noise covariance matrix, Q
  };

  /**
   * @class ExtendedKalmanFilter
   * @brief Implements an Extended Kalman Filter (EKF) for state estimation in
   * non-linear systems.
   * @tparam STATE_DIM Dimension of the state vector.
   * @tparam CONTROL_DIM Dimension of the control vector (defaults to 0).
   *
   * The ExtendedKalmanFilter class extends the functionality of the KalmanFilter
   * to handle non-linear state transitions and measurements. The update method is
   * templated on MEASUREMENT_DIM, allowing measurements of arbitrary dimension.
   * It uses Jacobians to linearize the system around the current state estimate
   * for each predict and update step.
   */
  template <size_t STATE_DIM, size_t CONTROL_DIM = 0>
  class ExtendedKalmanFilter : public KalmanFilter<STATE_DIM, CONTROL_DIM>
  {
  public:
    /**
     * @brief Constructor to initialize the Extended Kalman Filter.
     *
     * Inherits the initialization of state, covariance, and other matrices from
     * the base KalmanFilter class.
     */
    ExtendedKalmanFilter() : KalmanFilter<STATE_DIM, CONTROL_DIM>() {}

    // Prevent use of linear predict in EKF
    void predict(const linalg::Vector<CONTROL_DIM>& = linalg::Vector<CONTROL_DIM>()) = delete;

    /**
     * @brief Performs the update step of the Extended Kalman Filter (EKF).
     *
     * The update step incorporates a new measurement to correct the predicted
     * state and covariance using a non-linear measurement function and its
     * Jacobian.
     *
     * This method is templated on MEASUREMENT_DIM, allowing measurements of
     * arbitrary dimension.
     *
     * Equations:
     * - Innovation: y = z - h(x)
     * - Innovation covariance: S = H * P * H^T + R
     * - Kalman gain: K = P * H^T * S^(-1)
     * - Updated state: x = x + K * y
     * - Updated covariance: P = (I - K * H) * P
     *
     * @param measurement The observed measurement vector, z.
     * @param measurement_noise The measurement noise covariance matrix, R.
     * @param measurement_function Non-linear measurement function, h(x).
     * @param jacobian_measurement Jacobian of the measurement function, H(x).
     */
    template <size_t MEASUREMENT_DIM>
    void
    update(const linalg::Vector<MEASUREMENT_DIM>&                  measurement,
           const linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM>& measurement_noise,
           const std::function<linalg::Vector<MEASUREMENT_DIM>(const linalg::Vector<STATE_DIM>&)>&
                                                   measurement_function,
           const std::function<linalg::Matrix<MEASUREMENT_DIM, STATE_DIM>(
               const linalg::Vector<STATE_DIM>&)>& jacobian_measurement)
    {
      static_assert(MEASUREMENT_DIM > 0, "Measurement dimension must be greater than 0");

      // Compute the predicted measurement: h(x)
      linalg::Vector<MEASUREMENT_DIM> h_x = measurement_function(this->state);

      // Compute the innovation (residual): y = z - h(x)
      linalg::Vector<MEASUREMENT_DIM> y = measurement - h_x;

      // Compute the Jacobian of the measurement function: H
      linalg::Matrix<MEASUREMENT_DIM, STATE_DIM> H = jacobian_measurement(this->state);

      // Transpose of Jacobian
      linalg::Matrix<STATE_DIM, MEASUREMENT_DIM> Ht = H.transpose();

      // Predicted covariance mapped to measurement space: P * H^T
      linalg::Matrix<STATE_DIM, MEASUREMENT_DIM> PHt = this->covariance * Ht;

      // Innovation covariance S = H * P * H^T + R
      linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM> S = H * PHt + measurement_noise;

      // Kalman gain: K = P * H^T * (H * P * H^T + R)^-1
      linalg::Matrix<STATE_DIM, MEASUREMENT_DIM> K = PHt * S.invert();

      // Update the state estimate: x = x + K * y
      this->state = this->state + K * y;

      // Update the covariance: P = (I - K * H) * P
      linalg::Matrix<STATE_DIM, STATE_DIM> I = linalg::Matrix<STATE_DIM, STATE_DIM>::identity();
      this->covariance                       = (I - K * H) * this->covariance;
    }

    /**
     * @brief Performs the prediction step of the Extended Kalman Filter using a
     * non-linear state transition.
     *
     * @param state_transition_function Non-linear state transition function, f(x,
     * u).
     * @param jacobian_transition Jacobian of the state transition function,
     * J_f(x, u).
     * @param control The control vector, u (optional).
     */
    void predict(
        const std::function<linalg::Vector<STATE_DIM>(const linalg::Vector<STATE_DIM>&,
                                                      const linalg::Vector<CONTROL_DIM>&)>&
            state_transition_function,
        const std::function<linalg::Matrix<STATE_DIM, STATE_DIM>(
            const linalg::Vector<STATE_DIM>&, const linalg::Vector<CONTROL_DIM>&)>&
                                           jacobian_transition,
        const linalg::Vector<CONTROL_DIM>& control = linalg::Vector<CONTROL_DIM>())
    {
      // Update state with non-linear function
      this->state = state_transition_function(this->state, control);

      // Linearize around current state with Jacobian
      linalg::Matrix<STATE_DIM, STATE_DIM> F = jacobian_transition(this->state, control);

      // Update covariance using linearized transition model
      this->covariance = F * this->covariance * F.transpose() + this->process_noise;
    }
  };
} // namespace kfplusplus
