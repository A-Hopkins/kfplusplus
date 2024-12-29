/**
 * @file kfplusplus.h
 * @brief Kalman Filter c++ Implementation
 *
 * This file defines the KalmanFilter class, which models a linear Kalman filter
 * for state estimation.
 */

#ifndef KFPLUSPLUS_H
#define KFPLUSPLUS_H

#include "linalg.h"

namespace kfplusplus
{
  /**
   * @class KalmanFilter
   * @brief Implements a linear Kalman filter for state estimation.
   */
  class KalmanFilter
  {
  public:

    /**
     * @brief Constructor to initialize the Kalman filter.
     * @param state_dim Dimension of the state vector.
     * @param measurement_dim Dimension of the measurement vector.
     */
    KalmanFilter(unsigned int state_dim, unsigned int measurement_dim) :
     state_dim(state_dim), measurement_dim(measurement_dim), control_dim(0),
     state(state_dim), covariance(linalg::Matrix::identity(state_dim)), transition_matrix(linalg::Matrix::identity(state_dim)),
     measurement_matrix(linalg::Matrix(measurement_dim, state_dim)), control_matrix(linalg::Matrix(state_dim, 0)),
     control_vector(0), process_noise(linalg::Matrix::identity(state_dim)), measurement_noise(linalg::Matrix::identity(measurement_dim)) {}

    /**
     * @brief Constructor to initialize the Kalman filter.
     * @param state_dim Dimension of the state vector.
     * @param measurement_dim Dimension of the measurement vector.
     * @param control_dim Dimesnion of control vector.
     */
    KalmanFilter(unsigned int state_dim, unsigned int measurement_dim, unsigned int control_dim) :
     state_dim(state_dim), measurement_dim(measurement_dim), control_dim(control_dim),
     state(state_dim), covariance(linalg::Matrix::identity(state_dim)), transition_matrix(linalg::Matrix::identity(state_dim)),
     measurement_matrix(linalg::Matrix(measurement_dim, state_dim)), control_matrix(linalg::Matrix(state_dim, control_dim)),
     control_vector(control_dim), process_noise(linalg::Matrix::identity(state_dim)), measurement_noise(linalg::Matrix::identity(measurement_dim)) {}

    /**
     * @brief Predicts the next state and covariance.
     * @param control The control vector (optional).
     */
    void predict(const linalg::Vector &control = linalg::Vector(0));

    /**
     * @brief Updates the state and covariance with a new measurement.
     * @param measurement The measurement vector.
     */
    void update(const linalg::Vector &measurement);

    /**
     * @brief Returns the current state vector.
     * @return The current state vector.
     */
    const linalg::Vector &get_state() const { return state; }

    /**
     * @brief Returns the current covariance matrix.
     * @return The current covariance matrix.
     */
    const linalg::Matrix &get_covariance() const { return covariance; }

    /**
     * @brief Sets the state transition matrix.
     * @param transition_matrix The new state transition matrix.
     */
    void set_transition(const linalg::Matrix &transition_matrix) { this->transition_matrix = transition_matrix; }

    /**
     * @brief Sets the control matrix.
     * @param control_matrix The new control matrix.
     */
    void set_control_matrix(const linalg::Matrix &control_matrix) { this->control_matrix = control_matrix; }

    /**
     * @brief Sets the process noise covariance matrix.
     * @param process_noise_matrix The new process noise covariance matrix.
     */
    void set_process_noise(const linalg::Matrix &process_noise_matrix) { this->process_noise = process_noise_matrix; }

    /**
     * @brief Sets the measurement matrix.
     * @param measurement_matrix The new measurement matrix.
     */
    void set_measurement_matrix(const linalg::Matrix &measurement_matrix) { this->measurement_matrix = measurement_matrix; }

    /**
     * @brief Sets the measurement noise covariance matrix.
     * @param measurement_noise_matrix The new measurement noise covariance matrix.
     */
    void set_measurement_noise(const linalg::Matrix &measurement_noise_matrix) { this->measurement_noise = measurement_noise_matrix; }

  private:
    unsigned int state_dim;
    unsigned int measurement_dim;
    unsigned int control_dim;

    linalg::Vector state;              ///< State vector, x
    linalg::Matrix covariance;         ///< Uncertainty covariance matrix, P
    linalg::Matrix transition_matrix;  ///< State transition matrix, F
    linalg::Matrix measurement_matrix; ///< Measurement/Observation matrix, H
    linalg::Matrix control_matrix;     ///< Control transition matrix, B 
    linalg::Vector control_vector;     ///< Control vector, u
    linalg::Matrix process_noise;      ///< Process noise covariance matrix, Q
    linalg::Matrix measurement_noise;  ///< Measurement noise covariance matrix, R
  };
}

#endif
