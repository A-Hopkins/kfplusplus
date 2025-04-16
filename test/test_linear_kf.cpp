#include <gtest/gtest.h>
#include "kfplusplus.h"

TEST(KalmanFilterTest, Initialization)
{
  // Define dimensions as compile-time constants
  constexpr size_t STATE_DIM = 4;

  // Create KalmanFilter with template parameters
  kfplusplus::KalmanFilter<STATE_DIM> kf;

  // Test initialization
  EXPECT_EQ(kf.get_state().size(), STATE_DIM);
  EXPECT_EQ(kf.get_covariance().row_size(), STATE_DIM);
  EXPECT_EQ(kf.get_covariance().col_size(), STATE_DIM);

  // Check zero initialization of state vector
  for (size_t i = 0; i < STATE_DIM; ++i)
  {
    EXPECT_DOUBLE_EQ(kf.get_state()(i), 0.0);
  }

  // Check identity initialization of covariance
  for (size_t i = 0; i < STATE_DIM; ++i)
  {
    for (size_t j = 0; j < STATE_DIM; ++j)
    {
      if (i == j)
      {
        EXPECT_DOUBLE_EQ(kf.get_covariance()(i, j), 1.0);
      }
      else
      {
        EXPECT_DOUBLE_EQ(kf.get_covariance()(i, j), 0.0);
      }
    }
  }
}

TEST(KalmanFilterTest, PredictWithoutControl)
{
  // Define dimensions as compile-time constants
  constexpr size_t STATE_DIM = 2;

  // Create KalmanFilter with template parameters
  kfplusplus::KalmanFilter<STATE_DIM> kf;

  // Create fixed-size matrices with template parameters
  linalg::Matrix<STATE_DIM, STATE_DIM> F({{1.0, 1.0},
                                          {0.0, 1.0}});
  linalg::Matrix<STATE_DIM, STATE_DIM> Q({{0.1, 0.0},
                                          {0.0, 0.1}});

  // Set filter parameters
  kf.set_transition(F);
  kf.set_process_noise(Q);

  // Perform prediction without control
  kf.predict();

  // Check state prediction (should be zero since initial state is zero)
  linalg::Vector<STATE_DIM> expected_state({0.0, 0.0});
  for (size_t i = 0; i < STATE_DIM; ++i)
  {
    EXPECT_DOUBLE_EQ(kf.get_state()(i), expected_state(i));
  }

  // Check covariance prediction
  linalg::Matrix<STATE_DIM, STATE_DIM> expected_covariance({{2.1, 1.0},
                                                            {1.0, 1.1}});

  for (size_t i = 0; i < STATE_DIM; ++i)
  {
    for (size_t j = 0; j < STATE_DIM; ++j)
    {
      EXPECT_NEAR(kf.get_covariance()(i, j), expected_covariance(i, j), 1e-10);
    }
  }
}

TEST(KalmanFilterTest, PredictWithControl)
{
  // Define dimensions as compile-time constants
  constexpr size_t STATE_DIM = 2;
  constexpr size_t CONTROL_DIM = 1;

  // Create KalmanFilter with template parameters
  kfplusplus::KalmanFilter<STATE_DIM, CONTROL_DIM> kf;

  // Create fixed-size matrices with template parameters
  linalg::Matrix<STATE_DIM, STATE_DIM> F({{1.0, 1.0},
                                          {0.0, 1.0}});
  linalg::Matrix<STATE_DIM, CONTROL_DIM> B({{0.5},
                                            {1.0}});
  linalg::Matrix<STATE_DIM, STATE_DIM> Q({{0.1, 0.0},
                                          {0.0, 0.1}});

  // Set filter parameters
  kf.set_transition(F);
  kf.set_control_matrix(B);
  kf.set_process_noise(Q);

  // Perform prediction with control input
  linalg::Vector<CONTROL_DIM> control({1.0});
  kf.predict(control);

  // Check state prediction
  linalg::Vector<STATE_DIM> expected_state({0.5, 1.0});
  for (size_t i = 0; i < STATE_DIM; ++i)
  {
    EXPECT_NEAR(kf.get_state()(i), expected_state(i), 1e-10);
  }

  // Check covariance prediction
  linalg::Matrix<STATE_DIM, STATE_DIM> expected_covariance({{2.1, 1.0},
                                                            {1.0, 1.1}});

  for (size_t i = 0; i < STATE_DIM; ++i)
  {
    for (size_t j = 0; j < STATE_DIM; ++j)
    {
      EXPECT_NEAR(kf.get_covariance()(i, j), expected_covariance(i, j), 1e-10);
    }
  }
}

TEST(KalmanFilterTest, Update)
{
  // Define dimensions as compile-time constants
  constexpr size_t STATE_DIM = 2;
  constexpr size_t MEASUREMENT_DIM = 1;

  // Create KalmanFilter with template parameters
  kfplusplus::KalmanFilter<STATE_DIM, MEASUREMENT_DIM> kf;

  // Set initial state and covariance for testing
  linalg::Vector<STATE_DIM> initial_state({1.0, 2.0});
  linalg::Matrix<STATE_DIM, STATE_DIM> initial_covariance({{1.0, 0.5},
                                                          {0.5, 1.0}});
  
  // We need a mutable vector for set_state - let's create a copy
  linalg::Vector<STATE_DIM> state_copy = initial_state;
  kf.set_state(state_copy);
  kf.set_covariance(initial_covariance);

  // Create fixed-size matrices with template parameters
  linalg::Matrix<MEASUREMENT_DIM, STATE_DIM> H({{1.0, 0.0}});
  linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM> R({{0.1}});

  // Perform update with measurement
  linalg::Vector<MEASUREMENT_DIM> measurement({1.5});
  kf.update(measurement, H, R);

  // The expected state after update can be calculated by hand:
  // Innovation: y = z - H*x = 1.5 - [1 0]*[1, 2]^T = 1.5 - 1 = 0.5
  // Innovation covariance: S = H*P*H^T + R = [1 0]*[1.0 0.5; 0.5 1.0]*[1; 0] + 0.1 = 1.0 + 0.1 = 1.1
  // Kalman gain: K = P*H^T*S^-1 = [1.0 0.5; 0.5 1.0]*[1; 0]*1/1.1 = [1.0; 0.5]/1.1 = [0.9091; 0.4545]
  // Updated state: x = x + K*y = [1.0; 2.0] + [0.9091; 0.4545]*0.5 = [1.4545; 2.2273]
  // Updated covariance: P = (I - K*H)*P = [1-0.9091 0; -0.4545 1]*[1.0 0.5; 0.5 1.0] = [0.0909 0.0455; 0.5-0.4545 1.0]
  
  linalg::Vector<STATE_DIM> expected_updated_state({1.4545, 2.2273});
  for (size_t i = 0; i < STATE_DIM; ++i)
  {
    EXPECT_NEAR(kf.get_state()(i), expected_updated_state(i), 1e-4);
  }

  // Check that covariance was properly updated
  EXPECT_LT(kf.get_covariance()(0, 0), initial_covariance(0, 0)); // Should be reduced by update
}

TEST(KalmanFilterTest, EndToEndTracking)
{
  // This test simulates a simple 1D constant-velocity tracking problem
  constexpr size_t STATE_DIM = 2;         // [position, velocity]
  constexpr size_t MEASUREMENT_DIM = 1;   // [position]
  constexpr size_t CONTROL_DIM = 0;       // No control input

  kfplusplus::KalmanFilter<STATE_DIM, CONTROL_DIM> kf;

  // State transition model (constant velocity)
  double dt = 1.0; // time step in seconds
  linalg::Matrix<STATE_DIM, STATE_DIM> F({{1.0, dt}, 
                                          {0.0, 1.0}});
  kf.set_transition(F);

  // Process noise (increases with time step)
  double process_noise_factor = 0.01;
  linalg::Matrix<STATE_DIM, STATE_DIM> Q({{0.25*dt*dt*dt*dt, 0.5*dt*dt*dt},
                                          {0.5*dt*dt*dt, dt*dt}});
  Q *= process_noise_factor;
  kf.set_process_noise(Q);

  // Measurement model (we only measure position)
  linalg::Matrix<MEASUREMENT_DIM, STATE_DIM> H({{1.0, 0.0}});

  // Measurement noise
  linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM> R({{0.1}});

  // Initial state [position, velocity]
  linalg::Vector<STATE_DIM> initial_state({0.0, 1.0});
  linalg::Matrix<STATE_DIM, STATE_DIM> initial_covariance({{1.0, 0.0},
                                                          {0.0, 1.0}});
  
  // We need a mutable vector for set_state - let's create a copy
  linalg::Vector<STATE_DIM> state_copy = initial_state;
  kf.set_state(state_copy);
  kf.set_covariance(initial_covariance);

  // Create some noisy measurements of an object moving with constant velocity = 1.0
  std::vector<double> measurements = {
                                        0.9,  // position at t=1 with noise
                                        2.1,  // position at t=2 with noise
                                        2.9,  // position at t=3 with noise
                                        4.2,  // position at t=4 with noise
                                        5.0   // position at t=5 with noise
                                      };

  // Track the object
  std::vector<double> estimated_positions;
  std::vector<double> estimated_velocities;

  for (const auto& meas : measurements)
  {
    // Predict step
    kf.predict();
    
    // Update step
    linalg::Vector<MEASUREMENT_DIM> z({meas});
    kf.update(z, H, R);
    
    // Store results
    estimated_positions.push_back(kf.get_state()(0));
    estimated_velocities.push_back(kf.get_state()(1));
  }

  // The filter should converge to the true velocity of approximately 1.0
  EXPECT_NEAR(estimated_velocities.back(), 1.0, 0.2);
  
  // The final position should be close to 5.0
  EXPECT_NEAR(estimated_positions.back(), 5.0, 0.5);
}
