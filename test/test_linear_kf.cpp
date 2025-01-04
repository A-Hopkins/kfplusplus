#include <gtest/gtest.h>
#include "kfplusplus.h"

TEST(KalmanFilterTest, Initialization)
{
  unsigned int state_dim = 4;
  unsigned int measurement_dim = 2;

  kfplusplus::KalmanFilter kf(state_dim, measurement_dim);

  EXPECT_EQ(kf.get_state().size(), state_dim);
  EXPECT_EQ(kf.get_covariance().row_size(), state_dim);
  EXPECT_EQ(kf.get_covariance().col_size(), state_dim);
}

TEST(KalmanFilterTest, PredictWithoutControl)
{
  unsigned int state_dim = 2;
  unsigned int measurement_dim = 1;

  kfplusplus::KalmanFilter kf(state_dim, measurement_dim);

  linalg::Matrix F({ {1.0, 1.0},
                     {0.0, 1.0} });
  linalg::Matrix Q({ {0.1, 0.0},
                     {0.0, 0.1} });

  kf.set_transition(F);
  kf.set_process_noise(Q);

  kf.predict();

  linalg::Vector expected_state({0.0, 0.0});
  EXPECT_EQ(kf.get_state().size(), expected_state.size());

  linalg::Matrix expected_covariance({ {1.1, 1.0},
                                       {1.0, 1.1} });

  EXPECT_EQ(kf.get_covariance().row_size(), expected_covariance.row_size());
  EXPECT_EQ(kf.get_covariance().col_size(), expected_covariance.col_size());
}

TEST(KalmanFilterTest, PredictWithControl)
{
  unsigned int state_dim = 2;
  unsigned int measurement_dim = 1;
  unsigned int control_dim = 1;

  kfplusplus::KalmanFilter kf(state_dim, measurement_dim, control_dim);

  linalg::Matrix F({ {1.0, 1.0},
                     {0.0, 1.0} });
  linalg::Matrix B({ {0.5},
                     {1.0} });
  linalg::Matrix Q({ {0.1, 0.0},
                     {0.0, 0.1} });

  kf.set_transition(F);
  kf.set_control_matrix(B);
  kf.set_process_noise(Q);

  linalg::Vector control({1.0});
  kf.predict(control);

  linalg::Vector expected_state({0.5, 1.0});
  EXPECT_EQ(kf.get_state().size(), expected_state.size());

  linalg::Matrix expected_covariance({ {1.1, 1.0},
                                       {1.0, 1.1} });
  EXPECT_EQ(kf.get_covariance().row_size(), expected_covariance.row_size());
}

TEST(KalmanFilterTest, Update)
{
  unsigned int state_dim = 2;
  unsigned int measurement_dim = 1;

  kfplusplus::KalmanFilter kf(state_dim, measurement_dim);

  linalg::Matrix H({ {1.0, 0.0} });
  linalg::Matrix R({ {0.1} });

  kf.set_measurement_matrix(H);
  kf.set_measurement_noise(R);

  linalg::Vector measurement({1.0});
  kf.update(measurement);

  linalg::Vector updated_state = kf.get_state();
  EXPECT_EQ(updated_state.size(), state_dim);

  linalg::Matrix updated_covariance = kf.get_covariance();
  EXPECT_EQ(updated_covariance.row_size(), state_dim);
}
