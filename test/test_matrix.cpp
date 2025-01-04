#include <gtest/gtest.h>
#include "linalg.h"

TEST(MatrixTest, ConstructorAndAccess)
{
  linalg::Matrix mat(2, 3);
  mat(0, 0) = 1.0; mat(0, 1) = 2.0; mat(0, 2) = 3.0;
  mat(1, 0) = 4.0; mat(1, 1) = 5.0; mat(1, 2) = 6.0;
  
  EXPECT_EQ(mat(0, 0), 1.0);
  EXPECT_EQ(mat(0, 1), 2.0);
  EXPECT_EQ(mat(1, 2), 6.0);

  EXPECT_THROW(mat(2, 0), std::out_of_range);
  EXPECT_THROW(mat(0, 3), std::out_of_range);
}

TEST(MatrixTest, InitializerListConstructor)
{
    linalg::Matrix mat({ {2.0, 3.0},
                         {1.0, 2.0} });

    EXPECT_EQ(mat.row_size(), 2);
    EXPECT_EQ(mat.col_size(), 2);

    EXPECT_EQ(mat(0, 0), 2.0);
    EXPECT_EQ(mat(0, 1), 3.0);
    EXPECT_EQ(mat(1, 0), 1.0);
    EXPECT_EQ(mat(1, 1), 2.0);

    linalg::Matrix mat2({ {1.0, 2.0, 3.0},
                          {4.0, 5.0, 6.0} });
    EXPECT_EQ(mat2.row_size(), 2);
    EXPECT_EQ(mat2.col_size(), 3);

    EXPECT_EQ(mat2(0, 0), 1.0);
    EXPECT_EQ(mat2(0, 1), 2.0);
    EXPECT_EQ(mat2(0, 2), 3.0);
    EXPECT_EQ(mat2(1, 0), 4.0);
    EXPECT_EQ(mat2(1, 1), 5.0);
    EXPECT_EQ(mat2(1, 2), 6.0);

    EXPECT_THROW(linalg::Matrix({ {1.0, 2.0}, {3.0} }), std::invalid_argument);
}

TEST(MatrixTest, Addition)
{
  linalg::Matrix mat1({ {1.0, 2.0},
                        {3.0, 4.0} });

  linalg::Matrix mat2({ {5.0, 6.0},
                        {7.0, 8.0} });

  linalg::Matrix expected({ {6.0, 8.0},
                            {10.0, 12.0} });

  linalg::Matrix result = mat1 + mat2;

  EXPECT_EQ(result.row_size(), expected.row_size());
  EXPECT_EQ(result.col_size(), expected.col_size());

  for (int i = 0; i < result.row_size(); ++i)
  {
    for (int j = 0; j < result.col_size(); ++j)
    {
      EXPECT_EQ(result(i, j), expected(i, j));
    }
  }
}

TEST(MatrixTest, Subtraction)
{
  linalg::Matrix mat1({ {5.0, 6.0},
                        {7.0, 8.0} });

  linalg::Matrix mat2({ {1.0, 2.0},
                        {3.0, 4.0} });

  linalg::Matrix expected({ {4.0, 4.0},
                            {4.0, 4.0} });

  linalg::Matrix result = mat1 - mat2;

  EXPECT_EQ(result.row_size(), expected.row_size());
  EXPECT_EQ(result.col_size(), expected.col_size());

  for (int i = 0; i < result.row_size(); ++i)
  {
    for (int j = 0; j < result.col_size(); ++j)
    {
      EXPECT_EQ(result(i, j), expected(i, j));
    }
  }
}

TEST(MatrixTest, Multiplication)
{
  linalg::Matrix mat1({ {1.0, 2.0},
                        {3.0, 4.0} });

  linalg::Matrix mat2({ {2.0, 0.0},
                        {1.0, 2.0} });

  linalg::Matrix expected({ {4.0, 4.0},
                            {10.0, 8.0} });

  linalg::Matrix result = mat1 * mat2;

  EXPECT_EQ(result.row_size(), expected.row_size());
  EXPECT_EQ(result.col_size(), expected.col_size());

  for (int i = 0; i < result.row_size(); ++i)
  {
    for (int j = 0; j < result.col_size(); ++j)
    {
      EXPECT_EQ(result(i, j), expected(i, j));
    }
  }
}

TEST(MatrixTest, VectorMultiplication)
{
  linalg::Matrix mat({ {1.0, 2.0},
                       {3.0, 4.0} });

  linalg::Vector vec({1.0, 2.0});

  linalg::Vector expected({5.0, 11.0});

  linalg::Vector result = mat * vec;

  EXPECT_EQ(result.size(), expected.size());

  for (int i = 0; i < result.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(result(i), expected(i));
  }
}

TEST(MatrixTest, ScalarMultiplication)
{
  linalg::Matrix mat({ {1.0, 2.0},
                       {3.0, 4.0} });

  double scalar = 2.0;
  linalg::Matrix expected({ {2.0, 4.0},
                            {6.0, 8.0} });

  linalg::Matrix result = mat * scalar;

  EXPECT_EQ(result.row_size(), expected.row_size());
  EXPECT_EQ(result.col_size(), expected.col_size());

  for (unsigned int i = 0; i < result.row_size(); ++i)
  {
    for (unsigned int j = 0; j < result.col_size(); ++j)
    {
      EXPECT_EQ(result(i, j), expected(i, j));
    }
  }

  mat *= scalar;
  for (unsigned int i = 0; i < mat.row_size(); ++i)
  {
    for (unsigned int j = 0; j < mat.col_size(); ++j)
    {
      EXPECT_EQ(mat(i, j), expected(i, j));
    }
  }
}

TEST(MatrixTest, Transpose)
{
  linalg::Matrix mat1({ {4.0, 4.0},
                        {10.0, 8.0} });

  linalg::Matrix mat1_expected({ {4.0, 10.0},
                                 {4.0, 8.0} });

  linalg::Matrix transposed1 = mat1.transpose();

  EXPECT_EQ(transposed1.row_size(), mat1_expected.row_size());
  EXPECT_EQ(transposed1.col_size(), mat1_expected.col_size());

  for (int i = 0; i < transposed1.row_size(); ++i)
  {
    for (int j = 0; j < transposed1.col_size(); ++j)
    {
      EXPECT_EQ(transposed1(i, j), mat1_expected(i, j));
    }
  }

  linalg::Matrix mat2({ {1.0, 2.0, 3.0},
                        {4.0, 5.0, 6.0} });
  linalg::Matrix mat2_expected({ {1.0, 4.0},
                                 {2.0, 5.0},
                                 {3.0, 6.0} });

  linalg::Matrix transposed2 = mat2.transpose();

  EXPECT_EQ(transposed2.row_size(), mat2_expected.row_size());
  EXPECT_EQ(transposed2.col_size(), mat2_expected.col_size());

  for (int i = 0; i < mat2_expected.row_size(); ++i)
  {
    for (int j = 0; j < mat2_expected.col_size(); ++j)
    {
      EXPECT_EQ(transposed2(i, j), mat2_expected(i, j));
    }
  }
}

TEST(MatrixTest, DeterminantBaseCases)
{
  // Test 1x1 matrix determinant
  linalg::Matrix mat1({ {5.0} });

  EXPECT_DOUBLE_EQ(mat1.determinant(), 5.0);

  // Test 2x2 matrix determinant
  linalg::Matrix mat2({ {1.0, 2.0},
                        {3.0, 4.0} });

  EXPECT_DOUBLE_EQ(mat2.determinant(), -2.0);
}

TEST(MatrixTest, Determinant)
{
  linalg::Matrix mat({ {6.0, 1.0, 1.0},
                       {4.0, -2.0, 5.0},
                       {2.0, 8.0, 7.0} });

  EXPECT_DOUBLE_EQ(mat.determinant(), -306.0);
}

TEST(MatrixTest, Inversion)
{
  linalg::Matrix mat({ {4.0, 7.0},
                       {2.0, 6.0} });

  linalg::Matrix expected({ {0.6, -0.7},
                            {-0.2, 0.4} });

  linalg::Matrix inverted = mat.invert();

  EXPECT_EQ(inverted.row_size(), expected.row_size());
  EXPECT_EQ(inverted.col_size(), expected.col_size());

  for (int i = 0; i < inverted.row_size(); ++i)
  {
    for (int j = 0; j < inverted.col_size(); ++j)
    {
      EXPECT_NEAR(inverted(i, j), expected(i, j), 1e-9);
    }
  }
}
