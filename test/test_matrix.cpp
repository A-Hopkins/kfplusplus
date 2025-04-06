#include <gtest/gtest.h>
#include "linalg.h"

TEST(MatrixTest, ConstructorAndAccess)
{
  linalg::Matrix<2, 3> mat;
  mat(0, 0) = 1.0; mat(0, 1) = 2.0; mat(0, 2) = 3.0;
  mat(1, 0) = 4.0; mat(1, 1) = 5.0; mat(1, 2) = 6.0;
  
  EXPECT_EQ(mat(0, 0), 1.0);
  EXPECT_EQ(mat(0, 1), 2.0);
  EXPECT_EQ(mat(1, 2), 6.0);
}

TEST(MatrixTest, InitializerListConstructor)
{
  linalg::Matrix<2, 2> mat({ {2.0, 3.0},
    {1.0, 2.0} });

  EXPECT_EQ(mat.row_size(), 2);
  EXPECT_EQ(mat.col_size(), 2);

  EXPECT_EQ(mat(0, 0), 2.0);
  EXPECT_EQ(mat(0, 1), 3.0);
  EXPECT_EQ(mat(1, 0), 1.0);
  EXPECT_EQ(mat(1, 1), 2.0);

  linalg::Matrix<2, 3> mat2({ {1.0, 2.0, 3.0},
                              {4.0, 5.0, 6.0} });

  EXPECT_EQ(mat2.row_size(), 2);
  EXPECT_EQ(mat2.col_size(), 3);

  EXPECT_EQ(mat2(0, 0), 1.0);
  EXPECT_EQ(mat2(0, 1), 2.0);
  EXPECT_EQ(mat2(0, 2), 3.0);
  EXPECT_EQ(mat2(1, 0), 4.0);
  EXPECT_EQ(mat2(1, 1), 5.0);
  EXPECT_EQ(mat2(1, 2), 6.0);
}

TEST(MatrixTest, Addition)
{
  linalg::Matrix<2, 2> mat1({ {1.0, 2.0},
                              {3.0, 4.0} });

  linalg::Matrix<2, 2> mat2({ {5.0, 6.0},
                              {7.0, 8.0} });

  linalg::Matrix<2, 2> expected({ {6.0, 8.0},
                                  {10.0, 12.0} });

  linalg::Matrix<2, 2> result = mat1 + mat2;

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
  linalg::Matrix<2, 2> mat1({ {5.0, 6.0},
                              {7.0, 8.0} });

  linalg::Matrix<2, 2> mat2({ {1.0, 2.0},
                              {3.0, 4.0} });

  linalg::Matrix<2, 2> expected({ {4.0, 4.0},
                                  {4.0, 4.0} });

  linalg::Matrix<2, 2> result = mat1 - mat2;

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
  linalg::Matrix<2, 2> mat1({ {1.0, 2.0},
                              {3.0, 4.0} });

  linalg::Matrix<2, 2> mat2({ {2.0, 0.0},
                              {1.0, 2.0} });

  linalg::Matrix<2, 2> expected({ {4.0, 4.0},
                                  {10.0, 8.0} });

  linalg::Matrix<2, 2> result = mat1 * mat2;

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
  linalg::Matrix<2, 2> mat({ {1.0, 2.0},
                             {3.0, 4.0} });

  linalg::Vector<2> vec({1.0, 2.0});

  linalg::Vector<2> expected({5.0, 11.0});

  linalg::Vector<2> result = mat * vec;

  EXPECT_EQ(result.size(), expected.size());

  for (int i = 0; i < result.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(result(i), expected(i));
  }
}

TEST(MatrixTest, ScalarMultiplication)
{
  linalg::Matrix<2, 2> mat({ {1.0, 2.0},
    {3.0, 4.0} });

  double scalar = 2.0;
  linalg::Matrix<2, 2> expected({ {2.0, 4.0},
                                  {6.0, 8.0} });

  linalg::Matrix<2, 2> result = mat * scalar;

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
  linalg::Matrix<2, 2> mat1({ {4.0, 4.0},
                              {10.0, 8.0} });

  linalg::Matrix<2, 2> mat1_expected({ {4.0, 10.0},
                                       {4.0, 8.0} });

  linalg::Matrix<2, 2> transposed1 = mat1.transpose();

  EXPECT_EQ(transposed1.row_size(), mat1_expected.row_size());
  EXPECT_EQ(transposed1.col_size(), mat1_expected.col_size());

  for (int i = 0; i < transposed1.row_size(); ++i)
  {
    for (int j = 0; j < transposed1.col_size(); ++j)
    {
      EXPECT_EQ(transposed1(i, j), mat1_expected(i, j));
    }
  }

  linalg::Matrix<2, 3> mat2({ {1.0, 2.0, 3.0},
                              {4.0, 5.0, 6.0} });

linalg::Matrix<3, 2> mat2_expected({ {1.0, 4.0},
                                     {2.0, 5.0},
                                     {3.0, 6.0} });

linalg::Matrix<3, 2> transposed2 = mat2.transpose();


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
  linalg::Matrix<1, 1> mat1({ {5.0} });

  EXPECT_DOUBLE_EQ(mat1.determinant(), 5.0);

  // Test 2x2 matrix determinant
  linalg::Matrix<2, 2> mat2({ {1.0, 2.0},
                              {3.0, 4.0} });

  EXPECT_DOUBLE_EQ(mat2.determinant(), -2.0);
}

TEST(MatrixTest, Determinant)
{
  linalg::Matrix<3, 3> mat({ {6.0, 1.0, 1.0},
                             {4.0, -2.0, 5.0},
                             {2.0, 8.0, 7.0} });

  EXPECT_DOUBLE_EQ(mat.determinant(), -306.0);
}

TEST(MatrixTest, Inversion)
{
  linalg::Matrix<2, 2> mat({ {4.0, 7.0},
                             {2.0, 6.0} });

  linalg::Matrix<2, 2> expected({ {0.6, -0.7},
                                  {-0.2, 0.4} });

  linalg::Matrix<2, 2> inverted = mat.invert();

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
