#include <gtest/gtest.h>
#include "linalg.h"


TEST(VectorTest, Creation)
{
  linalg::Vector v1(3);
  EXPECT_EQ(v1.size(), 3);
  EXPECT_EQ(v1[0], 0.0);
  EXPECT_EQ(v1[1], 0.0);
  EXPECT_EQ(v1[2], 0.0);

  linalg::Vector v2({1.0, 2.0, 3.0});
  EXPECT_EQ(v2.size(), 3);
  EXPECT_EQ(v2[0], 1.0);
  EXPECT_EQ(v2[1], 2.0);
  EXPECT_EQ(v2[2], 3.0);
}

TEST(VectorTest, Addition)
{
  linalg::Vector v1({1.0, 2.0, 3.0});
  linalg::Vector v2({4.0, 5.0, 6.0});
  linalg::Vector result = v1 + v2;

  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], 5.0);
  EXPECT_EQ(result[1], 7.0);
  EXPECT_EQ(result[2], 9.0);
}

TEST(VectorTest, InPlaceAddition)
{
  linalg::Vector v1({1.0, 2.0, 3.0});
  linalg::Vector v2({4.0, 5.0, 6.0});
  v1 += v2;

  ASSERT_EQ(v1.size(), 3);
  EXPECT_EQ(v1[0], 5.0);
  EXPECT_EQ(v1[1], 7.0);
  EXPECT_EQ(v1[2], 9.0);
}

TEST(VectorTest, Subtraction)
{
  linalg::Vector v1({1.0, 2.0, 3.0});
  linalg::Vector v2({4.0, 5.0, 6.0});
  linalg::Vector result = v1 - v2;

  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], -3.0);
  EXPECT_EQ(result[1], -3.0);
  EXPECT_EQ(result[2], -3.0);
}

TEST(VectorTest, InPlaceSubtraction)
{
  linalg::Vector v1({1.0, 2.0, 3.0});
  linalg::Vector v2({4.0, 5.0, 6.0});
  v1 -= v2;

  ASSERT_EQ(v1.size(), 3);
  EXPECT_EQ(v1[0], -3.0);
  EXPECT_EQ(v1[1], -3.0);
  EXPECT_EQ(v1[2], -3.0);
}

TEST(VectorTest, DotProduct)
{
  linalg::Vector v1({1.0, 2.0, 3.0});
  linalg::Vector v2({4.0, 5.0, 6.0});
  double result = v1.dot(v2);

  EXPECT_DOUBLE_EQ(result, 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
}