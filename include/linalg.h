/**
 * @file linalg.h
 * @brief Linear Algebra Library for Vectors and Matrices.
 *
 * This library provides basic functionalities for performing linear algebra
 * operations such as addition, subtraction, scalar multiplication, dot
 * products, and matrix-vector or matrix-matrix multiplication.
 *
 * Uses fixed-size arrays with compile-time dimensions and assertions for error
 * checking.
 *
 * @section examples Examples
 *
 * **Accessing Matrix and Vector Elements Using operator():**
 * @code
 * linalg::Matrix<2, 2> mat({{1.0, 2.0}, {3.0, 4.0}});
 * linalg::Vector<2> vec({5.0, 6.0});
 *
 * // Access and modify matrix elements
 * std::cout << "mat(0, 1) = " << mat(0, 1) << std::endl;
 * // Output: mat(0, 1) = 2.0
 * mat(0, 1) = 7.0;
 * std::cout << "mat(0, 1) = " << mat(0, 1) << std::endl;
 * // Output: mat(0, 1) = 7.0
 *
 * // Access and modify vector elements
 * vec(1) = 10.0;
 * std::cout << "vec(1) = " << vec(1) << std::endl;
 * // Output: vec(1) = 10.0
 * @endcode
 *
 * **Matrix Initialization and Printing:**
 * @code
 * linalg::Matrix<2, 2> mat({{1.0, 2.0}, {3.0, 4.0}});
 * mat.print();
 * // Output:
 * //  1.0 2.0
 * //  3.0 4.0
 * @endcode
 *
 * **Vector Initialization and Operations:**
 * @code
 * linalg::Vector<3> vec({1.0, 2.0, 3.0});
 * linalg::Vector<3> scaled = vec * 2.0;
 * scaled.print();
 * // Output:
 * //  2.0  4.0  6.0
 * @endcode
 *
 * **Matrix-Vector Multiplication:**
 * @code
 * linalg::Matrix<2, 2> mat({{1.0, 2.0}, {3.0, 4.0}});
 * linalg::Vector<2> vec({1.0, 2.0});
 * linalg::Vector<2> result = mat * vec;
 * result.print();
 * // Output:
 * //  5.0 11.0
 * @endcode
 */

#pragma once
#include <array>
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>

namespace linalg
{
  /**
   * @class Vector
   * @brief Represents a mathematical vector with fixed size N.
   * @tparam N The compile-time size of the vector.
   */
  template <size_t N>
  class Vector
  {
  public:
    /**
     * @brief Constructs a vector initialized to 0.0.
     */
    Vector() { data.fill(0.0); }

    /**
     * @brief Constructs a vector from an initializer list.
     * @param init The initializer list of values.
     */
    Vector(std::initializer_list<double> init)
    {
      assert(init.size() == N && "Initializer size must match vector size");
      std::copy(init.begin(), init.end(), data.begin());
    }

    /**
     * @brief Accesses the element at the given index.
     * @param index The index of the element to access.
     * @return A reference to the element at the given index.
     */
    double& operator()(unsigned int index)
    {
      assert(index < N && "Vector index out of bounds");
      return data[index];
    }

    /**
     * @brief Accesses the element at the given index (const version).
     * @param index The index of the element to access.
     * @return A const reference to the element at the given index.
     */
    const double& operator()(unsigned int index) const
    {
      assert(index < N && "Vector index out of bounds");
      return data[index];
    }

    /**
     * @brief Adds two vectors.
     * @param other The vector to add.
     * @return A new vector representing the sum.
     */
    Vector operator+(const Vector& other) const
    {
      Vector result;
      for (size_t i = 0; i < N; ++i)
      {
        result.data[i] = data[i] + other.data[i];
      }
      return result;
    }

    /**
     * @brief Performs in-place vector addition.
     * @param other The vector to add.
     * @return A reference to the modified vector.
     */
    Vector& operator+=(const Vector& other)
    {
      for (size_t i = 0; i < N; ++i)
      {
        data[i] += other.data[i];
      }
      return *this;
    }

    /**
     * @brief Subtracts two vectors.
     * @param other The vector to subtract.
     * @return A new vector representing the difference.
     */
    Vector operator-(const Vector& other) const
    {
      Vector result;
      for (size_t i = 0; i < N; ++i)
      {
        result.data[i] = data[i] - other.data[i];
      }
      return result;
    }

    /**
     * @brief Performs in-place vector subtraction.
     * @param other The vector to subtract.
     * @return A reference to the modified vector.
     */
    Vector& operator-=(const Vector& other)
    {
      for (size_t i = 0; i < N; ++i)
      {
        data[i] -= other.data[i];
      }
      return *this;
    }

    /**
     * @brief Multiplies the vector by a scalar.
     * @param scalar The scalar value to multiply.
     * @return A new vector scaled by the scalar.
     */
    Vector operator*(double scalar) const
    {
      Vector result;
      for (size_t i = 0; i < N; ++i)
      {
        result.data[i] = data[i] * scalar;
      }
      return result;
    }

    /**
     * @brief Performs in-place scalar multiplication.
     * @param scalar The scalar value to multiply.
     * @return A reference to the modified vector.
     */
    Vector& operator*=(double scalar)
    {
      for (size_t i = 0; i < N; ++i)
      {
        data[i] *= scalar;
      }
      return *this;
    }

    /**
     * @brief Computes the dot product of two vectors.
     * @param other The vector to compute the dot product with.
     * @return The dot product value.
     */
    double dot(const Vector& other) const
    {
      double sum = 0.0;
      for (size_t i = 0; i < N; ++i)
      {
        sum += data[i] * other.data[i];
      }
      return sum;
    }

    /**
     * @brief Returns the size of the vector.
     * @return The size of the vector.
     */
    unsigned int size() const { return N; }

    /**
     * @brief Prints the vector
     */
    void print() const
    {
      for (size_t i = 0; i < N; ++i)
      {
        std::cout << " " << data[i];
      }
      std::cout << std::endl;
    }

    /**
     * @brief Converts to string
     * @return String in the form "[x1, x2, ..., xN]".
     */
    std::string str() const
    {
      std::ostringstream oss;
      oss << "[";
      for (size_t i = 0; i < N; ++i)
      {
        oss << data[i];
        if (i != N - 1)
        {
          oss << ", ";
        }
      }
      oss << "]";
      return oss.str();
    }

  private:
    std::array<double, N> data; ///< Internal storage for vector elements.
  };

  /**
   * @class Matrix
   * @brief Represents a mathematical matrix with fixed dimensions.
   * @tparam ROWS Number of rows.
   * @tparam COLS Number of columns.
   */
  template <size_t ROWS, size_t COLS>
  class Matrix
  {
  public:
    /**
     * @brief Constructs a matrix initialized to 0.0.
     */
    Matrix()
    {
      for (auto& row : data)
      {
        row.fill(0.0);
      }
    }

    /**
     * @brief Constructs a matrix from an initializer list.
     * @param values The initializer list of rows and their values.
     */
    Matrix(std::initializer_list<std::initializer_list<double>> values)
    {
      assert(values.size() == ROWS && "Matrix initializer must match row count");

      auto row_it = values.begin();
      for (size_t i = 0; i < ROWS; ++i, ++row_it)
      {
        assert(row_it->size() == COLS && "Matrix initializer must match column count");
        std::copy(row_it->begin(), row_it->end(), data[i].begin());
      }
    }

    /**
     * @brief Creates an identity matrix of the given size.
     * @return The identity matrix.
     */
    static Matrix<ROWS, COLS> identity()
    {
      assert(ROWS == COLS && "Identity matrix must be square");

      Matrix<ROWS, COLS> result;
      for (size_t i = 0; i < ROWS; ++i)
      {
        result.data[i][i] = 1.0;
      }
      return result;
    }

    /**
     * @brief Accesses the element at the given row and column.
     * @param row The row index.
     * @param col The column index.
     * @return A reference to the element at the given position.
     */
    double& operator()(unsigned int row, unsigned int col)
    {
      assert(row < ROWS && "Row index out of bounds");
      assert(col < COLS && "Column index out of bounds");
      return data[row][col];
    }

    /**
     * @brief Accesses the element at the given row and column (const version).
     * @param row The row index.
     * @param col The column index.
     * @return A const reference to the element at the given position.
     */
    const double& operator()(unsigned int row, unsigned int col) const
    {
      assert(row < ROWS && "Row index out of bounds");
      assert(col < COLS && "Column index out of bounds");
      return data[row][col];
    }

    /**
     * @brief Adds two matrices.
     * @param other The matrix to add.
     * @return A new matrix representing the sum.
     */
    Matrix operator+(const Matrix& other) const
    {
      Matrix result;
      for (size_t i = 0; i < ROWS; ++i)
      {
        for (size_t j = 0; j < COLS; ++j)
        {
          result.data[i][j] = data[i][j] + other.data[i][j];
        }
      }
      return result;
    }

    /**
     * @brief Performs in-place matrix addition.
     * @param other The matrix to add.
     * @return A reference to the modified matrix.
     */
    Matrix& operator+=(const Matrix& other)
    {
      for (size_t i = 0; i < ROWS; ++i)
      {
        for (size_t j = 0; j < COLS; ++j)
        {
          data[i][j] += other.data[i][j];
        }
      }
      return *this;
    }

    /**
     * @brief Subtracts two matrices.
     * @param other The matrix to subtract.
     * @return A new matrix representing the difference.
     */
    Matrix operator-(const Matrix& other) const
    {
      Matrix result;
      for (size_t i = 0; i < ROWS; ++i)
      {
        for (size_t j = 0; j < COLS; ++j)
        {
          result.data[i][j] = data[i][j] - other.data[i][j];
        }
      }
      return result;
    }

    /**
     * @brief Performs in-place matrix subtraction.
     * @param other The matrix to subtract.
     * @return A reference to the modified matrix.
     */
    Matrix& operator-=(const Matrix& other)
    {
      for (size_t i = 0; i < ROWS; ++i)
      {
        for (size_t j = 0; j < COLS; ++j)
        {
          data[i][j] -= other.data[i][j];
        }
      }
      return *this;
    }

    /**
     * @brief Multiplies two matrices.
     * @param other The matrix to multiply with.
     * @return A new matrix representing the product.
     */
    template <size_t OTHER_COLS>
    Matrix<ROWS, OTHER_COLS> operator*(const Matrix<COLS, OTHER_COLS>& other) const
    {
      Matrix<ROWS, OTHER_COLS> result;
      for (size_t i = 0; i < ROWS; ++i)
      {
        for (size_t j = 0; j < OTHER_COLS; ++j)
        {
          double sum = 0.0;
          for (size_t k = 0; k < COLS; ++k)
          {
            sum += data[i][k] * other(k, j);
          }
          result(i, j) = sum;
        }
      }
      return result;
    }

    /**
     * @brief Multiplies the matrix by a vector.
     * @param vec The vector to multiply with.
     * @return A new vector representing the product.
     */
    Vector<ROWS> operator*(const Vector<COLS>& vec) const
    {
      Vector<ROWS> result;
      for (size_t i = 0; i < ROWS; ++i)
      {
        double sum = 0.0;
        for (size_t j = 0; j < COLS; ++j)
        {
          sum += data[i][j] * vec(j);
        }
        result(i) = sum;
      }
      return result;
    }

    /**
     * @brief Multiplies the matrix by a scalar.
     * @param scalar The scalar value to multiply.
     * @return A new matrix scaled by the scalar.
     */
    Matrix operator*(double scalar) const
    {
      Matrix result;
      for (size_t i = 0; i < ROWS; ++i)
      {
        for (size_t j = 0; j < COLS; ++j)
        {
          result.data[i][j] = data[i][j] * scalar;
        }
      }
      return result;
    }

    /**
     * @brief Performs in-place scalar multiplication.
     * @param scalar The scalar value to multiply.
     * @return A reference to the modified matrix.
     */
    Matrix& operator*=(double scalar)
    {
      for (size_t i = 0; i < ROWS; ++i)
      {
        for (size_t j = 0; j < COLS; ++j)
        {
          data[i][j] *= scalar;
        }
      }
      return *this;
    }

    /**
     * @brief Computes the transpose of the matrix.
     * @return The transposed matrix.
     */
    Matrix<COLS, ROWS> transpose() const
    {
      Matrix<COLS, ROWS> result;
      for (size_t i = 0; i < ROWS; ++i)
      {
        for (size_t j = 0; j < COLS; ++j)
        {
          result(j, i) = data[i][j];
        }
      }
      return result;
    }

    /**
     * @brief Computes the determinant of the matrix.
     * @return The determinant value.
     * @note Requires a square matrix, checked with static_assert.
     */
    double determinant() const
    {
      // Square matrix check at compile time
      static_assert(ROWS == COLS, "Determinant can only be calculated for square matrices");

      // Special cases for small matrices for better performance
      if constexpr (ROWS == 1)
      {
        return data[0][0];
      }
      else if constexpr (ROWS == 2)
      {
        return data[0][0] * data[1][1] - data[0][1] * data[1][0];
      }
      else if constexpr (ROWS == 3)
      {
        return data[0][0] * (data[1][1] * data[2][2] - data[1][2] * data[2][1]) -
               data[0][1] * (data[1][0] * data[2][2] - data[1][2] * data[2][0]) +
               data[0][2] * (data[1][0] * data[2][1] - data[1][1] * data[2][0]);
      }
      else
      {
        // LU decomposition with partial pivoting
        std::array<std::array<double, COLS>, ROWS> lu = data;
        std::array<size_t, ROWS>                   perm;

        for (size_t i = 0; i < ROWS; ++i)
        {
          perm[i] = i;
        }

        int sign = 1;

        for (size_t k = 0; k < ROWS; ++k)
        {
          // pivot selection
          double maxVal = 0.0;
          size_t pivot  = k;
          for (size_t i = k; i < ROWS; ++i)
          {
            double v = std::abs(lu[i][k]);
            if (v > maxVal)
            {
              maxVal = v;
              pivot  = i;
            }
          }
          assert(maxVal > 1e-12 && "Matrix is singular");
          if (pivot != k)
          {
            std::swap(lu[k], lu[pivot]);
            std::swap(perm[k], perm[pivot]);
            sign = -sign;
          }
          // elimination
          for (size_t i = k + 1; i < ROWS; ++i)
          {
            lu[i][k] /= lu[k][k];
            for (size_t j = k + 1; j < ROWS; ++j)
            {
              lu[i][j] -= lu[i][k] * lu[k][j];
            }
          }
        }

        // determinant = sign * product of diagonal U
        double det = sign;
        for (size_t i = 0; i < ROWS; ++i)
        {
          det *= lu[i][i];
        }
        return det;
      }
    }

    /**
     * @brief Computes the inverse of the matrix using Gauss-Jordan elimination.
     * @return The inverted matrix.
     * @note Requires a square matrix, checked with static_assert.
     *       Will assert if the matrix is singular (determinant is zero).
     */
    Matrix invert() const
    {
      // Square matrix check at compile time
      static_assert(ROWS == COLS, "Matrix must be square to compute inverse");

      // Check if the matrix is invertible (non-singular)
      double det = determinant();
      assert(std::abs(det) > 1e-12 && "Matrix is singular and cannot be inverted");

      // For 1x1 matrix, simple reciprocal
      if constexpr (ROWS == 1)
      {
        Matrix result;
        result(0, 0) = 1.0 / data[0][0];
        return result;
      }
      // For 2x2 matrix, use analytical formula
      else if constexpr (ROWS == 2)
      {
        Matrix result;
        double inv_det = 1.0 / det;
        result(0, 0)   = data[1][1] * inv_det;
        result(0, 1)   = -data[0][1] * inv_det;
        result(1, 0)   = -data[1][0] * inv_det;
        result(1, 1)   = data[0][0] * inv_det;
        return result;
      }
      // For larger we use LU to invert
      else
      {
        // 1. Factor A = P·L·U
        std::array<std::array<double, COLS>, ROWS> lu = data;
        std::array<size_t, ROWS>                   perm;
        for (size_t i = 0; i < ROWS; ++i)
          perm[i] = i;

        for (size_t k = 0; k < ROWS; ++k)
        {
          double maxVal = 0.0;
          size_t pivot  = k;
          for (size_t i = k; i < ROWS; ++i)
          {
            double v = std::abs(lu[i][k]);
            if (v > maxVal)
            {
              maxVal = v;
              pivot  = i;
            }
          }
          assert(maxVal > 1e-12 && "Matrix is singular");
          if (pivot != k)
          {
            std::swap(lu[k], lu[pivot]);
            std::swap(perm[k], perm[pivot]);
          }
          for (size_t i = k + 1; i < ROWS; ++i)
          {
            lu[i][k] /= lu[k][k];
            for (size_t j = k + 1; j < ROWS; ++j)
            {
              lu[i][j] -= lu[i][k] * lu[k][j];
            }
          }
        }

        // 2. Solve n right‑hand sides: A^-1*E = X -> P*L*U*X = E
        Matrix result;
        for (size_t col = 0; col < COLS; ++col)
        {
          // build b = P·e_col
          std::array<double, ROWS> b{};
          for (size_t i = 0; i < ROWS; ++i)
          {
            b[i] = (perm[i] == col ? 1.0 : 0.0);
          }

          // forward solve L·y = b
          std::array<double, ROWS> y{};
          for (size_t i = 0; i < ROWS; ++i)
          {
            double sum = b[i];
            for (size_t j = 0; j < i; ++j)
            {
              sum -= lu[i][j] * y[j];
            }
            y[i] = sum;
          }
          // back solve U·x = y
          std::array<double, ROWS> x{};
          for (int i = int(ROWS) - 1; i >= 0; --i)
          {
            double sum = y[i];
            for (size_t j = i + 1; j < ROWS; ++j)
            {
              sum -= lu[i][j] * x[j];
            }
            x[i] = sum / lu[i][i];
          }
          // write into result’s col-th column
          for (size_t i = 0; i < ROWS; ++i)
          {
            result(i, col) = x[i];
          }
        }

        return result;
      }
    }

    /**
     * @brief Returns the number of rows in the matrix.
     * @return The number of rows.
     */
    constexpr size_t row_size() const { return ROWS; }

    /**
     * @brief Returns the number of columns in the matrix.
     * @return The number of columns.
     */
    constexpr size_t col_size() const { return COLS; }

    /**
     * @brief Prints the matrix in a readable format.
     */
    void print() const
    {
      for (size_t i = 0; i < ROWS; ++i)
      {
        for (size_t j = 0; j < COLS; ++j)
        {
          std::cout << " " << data[i][j];
        }
        std::cout << std::endl;
      }
    }

    /**
     * @brief Returns a formatted string representation of the matrix.
     * @return Multi-line string in the form:
     * [ [r1c1, r1c2, ...],
     *   [r2c1, r2c2, ...],
     *   ... ]
     */
    std::string str() const
    {
      std::ostringstream oss;
      oss << "[\n";
      for (size_t i = 0; i < ROWS; ++i)
      {
        oss << "  [";
        for (size_t j = 0; j < COLS; ++j)
        {
          oss << data[i][j];
          if (j != COLS - 1)
            oss << ", ";
        }
        oss << "]";
        if (i != ROWS - 1)
        {
          oss << ",";
        }
        oss << "\n";
      }
      oss << "]";
      return oss.str();
    }

  private:
    /**
     * @brief Internal storage for matrix elements as a fixed-size 2D structure.
     *
     * Each row of the matrix is represented as an inner `std::array<double,
     * COLS>` within the outer `std::array` of size ROWS. The element at the
     * `i`-th row and `j`-th column is accessed via `data[i][j]`. This
     * implementation uses compile-time fixed-size arrays to avoid dynamic memory
     * allocation.
     */
    std::array<std::array<double, COLS>, ROWS> data; ///< Internal storage for matrix elements.
  };

  /**
   * @brief Computes the norm (magnitude) of a vector.
   * @param v The vector to compute the norm for.
   * @return The norm of the vector.
   */
  template <size_t N>
  double norm(const Vector<N>& v)
  {
    assert(N > 0 && "Vector must have non-zero dimension for norm()");
    return std::sqrt(v.dot(v));
  }

} // namespace linalg
