#include <stdexcept>
#include <iomanip>
#include <iostream>

#include "linalg.h"

// TODO: Debate on throwing exceptions for this library
namespace linalg
{
  Matrix::Matrix(std::initializer_list<std::initializer_list<double>> values)
  {
    rows = values.size();
    cols = values.begin()->size();

    for (const auto& row : values)
    {
      if (row.size() != cols)
      {
        throw std::invalid_argument("All rows must have the same number of columns");
      }
    }

    data = std::vector<std::vector<double>>(rows, std::vector<double>(cols));
    int row_idx = 0;
    for (const auto& row : values)
    {
      int col_idx = 0;
      for (const auto& val : row)
      {
          data[row_idx][col_idx++] = val;
      }
      ++row_idx;
    }
  }

  Matrix Matrix::identity(unsigned int size)
  {
    Matrix I(size, size);

    for(int i = 0; i < size; ++i)
    {
      I(i, i) = 1.0;
    }

    return I;
  }

  double& Matrix::operator()(unsigned int row, unsigned int col)
  {
    if (row < 0 || row >= rows)
    {
      throw std::out_of_range("Row index out of bounds");
    }
    if (col < 0 || col >= cols)
    {
      throw std::out_of_range("Column index out of bounds");
    }
    return data[row][col];
  }

  const double& Matrix::operator()(unsigned int row, unsigned int col) const
  {
    if (row < 0 || row >= rows)
    {
      throw std::out_of_range("Row index out of bounds");
    }
    if (col < 0 || col >= cols)
    {
      throw std::out_of_range("Column index out of bounds");
    }
    return data[row][col];
  }

  Matrix Matrix::operator+(const Matrix& other) const
  {
    if (rows != other.rows || cols != other.cols)
    {
      throw std::invalid_argument("Matrix dimensions must match");
    }

    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
      for (int j = 0; j < cols; ++j)
      {
        result(i, j) = (*this)(i, j) + other(i, j);
      }
    }
    return result;
  }

  Matrix& Matrix::operator+=(const Matrix& other)
  {
    if (rows != other.rows || cols != other.cols)
    {
      throw std::invalid_argument("Matrix dimensions must match");
    }
    for (int i = 0; i < rows; ++i)
    {
      for (int j = 0; j < cols; ++j)
      {
        (*this)(i, j) += other(i, j);
      }
    }
    return *this;
  }

  Matrix Matrix::operator-(const Matrix& other) const
  {
    if (rows != other.rows || cols != other.cols)
    {
      throw std::invalid_argument("Matrix dimensions must match");
    }
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
      for (int j = 0; j < cols; ++j)
      {
        result(i, j) = (*this)(i, j) - other(i, j);
      }
    }
    return result;
  }

  Matrix& Matrix::operator-=(const Matrix& other)
  {
    if (rows != other.rows || cols != other.cols)
    {
      throw std::invalid_argument("Matrix dimensions must match");
    }
    for (int i = 0; i < rows; ++i)
    {
      for (int j = 0; j < cols; ++j)
      {
        (*this)(i, j) -= other(i, j);
      }
    }
    return (*this);
  }

  Matrix Matrix::operator*(const Matrix& other) const
  {
    if (cols != other.rows)
    {
      throw std::invalid_argument("Matrix dimensions must match for dot product");
    }
    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < other.cols; ++j)
        {
          for (int k = 0; k < cols; ++k)
          {
              result(i, j) += (*this)(i, k) * other(k, j);
          }
        }
    }
    return result;
  }

  Vector Matrix::operator*(const Vector& vec) const
  {
    if (cols != vec.size())
    {
      throw std::invalid_argument("Matrix columns must match vector size for multiplication");
    }

    Vector result(rows);
    for (int i = 0; i < rows; ++i)
    {
      for (int j = 0; j < cols; ++j)
      {
        result(i) += (*this)(i, j) * vec(j);
      }
    }
    return result;
  }

  Matrix Matrix::transpose() const
  {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; ++i)
    {
      for (int j = 0; j < cols; ++j)
      {
        result(j, i) = (*this)(i, j);
      }
    }
    return result;
  }

  double Matrix::determinant() const
  {
    if (rows != cols)
    {
      throw std::invalid_argument("Determinant can only be calculated for square matrices");
    }
    return calculate_determinant(data, rows);
  }

  double Matrix::calculate_determinant(const std::vector<std::vector<double>>& matrix, int n) const
  {
    if (n == 1)
    {
      return matrix[0][0];
    }
    if (n == 2)
    {
      return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }
    double det = 0.0;
    std::vector<std::vector<double>> sub_matrix(n - 1, std::vector<double>(n - 1));

    for (int i = 0; i < n; ++i)
    {
      for (int j = 1; j < n; ++j)
      {
        int col_index = 0;
        for (int k = 0; k < n; ++k)
        {
          if (k == i) continue;
          sub_matrix[j - 1][col_index++] = matrix[j][k];
        }
      }

      double sign = (i % 2 == 0) ? 1.0 : -1.0;
      det += sign * matrix[0][i] * calculate_determinant(sub_matrix, n - 1);
    }
    return det;
  }

  Matrix Matrix::invert() const
  {
    if (rows != cols)
    {
      throw std::invalid_argument("Inversion can only be calculated for square matrices");
    }

    // Form the augmented matrix [A | I]
    Matrix augmented(rows, 2 * rows);
    for (int i = 0; i < rows; ++i)
    {
      for (int j = 0; j < rows; ++j)
      {
        augmented(i, j) = (*this)(i, j);
      }
      augmented(i, rows + i) = 1.0;
    }

    // Perform Gauss-Jordan elimination
    for (int i = 0; i < rows; ++i)
    {
      // Make the diagonal element 1
      double diag = augmented(i, i);
      if (diag == 0)
      {
        throw std::runtime_error("Matrix is singular and cannot be inverted");
      }

      for (int j = 0; j < 2 * rows; ++j)
      {
        augmented(i, j) /= diag;
      }

      // Make other elements in the column 0
      for (int k = 0; k < rows; ++k)
      {
        if (k == i) continue;
        double factor = augmented(k, i);
        for (int j = 0; j < 2 * rows; ++j)
        {
          augmented(k, j) -= factor * augmented(i, j);
        }
      }
    }

    // Extract the inverse matrix from the augmented matrix
    Matrix inverse(rows, rows);
    for (int i = 0; i < rows; ++i)
    {
      for (int j = 0; j < rows; ++j)
      {
        inverse(i, j) = augmented(i, rows + j);
      }
    }
    return inverse;
  }

  void Matrix::print() const
  {
    for (const auto& row : data)
    {
      for (const auto& val : row)
      {
        std::cout << std::setw(5) << val << " ";
      }
      std::cout << std::endl;
    }
  }
}