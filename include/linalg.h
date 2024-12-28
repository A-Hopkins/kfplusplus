/**
 * @file linalg.h
 * @brief Linear Algebra Library for Vectors and Matrices.
 *
 * This library provides basic functionalities for performing linear algebra
 * operations such as addition, subtraction, scalar multiplication, dot products,
 * and matrix-vector or matrix-matrix multiplication.
 *
 * @section examples Examples
 *
 * **Accessing Matrix and Vector Elements Using operator():**
 * @code
 * linalg::Matrix mat({{1.0, 2.0}, {3.0, 4.0}});
 * linalg::Vector vec({5.0, 6.0});
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
 * linalg::Matrix mat({{1.0, 2.0}, {3.0, 4.0}});
 * mat.print();
 * // Output:
 * //  1.0 2.0
 * //  3.0 4.0
 * @endcode
 *
 * **Vector Initialization and Operations:**
 * @code
 * linalg::Vector vec({1.0, 2.0, 3.0});
 * linalg::Vector scaled = vec * 2.0;
 * scaled.print();
 * // Output:
 * //  2.0  4.0  6.0
 * @endcode
 *
 * **Matrix-Vector Multiplication:**
 * @code
 * linalg::Matrix mat({{1.0, 2.0}, {3.0, 4.0}});
 * linalg::Vector vec({1.0, 2.0});
 * linalg::Vector result = mat * vec;
 * result.print();
 * // Output:
 * //  5.0 11.0
 * @endcode
 */

#ifndef LINALG_H
#define LINALG_H

#include <vector>

namespace linalg 
{
  /**
   * @class Vector
   * @brief Represents a mathematical vector with basic operations.
   */
  class Vector
  {
  public:
    /**
     * @brief Constructs a vector of the given size, initialized to 0.0.
     * @param size The size of the vector.
     */
    Vector(unsigned int size) : data(size, 0.0) {}

    /**
     * @brief Constructs a vector from a std::vector.
     * @param data The data to initialize the vector.
     */
    Vector(const std::vector<double>& data) : data(data) {}

    /**
     * @brief Constructs a vector from an initializer list.
     * @param init The initializer list of values.
     */
    Vector(std::initializer_list<double> init) : data(init) {}

    /**
     * @brief Accesses the element at the given index.
     * @param index The index of the element to access.
     * @return A reference to the element at the given index.
     * @throws std::out_of_range If the index is out of bounds.
     */
    double& operator()(unsigned int index);
    /**
     * @brief Accesses the element at the given index (const version).
     * @param index The index of the element to access.
     * @return A const reference to the element at the given index.
     * @throws std::out_of_range If the index is out of bounds.
     */
    const double& operator()(unsigned int index) const;

    /**
     * @brief Adds two vectors.
     * @param other The vector to add.
     * @return A new vector representing the sum.
     * @throws std::invalid_argument If the vector sizes do not match.
     */
    Vector operator+(const Vector& other) const;

    /**
     * @brief Performs in-place vector addition.
     * @param other The vector to add.
     * @return A reference to the modified vector.
     * @throws std::invalid_argument If the vector sizes do not match.
     */
    Vector& operator+=(const Vector& other);

    /**
     * @brief Subtracts two vectors.
     * @param other The vector to subtract.
     * @return A new vector representing the difference.
     * @throws std::invalid_argument If the vector sizes do not match.
     */
    Vector operator-(const Vector& other) const;

    /**
     * @brief Performs in-place vector subtraction.
     * @param other The vector to subtract.
     * @return A reference to the modified vector.
     * @throws std::invalid_argument If the vector sizes do not match.
     */
    Vector& operator-=(const Vector& other);

    /**
     * @brief Multiplies the vector by a scalar.
     * @param scalar The scalar value to multiply.
     * @return A new vector scaled by the scalar.
     */
    Vector operator*(double scalar) const;

    /**
     * @brief Performs in-place scalar multiplication.
     * @param scalar The scalar value to multiply.
     * @return A reference to the modified vector.
     */
    Vector& operator*=(double scalar);

    /**
     * @brief Computes the dot product of two vectors.
     * @param other The vector to compute the dot product with.
     * @return The dot product value.
     * @throws std::invalid_argument If the vector sizes do not match.
     */
    double dot(const Vector& other) const;

    /**
     * @brief Returns the size of the vector.
     * @return The size of the vector.
     */
    unsigned int size() const { return data.size(); }

    /**
     * @brief Prints the vector
     */
    void print() const;


  private:
    std::vector<double> data; ///< Internal storage for vector elements.
  };

  /**
   * @class Matrix
   * @brief Represents a mathematical matrix with basic operations.
   */
  class Matrix
  {
  public:
    /**
     * @brief Constructs a matrix with the given dimensions, initialized to 0.0.
     * @param rows The number of rows.
     * @param cols The number of columns.
     */
    Matrix(unsigned int rows, unsigned int cols) : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0)) {}

    /**
     * @brief Constructs a matrix from an initializer list.
     * @param values The initializer list of rows and their values.
     */
    Matrix(std::initializer_list<std::initializer_list<double>> values);

    /**
     * @brief Creates an identity matrix of the given size.
     * @param size The size of the identity matrix.
     * @return The identity matrix.
     */
    static Matrix identity(unsigned int size);

    /**
     * @brief Accesses the element at the given row and column.
     * @param row The row index.
     * @param col The column index.
     * @return A reference to the element at the given position.
     * @throws std::out_of_range If the row or column index is out of bounds.
     */
    double& operator()(unsigned int row, unsigned int col);

    /**
     * @brief Accesses the element at the given row and column (const version).
     * @param row The row index.
     * @param col The column index.
     * @return A const reference to the element at the given position.
     * @throws std::out_of_range If the row or column index is out of bounds.
     */
    const double& operator()(unsigned int row, unsigned int col) const;

    /**
     * @brief Adds two matrices.
     * @param other The matrix to add.
     * @return A new matrix representing the sum.
     * @throws std::invalid_argument If the matrix dimensions do not match.
     */
    Matrix operator+(const Matrix& other) const;

    /**
     * @brief Performs in-place matrix addition.
     * @param other The matrix to add.
     * @return A reference to the modified matrix.
     * @throws std::invalid_argument If the matrix dimensions do not match.
     */
    Matrix& operator+=(const Matrix& other);

    /**
     * @brief Subtracts two matrices.
     * @param other The matrix to subtract.
     * @return A new matrix representing the difference.
     * @throws std::invalid_argument If the matrix dimensions do not match.
     */
    Matrix operator-(const Matrix& other) const;

    /**
     * @brief Performs in-place matrix subtraction.
     * @param other The matrix to subtract.
     * @return A reference to the modified matrix.
     * @throws std::invalid_argument If the matrix dimensions do not match.
     */
    Matrix& operator-=(const Matrix& other);

    /**
     * @brief Multiplies two matrices.
     * @param other The matrix to multiply with.
     * @return A new matrix representing the product.
     * @throws std::invalid_argument If the dimensions do not align for multiplication.
     */
    Matrix operator*(const Matrix& other) const;

    /**
     * @brief Multiplies the matrix by a vector. A dot v
     * @param vec The vector to multiply with.
     * @return A new vector representing the product.
     * @throws std::invalid_argument If the matrix columns do not match the vector size.
     */
    Vector operator*(const Vector& vec) const;

    /**
     * @brief Computes the transpose of the matrix.
     * @return The transposed matrix.
     */
    Matrix transpose() const;

    /**
     * @brief Computes the determinant of the matrix.
     * @return The determinant value.
     * @throws std::invalid_argument If the matrix is not square.
     */
    double determinant() const;

    /**
     * @brief Computes the inverse of the matrix using Gauss-Jordan elimination.
     * @return The inverted matrix.
     * @throws std::runtime_error If the matrix is singular or not square.
     */
    Matrix invert() const;

    /**
     * @brief Returns the number of rows in the matrix.
     * @return The number of rows.
     */
    unsigned int row_size() const { return rows; }

    /**
     * @brief Returns the number of columns in the matrix.
     * @return The number of columns.
     */
    unsigned int col_size() const { return cols; }

    /**
     * @brief Prints the matrix in a readable format.
     */
    void print() const;

  private:
    unsigned int rows; ///< Number of rows in the matrix.
    unsigned int cols; ///< Number of columns in the matrix.

    /**
     * @brief Internal storage for matrix elements as a 2D structure.
     *
     * Each row of the matrix is represented as an inner `std::vector<double>` within
     * the outer `std::vector`. The element at the `i`-th row and `j`-th column
     * is accessed via `data[i][j]`.
     */
    std::vector<std::vector<double>> data;

    /**
     * @brief Helper function for recursive determinant calculation.
     * @param matrix The matrix to compute the determinant for.
     * @param n The size of the matrix.
     * @return The determinant value.
     */
    double calculate_determinant(const std::vector<std::vector<double>>& matrix, int n) const;
  };
}
#endif
