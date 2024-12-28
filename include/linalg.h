#ifndef LINALG_H
#define LINALG_H

#include <vector>

namespace linalg 
{
  class Vector
  {
  public:
    Vector(unsigned int size) : data(size, 0.0) {}
    Vector(const std::vector<double>& data) : data(data) {}
    Vector(std::initializer_list<double> init) : data(init) {}

    double& operator()(unsigned int index);
    const double& operator()(unsigned int index) const;
    Vector operator+(const Vector& other) const;
    Vector& operator+=(const Vector& other);
    Vector operator-(const Vector& other) const;
    Vector& operator-=(const Vector& other);
    Vector operator*(double scalar) const;
    Vector& operator*=(double scalar);

    unsigned int size() const { return data.size(); }
    double dot(const Vector& other) const;

    void print() const;


  private:
    std::vector<double> data;
  };

  class Matrix
  {
  public:
    Matrix(unsigned int rows, unsigned int cols) : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0)) {}
    Matrix(std::initializer_list<std::initializer_list<double>> values);

    static Matrix identity(unsigned int size);

    double& operator()(unsigned int row, unsigned int col);
    const double& operator()(unsigned int row, unsigned int col) const;
    Matrix operator+(const Matrix& other) const;
    Matrix& operator+=(const Matrix& other);
    Matrix operator-(const Matrix& other) const;
    Matrix& operator-=(const Matrix& other);

    Matrix operator*(const Matrix& other) const;
    Vector operator*(const Vector& vec) const;

    Matrix transpose() const;
    double determinant() const;
    Matrix invert() const;

    unsigned int row_size() const { return rows; }
    unsigned int col_size() const { return cols; }

    void print() const;

  private:
    unsigned int rows;
    unsigned int cols;
    std::vector<std::vector<double>> data;

    double calculate_determinant(const std::vector<std::vector<double>>& matrix, int n) const;
  };
}
#endif
