#include <stdexcept>
#include <iostream>

#include "linalg.h"

// TODO: Debate on throwing exceptions for this library
namespace linalg
{

  double& Vector::operator[](int index)
  {
    if (index < 0 || index >= size())
    {
        throw std::out_of_range("Index out of bounds");
    }

    return data[index];
  }
  
  const double& Vector::operator[](int index) const
  {
    if (index < 0 || index >= size())
    {
        throw std::out_of_range("Index out of bounds");
    }

    return data[index];
  }

  Vector Vector::operator+(const Vector& other) const
  {
    if (size() != other.size())
    {
     
      throw std::invalid_argument("Vector sizes do not match");
    }
    Vector result(size());

    for (int i = 0; i < size(); ++i)
    {
      result[i] = data[i] + other[i];
    }
    return result;
  }

  Vector& Vector::operator+=(const Vector& other)
  {
          if (size() != other.size())
          {
              throw std::invalid_argument("Vector sizes do not match");
          }

          for (int i = 0; i < size(); ++i)
          {
              data[i] += other[i];
          }
          return *this;
  }

  Vector Vector::operator-(const Vector& other) const
  {
    if (size() != other.size())
    {
     
      throw std::invalid_argument("Vector sizes do not match");
    }
    Vector result(size());

    for (int i = 0; i < size(); ++i)
    {
      result[i] = data[i] - other[i];
    }
    return result;
  }

  Vector& Vector::operator-=(const Vector& other)
  {
          if (size() != other.size())
          {
              throw std::invalid_argument("Vector sizes do not match");
          }

          for (int i = 0; i < size(); ++i)
          {
              data[i] -= other[i];
          }
          return *this;
  }

  int Vector::size() const
  {
    return data.size();
  }

  double Vector::dot(const Vector& other) const
  {
    if (size() != other.size())
    {
        throw std::invalid_argument("Vector sizes do not match");
    }

    double result = 0.0;
    for (int i = 0; i < size(); ++i)
    {
        result += data[i] * other[i];
    }

    return result;
  }

  void Vector::print() const
  {
    for (double val : data)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
  }
}
