#include <vector>

namespace linalg 
{
  class Vector
  {
  public:
    Vector(int size) : data(size, 0.0) {}
    Vector(const std::vector<double>& data) : data(data) {}
    Vector(std::initializer_list<double> init) : data(init) {}

    double& operator[](int index);
    const double& operator[](int index) const;
    Vector operator+(const Vector& other) const;
    Vector& operator+=(const Vector& other);
    Vector operator-(const Vector& other) const;
    Vector& operator-=(const Vector& other);

    int size() const;
    double dot(const Vector& other) const;

    void print() const;


  private:
    std::vector<double> data;

  };

  class Matrix
  {

  };

}
