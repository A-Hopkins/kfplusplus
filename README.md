# Kalman Filter and Linear Algebra Library

## Overview
This project implements a **Linear Algebra Library** and a **Kalman Filter** module. The linear algebra library provides operations for vectors and matrices, while the Kalman filter facilitates state estimation in linear dynamic systems.

## Design Features
- **Zero Dynamic Memory Allocation**: Uses compile-time dimensions with C++ templates
- **Memory-Safe Operations**: No heap allocations, no exceptions, no memory leaks
- **Compile-Time Checking**: Uses static assertions to validate dimensions at compile time
- **Header-Only Implementation**: Easy integration with other projects

## Features
### Linear Algebra Library
- **Vectors**:
  - Fixed-size vectors defined at compile time
  - Basic operations: addition, subtraction, scalar multiplication, dot product
  - Access and modification of elements using `operator()`
  - No dynamic memory allocation

- **Matrices**:
  - Fixed-size matrices defined at compile time
  - Basic operations: addition, subtraction, multiplication (matrix-matrix, matrix-vector, scalars)
  - Determinant, transpose, and inversion functionalities
  - Access and modification of elements using `operator()` [per this instruction](https://isocpp.org/wiki/faq/operator-overloading#matrix-array-of-array).
  - Identity matrix generation.
  - Printing for visualization.
  - No dynamic memory allocation

### Kalman Filter
#### Linear Kalman Filter
- Linear Kalman filter implementation for state estimation.
- Configurable state, measurement, and control dimensions.
- Supports:
  - Prediction step with or without control input.
  - Update step with measurement input.
  - Adjustable process and measurement noise covariance matrices.
#### Extended Kalman Filter
- Extends the linear Kalman filter to handle non-linear systems.
- Incorporates non-linear measurement functions and their Jacobians.
- Compile-time size checking for all operations

## Getting Started
### Prerequisites
- C++ compiler (GCC 11.4.0 or compatible).
- Cmake
- gtest (dependencies added with cmake)

### Build Instructions
The project uses `CMake` for building and managing dependencies.
1. Clone the repository.
2. Configure the build:
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
   ```
3. Build the project:
   ```bash
   cmake --build build
   ```

### Running Tests
Unit tests are implemented using Google Test. To run the tests:
```bash
cd build
ctest
```

## Usage
### Examples
#### Vector Operations
```cpp
linalg::Vector<3> vec1({1.0, 2.0, 3.0});
linalg::Vector<3> vec2({4.0, 5.0, 6.0});
linalg::Vector<3> result = vec1 + vec2;
// Output: 5.0 7.0 9.0
```

#### Matrix Operations
```cpp
linalg::Matrix<2, 2> mat({{1.0, 2.0}, {3.0, 4.0}});
linalg::Matrix<2, 2> identity = linalg::Matrix<2, 2>::identity();
linalg::Matrix<2, 2> result = mat * identity;
// Output:
// 1.0 2.0
// 3.0 4.0
```

#### Kalman Filter
```cpp
constexpr size_t STATE_DIM = 4;
constexpr size_t MEASUREMENT_DIM = 2;

kfplusplus::KalmanFilter<STATE_DIM> kf;
linalg::Matrix<STATE_DIM, STATE_DIM> transition({
    {1.0, 0.0, 0.1, 0.0}, 
    {0.0, 1.0, 0.0, 0.1}, 
    {0.0, 0.0, 1.0, 0.0}, 
    {0.0, 0.0, 0.0, 1.0}
});
kf.set_transition(transition);

linalg::Matrix<MEASUREMENT_DIM, STATE_DIM> measurement_matrix({
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0}
});

linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM> measurement_noise =
    linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM>::identity() * 0.1; // Example noise

linalg::Vector<MEASUREMENT_DIM> measurement({5.0, 2.0});
kf.update<MEASUREMENT_DIM>(measurement, measurement_matrix, measurement_noise);
const linalg::Vector<STATE_DIM>& state = kf.get_state();
```

#### Extended Kalman Filter
```cpp
constexpr size_t STATE_DIM = 4;
constexpr size_t MEASUREMENT_DIM = 2;

kfplusplus::ExtendedKalmanFilter<STATE_DIM> ekf; // Note: MEASUREMENT_DIM is not a template parameter for the class itself

auto measurement_function = [](const linalg::Vector<STATE_DIM>& state)
{
  // Non-linear measurement function example: h(x)
  // Directly observes the first two state variables (e.g., position)
  return linalg::Vector<MEASUREMENT_DIM>({state(0), state(1)});
};

auto jacobian_measurement = [](const linalg::Vector<STATE_DIM>& state)
{
  // Jacobian matrix of the measurement function: H(x)
  // Partial derivatives of h(x) with respect to state variables
  return linalg::Matrix<MEASUREMENT_DIM, STATE_DIM>({
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0}
  });
};

// Define measurement vector (z)
linalg::Vector<MEASUREMENT_DIM> measurement({5.0, 2.0});

// Define measurement noise covariance matrix (R)
linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM> measurement_noise =
    linalg::Matrix<MEASUREMENT_DIM, MEASUREMENT_DIM>::identity() * 0.1; // Example noise

// Perform the EKF update step
ekf.update<MEASUREMENT_DIM>(measurement, measurement_noise, measurement_function, jacobian_measurement);

// Get the updated state
const linalg::Vector<STATE_DIM>& state = ekf.get_state();
```

## Project Structure
- `include/linalg.h`: Linear algebra library implementation (header-only)
- `include/kfplusplus.h`: Kalman and Extended Kalman Filters (header-only)
- `test/`: Unit tests for linear algebra and Kalman filter components
- `examples/`: Example applications using the library
- `CMakeLists.txt`: Build configuration
- `CMakePresets.json`: Presets for building with GCC

## Future additions
- Develop Kalman filter analysis tools.

## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to [Roger Labbe's Kalman Filter book](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)

---

