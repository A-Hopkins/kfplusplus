# Kalman Filter and Linear Algebra Library

## Overview
This project implements a **Linear Algebra Library** and a **Kalman Filter** module. The linear algebra library provides operations for vectors and matrices, while the Kalman filter facilitates state estimation in linear dynamic systems.

## Features
### Linear Algebra Library
- **Vectors**:
  - Creation from initializer lists or predefined size.
  - Basic operations: addition, subtraction, scalar multiplication, dot product.
  - Access and modification of elements using `operator()`.
  - Printing for visualization.

- **Matrices**:
  - Creation from initializer lists or specified dimensions.
  - Basic operations: addition, subtraction, multiplication (matrix-matrix and matrix-vector).
  - Determinant, transpose, and inversion functionalities.
  - Access and modification of elements using `operator()`.
  - Identity matrix generation.
  - Printing for visualization.

### Kalman Filter
- Linear Kalman filter implementation for state estimation.
- Configurable state, measurement, and control dimensions.
- Supports:
  - Prediction step with or without control input.
  - Update step with measurement input.
  - Adjustable process and measurement noise covariance matrices.

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
linalg::Vector vec1({1.0, 2.0, 3.0});
linalg::Vector vec2({4.0, 5.0, 6.0});
linalg::Vector result = vec1 + vec2;
result.print();
// Output: 5.0 7.0 9.0
```

#### Matrix Operations
```cpp
linalg::Matrix mat({{1.0, 2.0}, {3.0, 4.0}});
linalg::Matrix identity = linalg::Matrix::identity(2);
linalg::Matrix result = mat * identity;
result.print();
// Output:
// 1.0 2.0
// 3.0 4.0
```

#### Kalman Filter
```cpp
kfplusplus::KalmanFilter kf(4, 2);
kf.set_transition(linalg::Matrix({{1.0, 0.0, 0.1, 0.0}, {0.0, 1.0, 0.0, 0.1}, {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0}}));
linalg::Vector measurement({5.0, 2.0});
kf.update(measurement);
const linalg::Vector &state = kf.get_state();
state.print();
```

## Project Structure
- `linalg.h` and `linalg.cpp`: Linear Algebra Library implementation.
- `kfplusplus.h` and `linear_kf.cpp`: Kalman Filter implementation.
- `test_vector.cpp` and `test_matrix.cpp`: Unit tests for vectors and matrices.
- `CMakeLists.txt`: Build configuration.
- `CMakePresets.json`: Presets for building with GCC.

## Future additions

- Create an Extended Kalman Filter (EKF) extension.
- Add more comprehensive examples.
- Develop Kalman filter analysis tools.

## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to [Roger Labbe's Kalman Filter book](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)

---

