#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "kfplusplus.h"
#include "linalg.h"

namespace py = pybind11;

// Helper: Expose a fixed-size Vector
template <size_t N>
void bind_vector(py::module_ &m, const std::string &name) {
    using Vec = linalg::Vector<N>;
    py::class_<Vec>(m, name.c_str())
        .def(py::init<>())
        .def(py::init<std::initializer_list<double>>(), py::arg("init_list"))
        .def("__getitem__", [](const Vec &v, size_t i) { return v(i); })
        .def("__setitem__", [](Vec &v, size_t i, double val) { v(i) = val; })
        .def("size", &Vec::size)
        .def("dot", &Vec::dot)
        .def("print", &Vec::print)
        .def("str", &Vec::str)
        .def("__str__", &Vec::str)
        .def("__add__", [](const Vec &a, const Vec &b) { return a + b; })
        .def("__iadd__", [](Vec &a, const Vec &b) { return a += b; })
        .def("__sub__", [](const Vec &a, const Vec &b) { return a - b; })
        .def("__isub__", [](Vec &a, const Vec &b) { return a -= b; })
        .def("__mul__", [](const Vec &a, double s) { return a * s; }, py::is_operator())
        .def("__imul__", [](Vec &a, double s) { return a *= s; }, py::is_operator())
        ;
}

// Helper: Expose a fixed-size Matrix
template <size_t ROWS, size_t COLS>
void bind_matrix(py::module_ &m, const std::string &name) {
    using Mat = linalg::Matrix<ROWS, COLS>;
    py::class_<Mat> cls(m, name.c_str());
    cls.def(py::init<>())
        .def(py::init<std::initializer_list<std::initializer_list<double>>>())
        .def_static("identity", &Mat::identity)
        .def("__getitem__", [](const Mat &mat, std::pair<size_t, size_t> idx) { return mat(idx.first, idx.second); })
        .def("__setitem__", [](Mat &mat, std::pair<size_t, size_t> idx, double val) { mat(idx.first, idx.second) = val; })
        .def("row_size", &Mat::row_size)
        .def("col_size", &Mat::col_size)
        .def("print", &Mat::print)
        .def("str", &Mat::str)
        .def("__str__", &Mat::str)
        .def("transpose", &Mat::transpose)
        .def("__add__", [](const Mat &a, const Mat &b) { return a + b; })
        .def("__iadd__", [](Mat &a, const Mat &b) { return a += b; })
        .def("__sub__", [](const Mat &a, const Mat &b) { return a - b; })
        .def("__isub__", [](Mat &a, const Mat &b) { return a -= b; })
        .def("__mul__", [](const Mat &a, double s) { return a * s; }, py::is_operator())
        .def("__imul__", [](Mat &a, double s) { return a *= s; }, py::is_operator());

    // Only bind matrix * vector if COLS == N for some N you expose
    if constexpr (COLS == 2) {
        cls.def("__matmul__", [](const Mat &a, const linalg::Vector<2> &v) { return a * v; }, py::is_operator());
    }
    if constexpr (COLS == 5) {
        cls.def("__matmul__", [](const Mat &a, const linalg::Vector<5> &v) { return a * v; }, py::is_operator());
    }
    if constexpr (COLS == 6) {
        cls.def("__matmul__", [](const Mat &a, const linalg::Vector<6> &v) { return a * v; }, py::is_operator());
    }

    // Only bind matrix * matrix, determinant, invert if square
    if constexpr (ROWS == COLS) {
        cls.def("__matmul__", [](const Mat &a, const Mat &b) { return a * b; }, py::is_operator())
           .def("determinant", &Mat::determinant)
           .def("invert", &Mat::invert);
    }
}

PYBIND11_MODULE(kfplusplus, m) {
    // Bind all required Vectors
    bind_vector<0>(m, "Vector0");
    bind_vector<2>(m, "Vector2");
    bind_vector<5>(m, "Vector5");
    bind_vector<6>(m, "Vector6");

    // Bind all required Matrices
    bind_matrix<2,2>(m, "Matrix2x2");
    bind_matrix<5,5>(m, "Matrix5x5");
    bind_matrix<6,6>(m, "Matrix6x6");
    bind_matrix<2,5>(m, "Matrix2x5");
    bind_matrix<5,2>(m, "Matrix5x2");
    bind_matrix<2,6>(m, "Matrix2x6");
    bind_matrix<5,6>(m, "Matrix5x6");
    bind_matrix<6,5>(m, "Matrix6x5");

    // KalmanFilter<5,0>
    using KF50 = kfplusplus::KalmanFilter<5, 0>;
    py::class_<KF50>(m, "KalmanFilter5x0")
        .def(py::init<>())
        .def("predict", &KF50::predict)
        .def("update", [](KF50 &kf,
                          const linalg::Vector<5> &measurement,
                          const linalg::Matrix<5, 5> &measurement_matrix,
                          const linalg::Matrix<5, 5> &measurement_noise) {
                kf.update(measurement, measurement_matrix, measurement_noise);
            },
            py::arg("measurement"),
            py::arg("measurement_matrix"),
            py::arg("measurement_noise"))
        .def("get_state", &KF50::get_state, py::return_value_policy::reference_internal)
        .def("get_covariance", &KF50::get_covariance, py::return_value_policy::reference_internal)
        .def("set_state", &KF50::set_state)
        .def("set_covariance", &KF50::set_covariance)
        .def("set_transition", &KF50::set_transition)
        .def("set_process_noise", &KF50::set_process_noise)
        ;

    // KalmanFilter<5,6>
    using KF56 = kfplusplus::KalmanFilter<5, 6>;
    py::class_<KF56>(m, "KalmanFilter5x6")
        .def(py::init<>())
        .def("predict", &KF56::predict, py::arg("control") = linalg::Vector<6>())
        .def("update", [](KF56 &kf,
                          const linalg::Vector<5> &measurement,
                          const linalg::Matrix<5, 5> &measurement_matrix,
                          const linalg::Matrix<5, 5> &measurement_noise) {
                kf.update(measurement, measurement_matrix, measurement_noise);
            },
            py::arg("measurement"),
            py::arg("measurement_matrix"),
            py::arg("measurement_noise"))
        .def("get_state", &KF56::get_state, py::return_value_policy::reference_internal)
        .def("get_covariance", &KF56::get_covariance, py::return_value_policy::reference_internal)
        .def("set_state", &KF56::set_state)
        .def("set_covariance", &KF56::set_covariance)
        .def("set_transition", &KF56::set_transition)
        .def("set_control_matrix", &KF56::set_control_matrix)
        .def("set_process_noise", &KF56::set_process_noise)
        ;

    // ExtendedKalmanFilter<5,0>
    using EKF50 = kfplusplus::ExtendedKalmanFilter<5, 0>;
    py::class_<EKF50, KF50>(m, "ExtendedKalmanFilter5x0")
        .def(py::init<>())
        // Nonlinear predict
        .def("predict",
        [](EKF50 &ekf,
           py::function state_transition_function,
           py::function jacobian_transition,
           const linalg::Vector<0> &control = linalg::Vector<0>()) {
            auto stf = [state_transition_function](const linalg::Vector<5> &state, const linalg::Vector<0> &control) {
                py::object result = state_transition_function(state, control);
                return result.cast<linalg::Vector<5>>();
            };
            auto jt = [jacobian_transition](const linalg::Vector<5> &state, const linalg::Vector<0> &control) {
                py::object result = jacobian_transition(state, control);
                return result.cast<linalg::Matrix<5,5>>();
            };
            ekf.predict(stf, jt, control);
        },
        py::arg("state_transition_function"),
        py::arg("jacobian_transition"),
        py::arg("control") = linalg::Vector<0>())
        // ODOM update (measurement_dim = 5)
        .def("update_odom",
            [](EKF50 &ekf,
               const linalg::Vector<5> &measurement,
               const linalg::Matrix<5,5> &measurement_noise,
               py::function measurement_function,
               py::function jacobian_measurement) {
                auto mf = [measurement_function](const linalg::Vector<5> &state) {
                    py::object result = measurement_function(state);
                    return result.cast<linalg::Vector<5>>();
                };
                auto jm = [jacobian_measurement](const linalg::Vector<5> &state) {
                    py::object result = jacobian_measurement(state);
                    return result.cast<linalg::Matrix<5,5>>();
                };
                ekf.update<5>(measurement, measurement_noise, mf, jm);
            },
            py::arg("measurement"),
            py::arg("measurement_noise"),
            py::arg("measurement_function"),
            py::arg("jacobian_measurement"))
        // IMU update (measurement_dim = 2)
        .def("update_imu",
            [](EKF50 &ekf,
               const linalg::Vector<2> &measurement,
               const linalg::Matrix<2,2> &measurement_noise,
               py::function measurement_function,
               py::function jacobian_measurement) {
                auto mf = [measurement_function](const linalg::Vector<5> &state) {
                    py::object result = measurement_function(state);
                    return result.cast<linalg::Vector<2>>();
                };
                auto jm = [jacobian_measurement](const linalg::Vector<5> &state) {
                    py::object result = jacobian_measurement(state);
                    return result.cast<linalg::Matrix<2,5>>();
                };
                ekf.update<2>(measurement, measurement_noise, mf, jm);
            },
            py::arg("measurement"),
            py::arg("measurement_noise"),
            py::arg("measurement_function"),
            py::arg("jacobian_measurement"))
        ;

    // ExtendedKalmanFilter<5,6>
    using EKF56 = kfplusplus::ExtendedKalmanFilter<5, 6>;
    py::class_<EKF56, KF56>(m, "ExtendedKalmanFilter5x6")
        .def(py::init<>())
        // Nonlinear predict
        .def("predict",
        [](EKF56 &ekf,
           py::function state_transition_function,
           py::function jacobian_transition,
           const linalg::Vector<6> &control = linalg::Vector<6>()) {
            auto stf = [state_transition_function](const linalg::Vector<5> &state, const linalg::Vector<6> &control) {
                py::object result = state_transition_function(state, control);
                return result.cast<linalg::Vector<5>>();
            };
            auto jt = [jacobian_transition](const linalg::Vector<5> &state, const linalg::Vector<6> &control) {
                py::object result = jacobian_transition(state, control);
                return result.cast<linalg::Matrix<5,5>>();
            };
            ekf.predict(stf, jt, control);
        },
        py::arg("state_transition_function"),
        py::arg("jacobian_transition"),
        py::arg("control") = linalg::Vector<6>())
        // ODOM update (measurement_dim = 5)
        .def("update_odom",
            [](EKF56 &ekf,
               const linalg::Vector<5> &measurement,
               const linalg::Matrix<5,5> &measurement_noise,
               py::function measurement_function,
               py::function jacobian_measurement) {
                auto mf = [measurement_function](const linalg::Vector<5> &state) {
                    py::object result = measurement_function(state);
                    return result.cast<linalg::Vector<5>>();
                };
                auto jm = [jacobian_measurement](const linalg::Vector<5> &state) {
                    py::object result = jacobian_measurement(state);
                    return result.cast<linalg::Matrix<5,5>>();
                };
                ekf.update<5>(measurement, measurement_noise, mf, jm);
            },
            py::arg("measurement"),
            py::arg("measurement_noise"),
            py::arg("measurement_function"),
            py::arg("jacobian_measurement"))
        // IMU update (measurement_dim = 2)
        .def("update_imu",
            [](EKF56 &ekf,
               const linalg::Vector<2> &measurement,
               const linalg::Matrix<2,2> &measurement_noise,
               py::function measurement_function,
               py::function jacobian_measurement) {
                auto mf = [measurement_function](const linalg::Vector<5> &state) {
                    py::object result = measurement_function(state);
                    return result.cast<linalg::Vector<2>>();
                };
                auto jm = [jacobian_measurement](const linalg::Vector<5> &state) {
                    py::object result = jacobian_measurement(state);
                    return result.cast<linalg::Matrix<2,5>>();
                };
                ekf.update<2>(measurement, measurement_noise, mf, jm);
            },
            py::arg("measurement"),
            py::arg("measurement_noise"),
            py::arg("measurement_function"),
            py::arg("jacobian_measurement"))
        ;
}