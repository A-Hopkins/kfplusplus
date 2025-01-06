add_executable(test_vector test_vector.cpp)
target_link_libraries(test_vector PRIVATE kfplusplus gtest gtest_main)

add_test(NAME TestVectors COMMAND test_vector)

add_executable(test_matrix test_matrix.cpp)
target_link_libraries(test_matrix PRIVATE kfplusplus gtest gtest_main)
add_test(NAME TestMatrices COMMAND test_matrix)

add_executable(test_linear_kf test_linear_kf.cpp)
target_link_libraries(test_linear_kf PRIVATE kfplusplus gtest gtest_main)
add_test(NAME TestLinearKalmanFilter COMMAND test_linear_kf)

add_executable(test_extended_kf test_extended_kf.cpp)
target_link_libraries(test_extended_kf PRIVATE kfplusplus gtest gtest_main)
add_test(NAME TestExtendedKalmanFilter COMMAND test_extended_kf)
