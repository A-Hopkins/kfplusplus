add_executable(test_vector test_vector.cpp)
target_link_libraries(test_vector PRIVATE kfplusplus gtest gtest_main)

add_test(NAME TestVectors COMMAND test_vector)
