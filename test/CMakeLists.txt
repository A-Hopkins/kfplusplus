add_executable(test_my_library test_hello.cpp)
target_link_libraries(test_my_library PRIVATE kfplusplus gtest gtest_main)

add_test(NAME TestMyLibrary COMMAND test_my_library)
