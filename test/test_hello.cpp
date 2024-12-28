#include <gtest/gtest.h>
#include "kfplusplus.h"

// Demonstrate some basic assertions.
TEST(MyLibraryTest, HelloFunction)
{
    testing::internal::CaptureStdout();
    kfplusplus::say_hello();
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "Hello, from kfplusplus!\n");
}
