FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM
)
# keep gtest's runtime consistent with ours on windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# every *_test.cpp under tests/ is compiled into the one othello_bot_tests executable
file(GLOB_RECURSE OTHELLO_BOT_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp)
add_executable(othello_bot_tests ${OTHELLO_BOT_TEST_SOURCES})
target_include_directories(othello_bot_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(othello_bot_tests PRIVATE othello_bot othello_bot_warnings GTest::gtest_main)
gtest_discover_tests(othello_bot_tests)
