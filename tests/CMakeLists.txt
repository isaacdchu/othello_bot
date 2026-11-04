FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM
)
# keep gtest's runtime consistent with ours on windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
# gtest gains nothing from LTO. the per-config property must be cleared too, since
# CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE sets it and it overrides the generic one
set_target_properties(gtest gtest_main gmock gmock_main PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION OFF
    INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF
)
# mingw gcc doesn't support the visibility attribute gtest marks its API with, so every
# translation unit including gtest warns. defining GTEST_API_ empty skips it (PUBLIC so
# the test sources get it too); gtest is linked statically, so nothing is lost
if(MINGW)
    target_compile_definitions(gtest PUBLIC GTEST_API_=)
endif()

include(GoogleTest)

# every *_test.cpp under tests/ is compiled into the one othello_bot_tests executable
file(GLOB_RECURSE OTHELLO_BOT_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp)
add_executable(othello_bot_tests ${OTHELLO_BOT_TEST_SOURCES})
target_include_directories(othello_bot_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(othello_bot_tests PRIVATE othello_bot othello_bot_warnings GTest::gtest_main)
gtest_discover_tests(othello_bot_tests)
