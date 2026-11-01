cmake_minimum_required(VERSION 3.25)
project(othello_bot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(OTHELLO_BOT_BUILD_TESTS "Build the othello_bot tests" ON)

# provides the othello::othello and mlfo::mlfo targets
include(cmake/dependencies.cmake)

# the bot's own headers live in include/, everything here links against this
add_library(othello_bot INTERFACE)
target_include_directories(othello_bot INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(othello_bot INTERFACE othello::othello mlfo::mlfo)

# warnings shared by every executable in this project
add_library(othello_bot_warnings INTERFACE)
target_compile_options(othello_bot_warnings INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE othello_bot othello_bot_warnings)

if(OTHELLO_BOT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
