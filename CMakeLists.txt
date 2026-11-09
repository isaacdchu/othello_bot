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
option(OTHELLO_BOT_NATIVE "Release builds: tune for this machine's CPU (-march=native)" ON)
option(OTHELLO_BOT_LTO "Release builds: enable link-time optimization" ON)
option(OTHELLO_BOT_BUILD_PYTHON "Build the nanobind Python extension used by the GUI" OFF)

# set before the dependencies are included so othello and mlfo are optimized the same way
if(OTHELLO_BOT_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options($<$<CONFIG:Release>:-march=native>)
endif()

if(OTHELLO_BOT_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT othello_bot_ipo_supported OUTPUT othello_bot_ipo_error LANGUAGES CXX)
    if(othello_bot_ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
        # gcc's lto-wrapper runs its parallel LTRANS jobs through $MAKE (default "make"),
        # which mingw usually lacks, so it falls back to serial and warns. point MAKE at
        # a make that exists for every link step
        if(MINGW AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_CXX_LINKER_LAUNCHER)
            find_program(OTHELLO_BOT_LTO_MAKE NAMES mingw32-make make)
            if(OTHELLO_BOT_LTO_MAKE)
                set(CMAKE_CXX_LINKER_LAUNCHER ${CMAKE_COMMAND} -E env MAKE=${OTHELLO_BOT_LTO_MAKE})
            endif()
        endif()
    else()
        message(WARNING "LTO not supported, building without it: ${othello_bot_ipo_error}")
    endif()
endif()

# provides the othello::othello and mlfo::mlfo targets
include(cmake/dependencies.cmake)

# warnings shared by every target in this project
add_library(othello_bot_warnings INTERFACE)
target_compile_options(othello_bot_warnings INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# the bot's headers live in include/, its sources in src/; everything here links against this
add_library(othello_bot STATIC
    src/encoding.cpp
    src/mcts_player.cpp
    src/minimax_player.cpp
    src/network_player.cpp
    src/run_state.cpp
    src/self_play.cpp
    src/train_config.cpp
    src/value_network.cpp
)
target_include_directories(othello_bot PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(othello_bot PUBLIC othello::othello mlfo::mlfo PRIVATE othello_bot_warnings)
# linked into the Python extension, a shared library, on platforms that need PIC
set_target_properties(othello_bot PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE othello_bot othello_bot_warnings)

# self-play training loop for the CNN value network
add_executable(train train.cpp)
target_link_libraries(train PRIVATE othello_bot othello_bot_warnings)

# plays a trained network against mobility minimax baselines, on every core
add_executable(evaluate evaluate.cpp)
target_link_libraries(evaluate PRIVATE othello_bot othello_bot_warnings)

# plays configured players against each other and writes every game to a CSV
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE othello_bot othello_bot_warnings)

# the Python extension used by the GUI (see pyproject.toml); built by `uv sync`
if(OTHELLO_BOT_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(OTHELLO_BOT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
