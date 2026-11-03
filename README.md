# othello_bot

An Othello bot written in C++23, built on [othello](https://github.com/isaacdchu/othello) (game engine) and [mlfo](https://github.com/isaacdchu/mlfo) (machine learning).

## Requirements

- CMake 3.25+
- A C++23 compiler (GCC, Clang or MSVC)
- Git and network access on first configure, since dependencies are fetched with `FetchContent`

## Build and test

```sh
cmake -B build
cmake --build build
ctest --test-dir build
```

Run the game with `./build/main`.

Set `-DOTHELLO_BOT_BUILD_TESTS=OFF` to skip building the tests.

## Dependencies

`cmake/dependencies.cmake` fetches othello and mlfo at the tags set by `OTHELLO_GIT_TAG` and `MLFO_GIT_TAG` (default `latest`, i.e. the newest `v*` tag; `v0.2.0` if the remote can't be reached). To use a different version:

```sh
cmake -B build -DOTHELLO_GIT_TAG=v0.2.0 -DMLFO_GIT_TAG=v0.2.0
```

To develop against local checkouts instead of the fetched copies:

```sh
cmake -B build -DFETCHCONTENT_SOURCE_DIR_OTHELLO=../othello -DFETCHCONTENT_SOURCE_DIR_MLFO=../mlfo
```

## Layout

- `include/othello_bot/` - the bot's headers
- `main.cpp` - entry point
- `tests/` - GoogleTest tests; every `*_test.cpp` is picked up automatically
- `cmake/` - dependency setup
