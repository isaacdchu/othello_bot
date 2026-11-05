# othello_bot

An Othello bot written in C++23, built on [othello](https://github.com/isaacdchu/othello) (game engine) and [mlfo](https://github.com/isaacdchu/mlfo) (machine learning).

## Requirements

- CMake 3.25+
- A C++23 compiler (GCC)
- Git and network access on first configure, since dependencies are fetched with `FetchContent`

## Build and test

```sh
cmake -B build
cmake --build build
ctest --test-dir build
```

Run the game with `./build/main`.

## Training and evaluation

```sh
./build/train --config configs/v2.txt            # see train.cpp for every setting
./build/evaluate runs/<run>/latest.ckpt --depth 4 --baselines 8,11 --openings 50
```

`train` plays self-play and evaluation games on every core (`--threads N` to limit it); the games, and so the run, are the same for any thread count. `evaluate` plays a model against mobility minimax at each baseline depth from paired random openings and reports score, mean disc margin and seconds per move for each side. Pass `mobility_dN` as the model to compare two baseline depths.

Set `-DOTHELLO_BOT_BUILD_TESTS=OFF` to skip building the tests.

## GUI

A tkinter GUI to play against the minimax, MCTS or a trained network. It uses the C++ players through a [nanobind](https://github.com/wjakob/nanobind) extension (`python/bindings.cpp`), built by [uv](https://docs.astral.sh/uv/) with scikit-build-core:

```sh
uv run othello-gui
```

The first run configures and builds the extension into `build/python/` (about a minute); uv rebuilds it whenever the C++ changes. Pick your color and an opponent, then press **New game**. The model option defaults to the newest `runs/*/latest.ckpt`. On Windows the extension is built with MinGW gcc (`MinGW Makefiles`), the same toolchain as the rest of the project.

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
- `python/` - the nanobind bindings and the `othello_gui` package
- `tests/` - GoogleTest tests; every `*_test.cpp` is picked up automatically
- `cmake/` - dependency setup
