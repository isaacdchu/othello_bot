include(FetchContent)

# provides the othello::othello and mlfo::mlfo targets. when pulled in this way,
# neither library builds its own demo or tests.
#
# to develop against a local checkout instead of the fetched copy, configure with e.g.
#   cmake -B build -DFETCHCONTENT_SOURCE_DIR_OTHELLO=../othello -DFETCHCONTENT_SOURCE_DIR_MLFO=../mlfo

set(OTHELLO_GIT_TAG v0.1.0 CACHE STRING "Branch, tag or commit of isaacdchu/othello to use")
set(MLFO_GIT_TAG v0.1.0 CACHE STRING "Branch, tag or commit of isaacdchu/mlfo to use")

# SYSTEM keeps the libraries' headers out of this project's warnings
FetchContent_Declare(othello
    GIT_REPOSITORY https://github.com/isaacdchu/othello.git
    GIT_TAG ${OTHELLO_GIT_TAG}
    GIT_SHALLOW TRUE
    SYSTEM
)
FetchContent_Declare(mlfo
    GIT_REPOSITORY https://github.com/isaacdchu/mlfo.git
    GIT_TAG ${MLFO_GIT_TAG}
    GIT_SHALLOW TRUE
    SYSTEM
)
FetchContent_MakeAvailable(othello mlfo)
