include(FetchContent)
find_package(Git REQUIRED)

# provides the othello::othello and mlfo::mlfo targets. when pulled in this way,
# neither library builds its own demo or tests.
#
# by default the newest v* tag of each library is looked up at configure time. to pin a
# version instead, configure with e.g.
#   cmake -B build -DOTHELLO_GIT_TAG=v0.3.0 -DMLFO_GIT_TAG=v0.4.0
#
# to develop against a local checkout instead of the fetched copy, configure with e.g.
#   cmake -B build -DFETCHCONTENT_SOURCE_DIR_OTHELLO=../othello -DFETCHCONTENT_SOURCE_DIR_MLFO=../mlfo

set(OTHELLO_GIT_TAG latest CACHE STRING "Branch, tag or commit of isaacdchu/othello to use, or 'latest'")
set(MLFO_GIT_TAG latest CACHE STRING "Branch, tag or commit of isaacdchu/mlfo to use, or 'latest'")

# fallbacks used when 'latest' is requested but the remote can't be queried (e.g. offline).
# these are also the oldest versions the bot works with: it needs othello's board
# symmetries and margin, and mlfo's channels-last conv2d (0.4.0)
set(_OTHELLO_FALLBACK_TAG v0.3.0)
set(_MLFO_FALLBACK_TAG v0.4.0)

# sets out_var to the highest v* tag on repo (version-sorted, so v0.10.0 > v0.9.0).
# leaves out_var unchanged if the lookup fails. requires git >= 2.18.
function(latest_release_tag repo out_var)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} ls-remote --tags --refs --sort=-v:refname ${repo} "v*"
        OUTPUT_VARIABLE refs
        RESULT_VARIABLE rc
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(rc EQUAL 0 AND refs MATCHES "refs/tags/([^\n]+)")
        set(${out_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    else()
        message(WARNING "Could not query latest tag of ${repo}; using ${${out_var}}")
    endif()
endfunction()

set(_othello_tag ${OTHELLO_GIT_TAG})
if(_othello_tag STREQUAL "latest")
    set(_othello_tag ${_OTHELLO_FALLBACK_TAG})
    latest_release_tag(https://github.com/isaacdchu/othello.git _othello_tag)
endif()

set(_mlfo_tag ${MLFO_GIT_TAG})
if(_mlfo_tag STREQUAL "latest")
    set(_mlfo_tag ${_MLFO_FALLBACK_TAG})
    latest_release_tag(https://github.com/isaacdchu/mlfo.git _mlfo_tag)
endif()

message(STATUS "Using othello ${_othello_tag}, mlfo ${_mlfo_tag}")

# SYSTEM keeps the libraries' headers out of this project's warnings
FetchContent_Declare(othello
    GIT_REPOSITORY https://github.com/isaacdchu/othello.git
    GIT_TAG ${_othello_tag}
    GIT_SHALLOW TRUE
    SYSTEM
)
FetchContent_Declare(mlfo
    GIT_REPOSITORY https://github.com/isaacdchu/mlfo.git
    GIT_TAG ${_mlfo_tag}
    GIT_SHALLOW TRUE
    SYSTEM
)
FetchContent_MakeAvailable(othello mlfo)
