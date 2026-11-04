find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
# scikit-build-core puts the nanobind from the build requirements on CMAKE_PREFIX_PATH;
# outside of it, ask the interpreter where nanobind is installed
if(NOT nanobind_ROOT)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
        OUTPUT_VARIABLE nanobind_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()
find_package(nanobind CONFIG REQUIRED)

# importable as othello_gui._othello_bot
nanobind_add_module(_othello_bot NB_STATIC bindings.cpp)
target_link_libraries(_othello_bot PRIVATE othello_bot othello_bot_warnings)

# python doesn't search PATH for an extension's DLLs, so the gcc runtime
# (libstdc++, libgcc, winpthread) has to be linked into the module itself
if(MINGW)
    target_link_options(_othello_bot PRIVATE -static -static-libgcc -static-libstdc++)
endif()

install(TARGETS _othello_bot LIBRARY DESTINATION othello_gui RUNTIME DESTINATION othello_gui)
