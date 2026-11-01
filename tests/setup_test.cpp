#include <gtest/gtest.h>

#include <mlfo/mlfo.hpp>
#include <othello/othello.hpp>

// checks that both libraries are found and compile together
TEST(Setup, LibrariesLink) {
    mlfo::initializer::Generator generator(42);
    othello::board::Board board;
    SUCCEED();
}
