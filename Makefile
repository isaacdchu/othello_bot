CXX = g++-15
# CXXFLAGS = -fopenmp -O3 -std=c++23 -Wall -Wextra -pedantic
CXXFLAGS = -std=c++23 -Wall -Wextra -pedantic
SRC_DIR = src
BUILD_DIR = build
TARGET = othello_bot


# Find all .cpp files in src/ and subdirectories
SOURCES = $(shell find $(SRC_DIR) -name '*.cpp')
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Pattern rule to build .o files from .cpp files in any subdirectory
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

run:
	@$(MAKE) all
	./$(TARGET)
.PHONY: all clean run