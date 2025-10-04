CXX = g++-15
CXXFLAGS = -std=c++23 -Wall -Wextra -pedantic
DEPFLAGS = -MMD -MP
SRC_DIR = src
BUILD_DIR = build

MAIN_SRC = $(SRC_DIR)/main.cpp
TRAIN_SRC = $(SRC_DIR)/train.cpp
EVALUATE_SRC = $(SRC_DIR)/evaluate.cpp

# All .cpp files except main.cpp and train.cpp
SHARED_SOURCES = $(filter-out $(MAIN_SRC) $(TRAIN_SRC), $(shell find $(SRC_DIR) -name '*.cpp'))
SHARED_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SHARED_SOURCES))

MAIN_OBJ = $(BUILD_DIR)/main.o
TRAIN_OBJ = $(BUILD_DIR)/train.o
EVALUATE_OBJ = $(BUILD_DIR)/evaluate.o

OBJECTS = $(SHARED_OBJECTS) $(MAIN_OBJ) $(TRAIN_OBJ) $(EVALUATE_OBJ)
DEPS = $(patsubst $(BUILD_DIR)/%.o,$(BUILD_DIR)/%.d,$(OBJECTS))

MAIN_EXE = othello_bot_main
TRAIN_EXE = othello_bot_train
EVALUATE_EXE = othello_bot_evaluate

all: $(MAIN_EXE) $(TRAIN_EXE) $(EVALUATE_EXE)

$(MAIN_EXE): $(SHARED_OBJECTS) $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TRAIN_EXE): $(SHARED_OBJECTS) $(TRAIN_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(EVALUATE_EXE): $(SHARED_OBJECTS) $(EVALUATE_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(MAIN_EXE) $(TRAIN_EXE) $(EVALUATE_EXE)

run: $(MAIN_EXE)
	./$(MAIN_EXE)

train: $(TRAIN_EXE)
	./$(TRAIN_EXE)

evaluate: $(EVALUATE_EXE)
	./$(EVALUATE_EXE)

# Include dependency files (ignore missing ones)
-include $(DEPS)

.PHONY: all clean run train evaluate