PROGRAM = mlperceptron

CC = clang
OBJ_DIR = ./obj
SRC_DIR = ./src
DEP_DIR = ./include
CFLAGS = -fsanitize=signed-integer-overflow -fsanitize=undefined -Ofast -std=gnu18 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable
DEPS = $(addprefix $(DEP_DIR)/, dataparser.h forwardprop.h backprop.h error.h mlp.h)
OBJECTS = $(addprefix $(OBJ_DIR)/, main.o dataparser.o forwardprop.o backprop.o error.o mlp.o)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(DEPS)
	$(CC) $(CFLAGS) -I $(DEP_DIR) -c $< -o $@

$(PROGRAM): $(OBJECTS)
	$(CC) -o $@ $^ $(CFLAGS) -I $(DEP_DIR) -lm

clean:
	rm -f $(OBJECTS)
	rm -f $(PROGRAM)
