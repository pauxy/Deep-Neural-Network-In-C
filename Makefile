PROGRAM_NAME = perceptron

CC = clang
CFLAGS = -fsanitize=signed-integer-overflow -fsanitize=undefined -Ofast -std=c18 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable
DEPS = dataparser.h forwardprop.h backprop.h loss.h mlp.h
OBJ = main.o dataparser.o forwardprop.o backprop.o loss.o mlp.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(PROGRAM_NAME): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm $(PROGRAM_NAME) *.o
