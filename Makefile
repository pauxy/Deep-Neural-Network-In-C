CC = clang
CFLAGS = -fsanitize=signed-integer-overflow -fsanitize=undefined -O2 -std=c18 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable
DEPS = dataparser.h forwardprop.h
OBJ = main.o dataparser.o forwardprop.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

perceptron: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm perceptron *.o graph.temp
