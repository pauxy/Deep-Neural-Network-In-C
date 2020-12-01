# Multi-Layered Perceptron
A multi-layered perceptron used to determine semen fertility outcomes using give dataset and machine
learning algorithms

## Provided binaries
- 32-bit Linux System: `mlperceptron-linux_i386.elf`
- 64-bit Linux System: `mlperceptron-linux_x86-64.elf`
- M1 MacOS System: `mlperceptron-macos11_arm64e.mach-o`
- Intel MacOS System: `mlperceptron-macos11_x86-64.mach-o`
- 32-bit Windows System: `mlperceptron-windows10_i686.exe`
- 64-bit Windows System: `mlperceptron-windows10_x86-64.exe`

## Compilation
```
$ make
clang -fsanitize=signed-integer-overflow -fsanitize=undefined -Ofast -std=gnu18 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -I ./include -c src/main.c -o obj/main.o
clang -fsanitize=signed-integer-overflow -fsanitize=undefined -Ofast -std=gnu18 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -I ./include -c src/dataparser.c -o obj/dataparser.o
clang -fsanitize=signed-integer-overflow -fsanitize=undefined -Ofast -std=gnu18 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -I ./include -c src/forwardprop.c -o obj/forwardprop.o
clang -fsanitize=signed-integer-overflow -fsanitize=undefined -Ofast -std=gnu18 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -I ./include -c src/backprop.c -o obj/backprop.o
clang -fsanitize=signed-integer-overflow -fsanitize=undefined -Ofast -std=gnu18 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -I ./include -c src/error.c -o obj/error.o
clang -fsanitize=signed-integer-overflow -fsanitize=undefined -Ofast -std=gnu18 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -I ./include -c src/mlp.c -o obj/mlp.o
clang -o mlperceptron obj/main.o obj/dataparser.o obj/forwardprop.o obj/backprop.o obj/error.o obj/mlp.o -fsanitize=signed-integer-overflow -fsanitize=undefined -Ofast -std=gnu18 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -I ./include -lm
$
```
## Execution
```
$ mlperceptron
```
