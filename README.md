# Multi-Layered Perceptron
A multi-layered perceptron used to determine semen fertility outcomes using give dataset and machine
learning algorithms

## Compilation
```
$ make
clang -Ofast -std=gnu18 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -I ./include -c src/main.c -o obj/main.o
clang -Ofast -std=gnu18 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -I ./include -c src/dataparser.c -o obj/dataparser.o
clang -Ofast -std=gnu18 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -I ./include -c src/forwardprop.c -o obj/forwardprop.o
clang -Ofast -std=gnu18 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -I ./include -c src/backprop.c -o obj/backprop.o
clang -Ofast -std=gnu18 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -I ./include -c src/error.c -o obj/error.o
clang -Ofast -std=gnu18 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -I ./include -c src/mlp.c -o obj/mlp.o
clang -o mlperceptron obj/main.o obj/dataparser.o obj/forwardprop.o obj/backprop.o obj/error.o obj/mlp.o -Ofast -std=gnu18 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -I ./include -lm
$
```
## Execution
```
$ ./mlperceptron
```
