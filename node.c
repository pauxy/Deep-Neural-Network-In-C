/* node.c -- 
 * Author: Lim Chun Yu
 */

#include <time.h>
#include <stdlib.h>

#include "node.h"

double* initBiasWeights(int inputFields) {
    double* biasWeights = (double*)malloc((inputFields + 1) * sizeof(double)); // Allocate space for weights +1 bias
    *biasWeights = 0;

    srand(time(NULL));
    for (int cols = 1; cols < inputFields + 1; cols++) {
        *(biasWeights + cols) = (double)rand() / (double)RAND_MAX; // Randomly assign weights and bias
    }
    return biasWeights;
}
