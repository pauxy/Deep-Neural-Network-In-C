/* backprop.c -- 
 * Author: Lim Chun Yu
 */

#include <math.h>
#include <stdlib.h>

#include "dataparser.h"

double meanAbsoluteValue(double** training, double* activatedVal, int val) {    //2c
    double total = 0.0;
    for (int rows = 0; rows < val; rows++) {
        total += fabs( *(activatedVal + rows) - training[rows][DATA_COLUMNS - 1]);
    }
    return total / val;
}


double* backwardsPropagation(double* biasWeights, double* activatedVal,
                             double** training, double* lr) {                   //2d
    double* newBiasWeights = (double*)malloc((ATTR_COLUMNS + 1) * sizeof(double));
    double biasTotal = 0.0;
    for (int cols = 0; cols < ATTR_COLUMNS; cols++) {
        double weightTotal = 0.0;
        for (int rows = 0; rows < TRAINING_MAX; rows++) {
            double ph = exp( *(lr + cols)) / pow(1.0 + exp( *(lr + cols)), 2.0);
            double ph1 = *(activatedVal + cols) - training[rows][DATA_COLUMNS - 1];
            weightTotal += (ph * ph1 * training[rows][cols]);
            if (cols == 0) {
                biasTotal += ph * ph1;
            }
        }
        *(newBiasWeights + 1 + cols) = *(biasWeights + 1 + cols) - (LEARNING_RATE * (weightTotal / TRAINING_MAX));
    }
    *newBiasWeights = *(biasWeights) - (LEARNING_RATE * (biasTotal / TRAINING_MAX));
    return newBiasWeights;
}
