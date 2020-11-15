/* backprop.c -- Handles MAE and Backward Propogation
 * Author: Lim Chun Yu
 */

#include <math.h>
#include <stdlib.h>

#include "dataparser.h"

/**
 * meanAbsoluteValue(): Calculates MAE for each iteration
 *
 * @data:               Attributes from dataset
 * @activatedVal:       Activated values passed from the sigmoid function
 * @val:                Size of data in rows
 *
 * @total:              Absolute sum of activated values minus true label of data sample
 *
 * Part 2c
 * Calculates the mean absolute value for each iteration where the activated values is subtracted by
 * the true label of the give dataset, and averaged using formula
 * \frac{\sum_{i=1}^{I} |\hat{y_{l}}^{t} - d_{i}|}{I}
 *
 * Return: Averaged sum of activated values subtracted by true label
 */
double meanAbsoluteValue(double** data, double* activatedVal, int val) {
    double total = 0.0;
    for (int rows = 0; rows < val; rows++) {
        total += fabs( *(activatedVal + rows) - /* TODO:description */
                data[rows][DATA_COLUMNS - 1]);
    }
    return total / val;
}


/**
 * backwardsPropagation():
 *
 * @biasWeights:
 * @activatedVal:
 * @data:
 * @lr:
 *
 * @newBiasWeights:
 * @biasTotal:
 * @weightTotal:
 * @weightBiasUpdate:
 *
 * Part 2d
 *
 * Return:
 */
double* backwardsPropagation(double* biasWeights, double* activatedVal,
                             double** data, double* lr) {
    double* newBiasWeights = (double*)malloc((ATTR_COLUMNS + 1) * sizeof(double));
    double biasTotal = 0.0;
    for (int cols = 0; cols < ATTR_COLUMNS; cols++) {
        double weightTotal = 0.0;
        for (int rows = 0; rows < TRAINING_MAX; rows++) {
            double weightBiasUpdate = exp( *(lr + cols)) / pow(1.0 + exp( *(lr + cols)), 2.0) *
                (*(activatedVal + cols) - data[rows][DATA_COLUMNS - 1]);
            weightTotal += (weightBiasUpdate * data[rows][cols]);

            if (cols == 0) biasTotal += weightBiasUpdate;
        }
        *(newBiasWeights + 1 + cols) = *(biasWeights + 1 + cols) - (LEARNING_RATE * (weightTotal / TRAINING_MAX));
    }
    *newBiasWeights = *(biasWeights) - (LEARNING_RATE * (biasTotal / TRAINING_MAX));
    return newBiasWeights;
}
