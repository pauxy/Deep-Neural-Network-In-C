/* backprop.c -- Handles MAE and Backward Propogation
 * Author: Lim Chun Yu
 */

#include <math.h>
#include <stdlib.h>

#include "dataparser.h"
#include "mlp.h"

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
        total += fabs( *(activatedVal + rows) -
                data[rows][DATA_COLUMNS - 1]); /* formula provided for MAE, calcadds everyvalue to be divided in line 32 */
    }
    return total / val;
}


/**
 * backwardsPropagation(): Updates weights and biases for each iteration
 *
 * @biasWeights:      Struct of bias and weights where bias is the first element and the rest are weights
 * @activatedVal:     Activated values passed from the sigmoid function
 * @data:             Attributes from dataset
 * @lr:               Array of the calculated sum of weights, inputs and biases using formula
 *
 * @biasTotal:        Diffrentiation of the cost function with respect to the biases
 * @weightTotal:      Diffrentiation of the cost function with respect to the weights
 * @weightBiasUpdate: Amount required to adjust both bias and weights using formula
 *
 * Part 2d
 * Backward propogation of both weights and bias for each iteration using formulas using the current
 * weights subtracted by the gradient of the cost function with respect to the weights multiplied by
 * the learning rate
 * \overrightarrow{\textbf{w}}^{t} - \eta \triangledown_{w}E^{t}
 * b_{i}^{t} - \eta \triangledown_{b}E^{t}
 *
 * Return: BiasWeights_t biasWeights
 */
BiasWeights_t backwardsPropagation(double** data, BiasWeights_t biasWeights,
                                   double* activatedVal, double* lr) {
    double biasTotal = 0.0;
    for (int cols = 0; cols < ATTR_COLUMNS; cols++) {
        double weightTotal = 0.0;
        for (int rows = 0; rows < TRAINING_MAX; rows++) {
            double weightBiasUpdate = exp( *(lr + cols)) / pow(1.0 + exp( *(lr + cols)), 2.0) *
                ( *(activatedVal + cols) - data[rows][DATA_COLUMNS - 1]); /* calculation of each val in summation of bet */
            weightTotal += (weightBiasUpdate * data[rows][cols]); /* summation of all values for one element in wet */

            if (cols == 0) biasTotal += weightBiasUpdate; /* summation of values for bet formula */
        }
        *(biasWeights.weights + cols) = *(biasWeights.weights + cols) -
            (LEARNING_RATE * (weightTotal / TRAINING_MAX)); /* storing of and calculation of wet values according to formula */
    }
    biasWeights.bias = biasWeights.bias - (LEARNING_RATE * (biasTotal / TRAINING_MAX)); /* calculation of bet using formula */
    return biasWeights;
}
