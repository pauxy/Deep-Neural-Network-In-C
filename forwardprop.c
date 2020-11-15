/* forwardprop.c -- Calculates linear regression and sigmoid function
 * Author: Lim Chun Yu
 */

#include <math.h>
#include <stdlib.h>

#include "dataparser.h"
#include "forwardprop.h"

/**
 * linearRegression(): Calculates linear regression for each iteration
 *
 * @data:              2D array of dataset where each columns are attributes
 * @biasWeights:       Array of bias and weights where the firest element is the bias and the
 *                     remaining elements are weights
 * @val:               Size of training data in rows
 * @lr:                Array of the calculated sum of weights, inputs and biases using formula
 *
 * Part 2a
 * Calculates linear regression for each iteration based on the inputs, weights and biases where
 * inputs and weights and vectors and bias is a scalar using formula
 * \overrightarrow{\textbf{w}}^{t} \cdot \overrightarrow{\textbf{x}}_{i} + b_{i}^{t}
 *
 * Return: double* lr
 */
double* linearRegression(double** data, double* biasWeights, int val){
    double* lr = (double*)malloc(val * sizeof(double));
    for (int rows = 0; rows < val; rows++) {
        *(lr + rows) = 0;                                /* TODO:description */
        for (int cols = 0; cols < ATTR_COLUMNS; cols++) {
            // printf("%f\n", *(lr + rows));
            *(lr + rows) += (*(biasWeights + 1 + cols) * /* TODO:description */
                    data[rows][cols]) + *biasWeights;
        }
    }
    return lr;
}


/**
 * sigmoid():     Squashes input into double between 0-1
 *
 * @lr:           Array of calculated sum from linearRegression()
 * @val:          Size of training data in rows
 *
 * @activatedVal: Array of lr after running sigmoid function
 *
 * Part 2b
 * Calculates the sigmoid activation function from the array output from linearRegression() for each
 * iteration using formula
 * \frac{1}{1+e^{-z_{i}(t)}}
 *
 * Return: double* activatedVal
 */
double* sigmoid(double* lr, int val) {
    double* activatedVal = (double*)malloc(val * sizeof(double));
    for(int rows = 0; rows < val; rows++){
        *(activatedVal + rows) = 1.0 / (1.0 + exp(- *(lr + rows))); /* TODO:description */
    }
    return activatedVal;
}
