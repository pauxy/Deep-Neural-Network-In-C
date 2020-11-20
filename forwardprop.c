/* forwardprop.c -- Calculates linear regression and sigmoid function
 * Author: Lim Chun Yu
 */

#include <math.h>
#include <stdlib.h>

#include "dataparser.h"
#include "forwardprop.h"
#include "mlp.h"

/**
 * linearRegression(): Calculates linear regression for each iteration
 *
 * @input:             2D array of input from previous layer
 * @biasWeights:       Struct of bias and weights where the firest element is the bias and the
 *                     remaining elements are weights
 * @batchSize:         Size of batch
 * @connections:       Number of connections perceptron will have
 *
 * @lr:                Array of the calculated sum of weights, inputs and biases using formula
 *
 * Part 2a
 * Calculates linear regression for each iteration based on the inputs, weights and biases where
 * inputs and weights and vectors and bias is a scalar using formula
 * \overrightarrow{\textbf{w}}^{t} \cdot \overrightarrow{\textbf{x}}_{i} + b_{i}^{t}
 *
 * Return: double* lr
 */
double* linearRegression(double** input, BiasWeights_t biasWeights,
                         int batchSize, int connections) {
    double* lr = (double*)malloc(batchSize * sizeof(double));
    for (int rows = 0; rows < batchSize; rows++) {
        *(lr + rows) = 0;                                /* initialise to zero the var for the addition of columns in data */
        for (int cols = 0; cols < connections; cols++) {
            // printf("%f\n", *(lr + rows));
            *(lr + rows) += ( *(biasWeights.weights + cols) *
                    input[rows][cols]) + biasWeights.bias; /* adds to counter and appends to array using lr fomula */
        }
    }
    return lr;
}


/**
 * sigmoid():     Squashes input into double between 0-1
 *
 * @lr:           Array of calculated sum from linearRegression()
 * @batchSize:    Size of batch
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
double* sigmoid(double* lr, int batchSize) {
    double* activatedVal = (double*)malloc(batchSize * sizeof(double));
    for(int rows = 0; rows < batchSize; rows++){
        *(activatedVal + rows) = 1.0 / (1.0 + exp(- *(lr + rows))); /* sigmoid formula as provided */
    }
    return activatedVal;
}
