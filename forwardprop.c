/* forwardprop.c -- Calculates linear regression and sigmoid function
 * Author: Lim Chun Yu
 */

#include <math.h>
#include <stdlib.h>

#include "forwardprop.h"
#include "mlp.h"

double* matmuladd(double**, BiasWeights_t, double*, int, int);
double sigmoid(double);

/**
 * forwardPropagation(): [TODO:description]
 * @input: [TODO:description]
 * @biasWeights: [TODO:description]
 * @muladd: [TODO:description]
 * @activatedVal: [TODO:description]
 * @batchSize: [TODO:description]
 * @connections: [TODO:description]
 *
 * [TODO:description]
 *
 * Return: [TODO:description]
 */
double* forwardPropagation(double** input, BiasWeights_t biasWeights, double* muladd,
                           double* activatedVal, int batchSize, int connections) {
    muladd = matmuladd(input, biasWeights, muladd, batchSize, connections);
    for(int rows = 0; rows < batchSize; rows++) {
        *(activatedVal + rows) = sigmoid(*(muladd + rows));
    }

    return activatedVal;
}

/**
 * matmuladd():  Calculates linear regression for each iteration
 *
 * @input:       2D array of input from previous layer
 * @biasWeights: Struct of bias and weights where the firest element is the bias and the
 * @muladd:      Array of the calculated sum of weights, inputs and biases using formula
 *               remaining elements are weights
 * @batchSize:   Size of batch
 * @connections: Number of connections perceptron will have
 *
 *
 * Part 2a
 * Calculates linear regression for each iteration based on the inputs, weights and biases where
 * inputs and weights and vectors and bias is a scalar using formula
 * \overrightarrow{\textbf{w}}^{t} \cdot \overrightarrow{\textbf{x}}_{i} + b_{i}^{t}
 *
 * Return: double* muladd
 */
double* matmuladd(double** input, BiasWeights_t biasWeights, double* muladd,
                  int batchSize, int connections) {
    for (int rows = 0; rows < batchSize; rows++) {
        *(muladd + rows) = 0; /* initialise to zero the var for the addition
                                    of columns in data */
        puts("hah");
        for (int cols = 0; cols < connections; cols++) {
            *(muladd + rows) += ( *(biasWeights.weights + cols) *
                    input[rows][cols]) + biasWeights.bias; /* adds to counter and appends to array
                                                              using lr fomula */
        }
    }
    return muladd;
}


/**
 * sigmoid():     Squashes input into double between 0-1
 *
 * @muladd:       Element of matmuladd()
 *
 *
 * Part 2b
 * Calculates the sigmoid activation function
 * \frac{1}{1+e^{-z_{i}(t)}}
 *
 * Return: activated value after sigmoid function
 */
double sigmoid(double muladd) {
    return 1.0 / (1.0 + exp(-muladd)); /* sigmoid formula as provided */
}
