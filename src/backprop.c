/* backprop.c -- Handles MAE and Backward Propogation
 * Author: Lim Chun Yu
 */

#include <math.h>
#include <stdlib.h>

#include "dataparser.h"
#include "mlp.h"

const double LEARNING_RATE = 0.05;

/**
 * deSigmoid(): Calculates the differentiated sigmoid
 * @muladd:     Value to be delta sigmoided
 *
 * Calculated differentiated sigmoid using formula
 * \frac{e^{z_i(t)}}{(1 + e^{z_i(t)})^2}
 *
 * Return: differentiated sigmoided value
 */
double deSigmoid(double muladd) {
    return exp(muladd) / pow(1.0 + exp(muladd), 2.0);
}


/**
 * deError():       Calculates the error
 * @activatedVal:   Predicted output
 * @expectedOutput: Expected output
 *
 * Calculates the differentiated square error
 * \hat{y} - d_i
 *
 * Return: calculated error
 */
double deError(double activatedVal, int expectedOutput) {
    return activatedVal - expectedOutput;
}

/**
 * backwardsPropagation(): Updates weights and biases for each iteration
 *
 * @input:                 2D array of input from previous layer
 * @output                 Expected output from dataset
 * @biasWeights:           Struct of bias and weights
 * @activatedVal:          Calculated output from previous node
 * @muladd:                Array of the calculated sum of weights, inputs and biases using formula
 * @batchSize:             Size of batch
 * @connections:           Number of connections perceptron will have
 *
 * @biasTotal:             Diffrentiation of the cost function with respect to the biases
 * @weightTotal:           Diffrentiation of the cost function with respect to the weights
 * @weightBiasUpdate:      Amount required to adjust both bias and weights using formula
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
BiasWeights_t backwardsPropagation(double** input, int* output, BiasWeights_t biasWeights,
                                   double* activatedVal, double* muladd,
                                   int batchSize, int connections) {
    double biasTotal = 0.0;
    for (int cols = 0; cols < connections; cols++) {
        double weightTotal = 0.0;
        for (int rows = 0; rows < batchSize; rows++) {
            double weightBiasUpdate = deSigmoid(muladd[cols]) *
                deError(activatedVal[cols], output[rows]); /* calculation of each val in
                                                              summation of bet */
            weightTotal += (weightBiasUpdate * input[rows][cols]); /* summation of all values for
                                                                      one element in wet */

            if (cols == 0) biasTotal += weightBiasUpdate; /* summation of values for bet formula */
        }

        *(biasWeights.weights + cols) -= (LEARNING_RATE *
                (weightTotal / batchSize)); /* storing of and calculation of wet values according to
                                               formula */
    }
    biasWeights.bias -= (LEARNING_RATE * (biasTotal / batchSize)); /* calculation of bet using
                                                                      formula */
    return biasWeights;
}
