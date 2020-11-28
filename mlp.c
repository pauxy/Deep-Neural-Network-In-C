/* mlp.c -- 
 * Author: Lim Chun Yu
 */

#include <time.h>
#include <stdlib.h>

#include "mlp.h"

BiasWeights_t initBiasWeights(int inputFields) {
    BiasWeights_t biasWeights;
    biasWeights.weights = (double*)malloc(inputFields * sizeof(double));
    biasWeights.bias = 0;

    srand(time(NULL));
    for (int cols = 0; cols < inputFields; cols++) {
        *(biasWeights.weights + cols) = (double)rand() / (double)RAND_MAX; /* Randomly assign
                                                                              weights and bias */
    }
    return biasWeights;
}

Layer_t genLayer(int nodes) {
    Layer_t layer;
    layer.numOfNodes = nodes;
    layer.nodes = (Node_t*)malloc(nodes * sizeof(Node_t));
    for (int i = 0; i < nodes; i++) {
        (layer.nodes + i)->biasWeights = initBiasWeights(nodes);
    }

    return layer;
}
