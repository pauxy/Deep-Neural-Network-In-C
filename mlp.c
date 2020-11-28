/* mlp.c -- 
 * Author: Lim Chun Yu
 */

#include <time.h>
#include <stdlib.h>

#include "dataparser.h"
#include "forwardprop.h"
#include "backprop.h"
#include "error.h"
#include "mlp.h"

BiasWeights_t initBiasWeights(int inputFields) {
    BiasWeights_t biasWeights;
    biasWeights.weights = (double*)malloc(inputFields * sizeof(double));
    biasWeights.bias = 0;

    srand(time(NULL));
    for (int cols = 0; cols < inputFields; cols++) {
        *(biasWeights.weights + cols) = (double)rand() /
            (double)RAND_MAX * 2.0 - 1.0; /* Randomly assign weights and bias */
    }
    return biasWeights;
}

Layer_t genLayer(int nodes, int conn, double reqMae,InputOutput_t trainTest) {
    Layer_t layer;
    layer.numOfNodes = nodes;
    layer.nodes = (Node_t*)malloc(nodes * sizeof(Node_t));
    for (int i = 0; i < nodes; i++) {
        (layer.nodes + i)->connections = conn;
        (layer.nodes + i)->biasWeights = initBiasWeights((layer.nodes + i)->connections);
        (layer.nodes + i)->muladd = (double*)malloc(TRAINING_MAX * sizeof(double));
        (layer.nodes + i)->activatedVal = (double*)malloc(TRAINING_MAX * sizeof(double));
    }
    trainNodes(layer.nodes, nodes, reqMae, trainTest);
    return layer;
}



void trainNodes(Node_t* nodes, int number, double reqMae,InputOutput_t trainTest){
    for (int i = 0; i < number; i++){
        double MAE_VAL=0;
        do {
            Node_t* node = nodes + i;
            node->activatedVal = forwardPropagation(trainTest.input, node->biasWeights,
                                                    node->muladd, node->activatedVal,
                                                    TRAINING_MAX, node->connections);
            MAE_VAL = meanAbsoluteValue(trainTest.output, node->activatedVal,
                                        TRAINING_MAX);
            if (MAE_VAL > 0.25) {
                node->biasWeights = backwardsPropagation(trainTest.input, trainTest.output,
                                                        node->biasWeights, node->activatedVal,
                                                        node->muladd, TRAINING_MAX, node->connections);
            }

        } while (MAE_VAL > reqMae);
    }
}
