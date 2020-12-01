/* mlp.c -- Where all the training magic of neural network happens
 * Author: Lim Chun Yu
 */

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "dataparser.h"
#include "forwardprop.h"
#include "backprop.h"
#include "error.h"
#include "mlp.h"

/**
 * initBiasWeights(): [TODO:description]
 * @connections: [TODO:description]
 *
 * [TODO:description]
 *
 * Return: [TODO:description]
 */
BiasWeights_t initBiasWeights(int connections) {
    BiasWeights_t biasWeights;
    biasWeights.weights = (double*)malloc(connections * sizeof(double));
    biasWeights.bias = 0;


    for (int cols = 0; cols < connections; cols++) {
        *(biasWeights.weights + cols) = (double)rand() /
            (double)RAND_MAX * 2.0 - 1.0; /* Randomly assign weights and bias */
    }
    return biasWeights;
}


/**
 * genLayer(): [TODO:description]
 * @nodesPerLayer: [TODO:description]
 * @conn: [TODO:description]
 * @prev: [TODO:description]
 *
 * [TODO:description]
 *
 * Return: [TODO:description]
 */
Layer_t genLayer(int nodesPerLayer, int conn, Layer_t* prev) {
    Layer_t layer;
    layer.numOfNodes = nodesPerLayer;
    layer.nodes = (Node_t*)malloc(nodesPerLayer * sizeof(Node_t));
    layer.prev = prev;

    for (int i = 0; i < nodesPerLayer; i++) {
        (layer.nodes + i)->connections = conn;
        (layer.nodes + i)->biasWeights = initBiasWeights((layer.nodes + i)->connections);
        (layer.nodes + i)->muladd = (double*)malloc(TRAINING_MAX * sizeof(double));
        (layer.nodes + i)->activatedVal = (double*)malloc(TRAINING_MAX * sizeof(double));
    }

    return layer;
}


/**
 * {name}(): [TODO:description]
 *
 * [TODO:description]
 *
 * Return: [TODO:description]
 */
double** trainLayer(Layer_t layer, double** input, double** layerActivatedValOutput) {
    for (int i = 0; i < layer.numOfNodes; i++) {
        (layer.nodes + i)->activatedVal =  forwardPropagation(input, (layer.nodes + i)->biasWeights,
                           (layer.nodes + i)->muladd, (layer.nodes + i)->activatedVal,
                           TRAINING_MAX, (layer.nodes + i)->connections);
        *(layerActivatedValOutput + i) = (layer.nodes + i)->activatedVal;
    }

    double** transposedAV = (double**)malloc(TRAINING_MAX * sizeof(double*));
    for (int row = 0; row < TRAINING_MAX; row++) {
        transposedAV[row] = (double*)malloc(layer.numOfNodes * sizeof(double));
        for (int col = 0; col < layer.numOfNodes; col++) {
            transposedAV[row][col] = layerActivatedValOutput[col][row];
        }
    }

    return transposedAV;
}


/**
 * {name}(): [TODO:description]
 *
 * [TODO:description]
 *
 * Return: [TODO:description]
 */
Node_t* trainNetwork(int numHiddenLayers, int* nodesPerLayer, InputOutput_t* trainTest,
                     double minMae, FILE* graph) {
    srand(time(NULL));

    double MAE_VAL;
    int totalLayers = numHiddenLayers + 1;
    InputOutput_t trainingData = *trainTest;
    InputOutput_t testingData = *(trainTest + 1);

    /* Allocate memory for layers */
    Layer_t* layers = (Layer_t*)malloc(totalLayers * sizeof(Layer_t));

    int* connections = (int*)malloc(totalLayers * sizeof(int));
    *connections = ATTR_COLUMNS;

    int* genNodes = (int*)malloc(totalLayers * sizeof(int));

    for (int i = 0; i < totalLayers; i++) {
        // Generate int* of needed nodes and connections
        if (i != numHiddenLayers) {
            *(connections + i + 1) = *(nodesPerLayer + i); // Append ATTR_COLUMNS to start of nodes pointer
            *(genNodes + i) = *(nodesPerLayer + i);
        } else {
            *(genNodes + i) = 1;
        }

        Layer_t* prev = i == 0 ? NULL : (layers + i - 1);

        *(layers + i) = genLayer(genNodes[i], connections[i], prev);
        (layers + i)->layerOutput = (double**)malloc(connections[i] * sizeof(double*));
    }

    Layer_t* outputLayer = layers + numHiddenLayers;

    int t = 0;
    do {
        if (t++ == 0) {
            puts("-Before Training-");
            printf("MMSE Training: %f\n",
                    minMeanSquareError(trainingData.output, outputLayer->nodes->activatedVal,
                                       TRAINING_MAX));
            printf("MMSE Testing: %f\n",
                    minMeanSquareError(testingData.output, outputLayer->nodes->activatedVal,
                                       TESTING_MAX));
        }

        for (int i = 0; i < totalLayers; i++) {
            if (i == 0) layers->layerOutput = trainLayer(*layers, trainingData.input,
                                                         layers->layerOutput);
            else (layers + i)->layerOutput = trainLayer(*(layers + i), (layers + i)->prev->layerOutput ,
                                                        (layers + i)->layerOutput);
        }


        MAE_VAL = meanAbsoluteValue(trainingData.output, outputLayer->nodes->activatedVal,
                                    TRAINING_MAX);
        fprintf(graph, "%i %lf \n", t, MAE_VAL);

        if (MAE_VAL > minMae) {
            for (int i = 0; i < totalLayers; i++) {
                Layer_t* currentLayer = layers + numHiddenLayers - i;
                for (int j = 0; j < currentLayer->numOfNodes; j++) {
                    double** av = trainingData.input;
                    if (currentLayer->prev != NULL) av = currentLayer->prev->layerOutput;

                    Node_t* currLayerNode = currentLayer->nodes + j;
                    currLayerNode->biasWeights =
                        backwardsPropagation(av, trainingData.output, currLayerNode->biasWeights,
                                             outputLayer->nodes->activatedVal, currLayerNode->muladd,
                                             TRAINING_MAX, currLayerNode->connections);
                }
            }
        }
    } while (MAE_VAL > minMae);

    return outputLayer->nodes;
}
