/* mlp.c -- 
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


Layer_t genLayer(int nodes, int conn, Layer_t* next, Layer_t* prev) {
    Layer_t layer;
    layer.numOfNodes = nodes;
    layer.nodes = (Node_t*)malloc(nodes * sizeof(Node_t));
    layer.next = next;
    layer.prev = prev;

    for (int i = 0; i < nodes; i++) {
        (layer.nodes + i)->connections = conn;
        (layer.nodes + i)->biasWeights = initBiasWeights((layer.nodes + i)->connections);
        (layer.nodes + i)->muladd = (double*)malloc(TRAINING_MAX * sizeof(double));
        (layer.nodes + i)->activatedVal = (double*)malloc(TRAINING_MAX * sizeof(double));
    }

    return layer;
}


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


Node_t* trainNetwork(int numOfHiddenLayers, int* nodes, InputOutput_t trainingData, int minMae, FILE* graph) {
    double MAE_VAL;
    int totalLayers = numOfHiddenLayers + 1;

    /* Allocate memory for layers */
    Layer_t* layers = (Layer_t*)malloc(totalLayers * sizeof(Layer_t));

    int* connections = (int*)malloc(totalLayers * sizeof(int));
    *connections = ATTR_COLUMNS;

    int* genNodes = (int*)malloc(totalLayers * sizeof(int));

    for (int i = 0; i < totalLayers; i++) {
        // Generate int* of needed nodes and connections
        if (i != numOfHiddenLayers) {
            *(connections + i + 1) = *(nodes + i); // Append ATTR_COLUMNS to start of nodes pointer
            *(genNodes + i) = *(nodes + i);
        } else {
            *(genNodes + i) = 1;
        }


        Layer_t* next = i == numOfHiddenLayers ? NULL : (layers + i + 1);
        Layer_t* prev = i == 0 ? NULL : (layers + i - 1);

        *(layers + i) = genLayer(genNodes[i], connections[i], next, prev);
        (layers + i)->layerOutput = (double**)malloc(connections[i] * sizeof(double*));
    }

    Layer_t outputLayer = *(layers + numOfHiddenLayers);

    int t = 0;

    do {
        // Calculate before MMSE
        if (t++ == 0) {
            printf("-Before Training-\nMMSE Training: %f\n",
                    minMeanSquareError(trainingData.output, outputLayer.nodes->activatedVal,
                                       TRAINING_MAX));
            printf("MMSE Testing: %f\n",
                    minMeanSquareError(trainingData.output, outputLayer.nodes->activatedVal, // TODO:testingData.output
                                       TESTING_MAX));
        }

        for (int i = 0; i < totalLayers; i++) {
                if (i == 0) layers->layerOutput = trainLayer(*layers, trainingData.input,
                                                             layers->layerOutput);
                else (layers + i)->layerOutput = trainLayer(*(layers + i), (layers + i)->prev->layerOutput ,
                                                            (layers + i)->layerOutput);
        }


        MAE_VAL = meanAbsoluteValue(trainingData.output, outputLayer.nodes->activatedVal,
                                    TRAINING_MAX);
        fprintf(graph, "%i %lf \n", t, MAE_VAL);

        if (MAE_VAL > minMae) {
            for (int i = 0; i < numOfHiddenLayers + 1; i++) {
                Layer_t* currentLayer = layers + numOfHiddenLayers - i;
                for (int j = 0; j < currentLayer->numOfNodes; j++) {

                    double** av = trainingData.input;
                    if (currentLayer->prev != NULL)
                        av = currentLayer->prev->layerOutput;


                    Node_t* currLayerNode = currentLayer->nodes + j;
                    currLayerNode->biasWeights =
                        backwardsPropagation(av, trainingData.output,
                                             currLayerNode->biasWeights, outputLayer.nodes->activatedVal,
                                             currLayerNode->muladd, TRAINING_MAX, currLayerNode->connections);
                }
            }
        }
        /* printf("%lf\n", MAE_VAL); */
    } while (MAE_VAL > minMae);
    return outputLayer.nodes;
}
