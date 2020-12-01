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
 * initBiasWeights(): Assigns random values to weights and bias to 0
 * @connections: Number of connections weights required initialise
 *
 * Sets bias to 0 and weights to a random double from -1 to 1
 *
 * Return: BiasWeights_t type containing both weights and bias
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
 * genLayer():     Generates layer with nodes
 * @nodesPerLayer: Number of nodes in layer
 * @conn:          Connections going to each node from previous layer
 * @prev:          Pointer to the previous layer
 *
 * Sets connections, number of nodes, pointer to the previous layer in Layer_t
 * Assigns memory and values to variables in nodes in layer
 *
 * Return: Data of one layer of nodes
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
 * trainLayer():             Trains each layer with forward prop
 * @layer:                   Layer to train
 * @input:                   The out put from the previous layer is the input for the current layer
 * @layerActivatedValOutput: The untransposed 2D matrix of sigmoided values from forwardprop
 *
 * Loops through nodes and calculates using the forward propagation algorithm
 * Transposes output matrix to batch size x number of nodes
 *
 * Return: Transposed activation value from forwawrd propagation
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
 * trainNetwork():   Trains the entire neural network
 * @numHiddenLayers: List of hidden layers
 * @nodesPerLayer:   An array of nodes for each layer
 * @trainTest:       Dataset values
 * @minMae:          Minimum MAE to hit before training stops
 * @graph:           File pointer to plot values on graph
 *
 * Loops through each layer, and for each layer loop through each node and conduct forward and
 * backward propagation
 *
 * Return: Pointer to the output node
 */
Layer_t* trainNetwork(int numHiddenLayers, int* nodesPerLayer, InputOutput_t* trainTest,
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

    return layers;
}


/**
 * predict():    Uses calculates weights and biases against test data
 * @data:        Testing dataset
 * @biasWeights: Bias and Weights after training
 *
 * Return:       Array of prediction
 */
int* predict(InputOutput_t data, BiasWeights_t biasWeights) {
    double* result = (double*)malloc(TESTING_MAX * sizeof(double));
    result = forwardPropagation(data.input, biasWeights, result,
                                result, TESTING_MAX, ATTR_COLUMNS);

    int* prediction = (int*)malloc(TESTING_MAX * sizeof(int));
    for (int i = 0; i < TESTING_MAX; i++) {
        if ( *(result + i) > 0.5)
            *(prediction + i) = 1;
        else
            *(prediction + i) = 0;
    }
    free(result);
    return prediction;
}


/**
 * testNetwork(): Test how effective neural network is
 * @network:      All the information of the neural network
 * @trainTest:    Dataset values
 * @numLayers:    Number of layers in neural network
 *
 * Checks for MMSE, and displays confusion matrix
 */
void testNetwork(Layer_t* network, InputOutput_t* trainTest, int numLayers) {
    double* finalAV = (network + numLayers - 1)->nodes->activatedVal;
    BiasWeights_t finalBiasWeights = (network + numLayers - 1)->nodes->biasWeights;
    printf("\n-After Training-\nMMSE Training: %f\n",
            minMeanSquareError(trainTest[0].output, finalAV, TRAINING_MAX));
    printf("MMSE Testing: %f\n\n",
            minMeanSquareError(trainTest[1].output, finalAV, TESTING_MAX));
    int* prediction = predict(trainTest[1], finalBiasWeights);
    int* cm = confusionMatrix(trainTest[1].output, prediction, TESTING_MAX);

    puts("-Confusion Matrix-");
    printf("True Positive: %d\n", cm[0]);
    printf("True Negative: %d\n", cm[1]);
    printf("False Positive: %d\n", cm[2]);
    printf("False Negative: %d\n", cm[3]);

    free(prediction);
    free(cm);
}
