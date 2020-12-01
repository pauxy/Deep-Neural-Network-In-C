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
    /* Initialises layer */
    Layer_t layer;
    layer.numOfNodes = nodesPerLayer;
    layer.nodes = (Node_t*)malloc(nodesPerLayer * sizeof(Node_t));
    layer.prev = prev;

    /* Initialises node */
    for (int i = 0; i < nodesPerLayer; i++) {
        (layer.nodes + i)->connections = conn;
        (layer.nodes + i)->biasWeights = initBiasWeights((layer.nodes + i)->connections);
        (layer.nodes + i)->muladd = (double*)malloc(TRAINING_MAX * sizeof(double));
        (layer.nodes + i)->activatedVal = (double*)malloc(TRAINING_MAX * sizeof(double));
    }

    return layer;
}


/**
 * trainTestLayer():         Trains or tests each layer with forward prop
 * @layer:                   Layer to train or test
 * @input:                   The out put from the previous layer is the input for the current layer
 * @layerActivatedValOutput: The untransposed 2D matrix of sigmoided values from forwardprop
 *
 * Loops through nodes and calculates using the forward propagation algorithm
 * Transposes output matrix to connections x number of nodes
 *
 * Return: Transposed activation value from forward propagation
 */
double** trainTestLayer(Layer_t layer, double** input,
                        double** layerActivatedValOutput, int batchSize) {
    for (int i = 0; i < layer.numOfNodes; i++) { /* Loops through nodes in layer */
        /* Runs forward propagation algorithm */
        (layer.nodes + i)->activatedVal =  forwardPropagation(input, (layer.nodes + i)->biasWeights,
                           (layer.nodes + i)->muladd, (layer.nodes + i)->activatedVal,
                           batchSize, (layer.nodes + i)->connections);

        *(layerActivatedValOutput + i) = (layer.nodes + i)->activatedVal; /* Stores activated values
                                                                             in matrix */
    }

    /* Transposes matrix from numOfNodes x conections to connections x numOfNodes */
    double** transposedAV = (double**)malloc(batchSize * sizeof(double*));
    for (int row = 0; row < batchSize; row++) {
        transposedAV[row] = (double*)malloc(layer.numOfNodes * sizeof(double));
        for (int col = 0; col < layer.numOfNodes; col++) {
            transposedAV[row][col] = layerActivatedValOutput[col][row];
        }
    }

    return transposedAV; /* Tranposed matrix of activated values */
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
    Layer_t* network = (Layer_t*)malloc(totalLayers * sizeof(Layer_t));

    /* Formatting number of layers and its nodes for generation*/
    int* connections = (int*)malloc(totalLayers * sizeof(int));
    *connections = ATTR_COLUMNS;
    int* genNodes = (int*)malloc(totalLayers * sizeof(int));
    for (int i = 0; i < totalLayers; i++) {
        /* Generate int* of needed nodes and connections */
        if (i != numHiddenLayers) {
            *(connections + i + 1) = *(nodesPerLayer + i); /* Append ATTR_COLUMNS to start
                                                              of nodes pointer */
            *(genNodes + i) = *(nodesPerLayer + i);
        } else {
            *(genNodes + i) = 1;
        }

        Layer_t* prev = i == 0 ? NULL : (network + i - 1); /* Stores previous layer in
                                                             current layer
                                                             NULL if input layer is the
                                                             previous layer */

        /* Generates layers */
        *(network + i) = genLayer(genNodes[i], connections[i], prev);

        /* Allocates memory of activated values from each layer */
        (network + i)->layerOutput = (double**)malloc(connections[i] * sizeof(double*));
    }

    free(connections);
    free(genNodes);

    Layer_t* outputLayer = network + numHiddenLayers; /* Gets pointer of output layer */

    int t = 0;
    do {
        /* Loops through layers and calculates forward prop */
        for (int i = 0; i < totalLayers; i++) {
            if (i == 0) network->layerOutput = trainTestLayer(*network, trainingData.input,
                                                         network->layerOutput, TRAINING_MAX);
            else (network + i)->layerOutput = trainTestLayer(*(network + i), (network + i)->prev->layerOutput ,
                                                        (network + i)->layerOutput, TRAINING_MAX);
        }

        if (t++ == 0) {
            /* Gets MMSE before training */
            puts("-Before Training-");
            printf("MMSE Training: %f\n",
                    minMeanSquareError(trainingData.output, outputLayer->nodes->activatedVal,
                                       TRAINING_MAX));
            printf("MMSE Testing: %f\n",
                    minMeanSquareError(testingData.output, outputLayer->nodes->activatedVal,
                                       TESTING_MAX));
        }

        MAE_VAL = meanAbsoluteValue(trainingData.output, outputLayer->nodes->activatedVal,
                                    TRAINING_MAX); /* Calculates MAE value */
        fprintf(graph, "%i %lf \n", t, MAE_VAL); /* Plot MAE by epoch graph */

        if (MAE_VAL > minMae) { /* Checks if MAE is miraculously meets the required MAE value */
            for (int i = 0; i < totalLayers; i++) { /* Loops through layers */
                Layer_t* currentLayer = network + numHiddenLayers - i; /* Get pointer of
                                                                         current layer */
                for (int j = 0; j < currentLayer->numOfNodes; j++) { /* Loops through each node */
                    /* Set activation value to previous layer output */
                    double** av = trainingData.input;
                    if (currentLayer->prev != NULL) av = currentLayer->prev->layerOutput;

                    Node_t* currLayerNode = currentLayer->nodes + j; /* Get current node that is
                                                                        being calculated */

                    /* Runs backward propagation function */
                    currLayerNode->biasWeights =
                        backwardsPropagation(av, trainingData.output, currLayerNode->biasWeights,
                                             outputLayer->nodes->activatedVal,
                                             currLayerNode->muladd, TRAINING_MAX,
                                             currLayerNode->connections);
                }
            }
        }
    } while (MAE_VAL > minMae); /* Cheks if required MAE value is met */

    return network;
}


/**
 * predict():    Uses calculates weights and biases against test data
 * @data:        Testing dataset
 * @biasWeights: Bias and Weights after training
 *
 * Return:       Array of prediction
 */
int* predict(InputOutput_t testingData, Layer_t* network, int numLayers) {
    for (int i = 0; i < numLayers; i++) {
        if (i == 0) network->layerOutput = trainTestLayer(*network, testingData.input,
                                                     network->layerOutput, TESTING_MAX);
        else (network + i)->layerOutput = trainTestLayer(*(network + i), (network + i)->prev->layerOutput ,
                                                    (network + i)->layerOutput, TESTING_MAX);
    }

    /* Predicts result from given testing input */
    double* result = (network + numLayers - 1)->nodes->activatedVal;

    /* Generated predicted output where 1 is true and 0 is no */
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
    /* Gets confusion matrix after training */
    int* prediction = predict(trainTest[1], network, numLayers);
    int* cm = confusionMatrix(trainTest[1].output, prediction, TESTING_MAX);

    /* Gets activated value of output node */
    double* finalAV = (network + numLayers - 1)->nodes->activatedVal;

    /* Calculates MMSE after training */
    printf("\n-After Training-\nMMSE Training: %f\n",
            minMeanSquareError(trainTest[0].output, finalAV, TRAINING_MAX));
    printf("MMSE Testing: %f\n\n",
            minMeanSquareError(trainTest[1].output, finalAV, TESTING_MAX));

    puts("-Confusion Matrix-");
    printf("True Positive: %d\n", cm[0]);
    printf("True Negative: %d\n", cm[1]);
    printf("False Positive: %d\n", cm[2]);
    printf("False Negative: %d\n", cm[3]);

    free(prediction);
    free(cm);
}
