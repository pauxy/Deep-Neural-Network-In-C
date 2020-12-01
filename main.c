/* main.c -- Home of the main function
 * Author: Lim Chun Yu
 */

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "dataparser.h"
#include "forwardprop.h"
#include "backprop.h"
#include "error.h"
#include "mlp.h"

#define true 1
#define false 0

typedef struct ResultPrediction_t {
    double* result;
    int* prediction;
} ResultPrediction_t;

ResultPrediction_t predict(InputOutput_t data, BiasWeights_t biasWeights) {
    ResultPrediction_t resPredict;

    resPredict.result = (double*)malloc(TESTING_MAX * sizeof(double));
    resPredict.result = forwardPropagation(data.input, biasWeights, resPredict.result,
                                           resPredict.result, TESTING_MAX, ATTR_COLUMNS);

    resPredict.prediction = (int*)malloc(TESTING_MAX * sizeof(int));
    for (int i = 0; i < TESTING_MAX; i++) {
        if ( *(resPredict.result + i) > 0.5)
            *(resPredict.prediction + i) = 1;
        else
            *(resPredict.prediction + i) = 0;
    }
    return resPredict;
}


/**
 * help(): Prints help info
 */
void help () {
    puts("Perceptron command line input help.");
    puts("Options are:");
    puts("-m <mae value> : sets new minimum mae value (default: 0.25)");
    puts("-i <file name> : sets input file (default: dataset/fertility_Diagnosis_Data_Group1_4-1.txt)");
    puts("-g <graph name> : sets new graph name (default: Mean Average Error)");
    puts("-o <file name> : sets new output file name (default: graph.temp)");
    puts("-l <no of hidden layer> : sets number of hidden layers (default: 2)");
    puts("-n <nodes per hidden layer> : sets nodes per hidden layer (default: 3,4)");
}


/**
 * checkMaeArg(): Checks if MAE give by user is between 0.2 to 1.0
 * @reqMae: Required MAE to stop training
 *
 * Return: If check passes
 */
int checkMaeArg(double reqMae) {
    if (reqMae < 0.2 || reqMae > 1.0) {
        fprintf(stderr, "Please choose a MAE value between 0.2 to 1.0\n");
        return false;
    }
    return true;
}


/**
 * checkHiddenLayerArg(): Checks if hidden layers given by user is between 0 to 10
 * @numHiddenLayers: Number of hidden layers
 *
 * Return: If check passes
 */
int checkHiddenLayerArg(int numHiddenLayers) {
    if (numHiddenLayers < 0 || numHiddenLayers > 10) {
        fprintf(stderr, "Please choose number of hidden layers between 0 to 10\n");
        return false;
    }
    return true;
}


/**
 * checkNodes():     Creates an array of nodes per layer using ',' as delimeter
 * @option:          String of command separated nodes per layer
 * @numHiddenLayers: Number of hidden layers
 *
 * Separates number of nodes per layer into an int*, checks if nodes per layer is between 0 to 10
 * and checks if no more or less nodes per hidden layer is declared
 *
 * Return: int* of nodes per layer
 */
int* checkNodes(char* option, int numHiddenLayers) {
    int* nodes = (int*)malloc(numHiddenLayers * sizeof(int));
    char* noNodes = strtok(option, ",");
    int i = 0;
    while (noNodes != NULL && i < numHiddenLayers) {
        nodes[i] = atoi(noNodes);
        if (nodes[i] < 1 || nodes[i] > 10) {
            fprintf(stderr, "Please choose a number of hidden layer nodes between 1-10\n");
            exit(1);
        }
        noNodes = strtok(NULL, ",");
        i++;
    }

    if (i != numHiddenLayers) {
        fprintf(stderr, "You entered %d values but have %d layers!\n", i, numHiddenLayers);
        exit(1);
    }
    return nodes;
}


/**
 * main(): The leader and the orchestrator
 *
 * Handles user arguments input, training, and predicting of neural network
 */
int main(int argc, char **argv) {
    int c;
    struct timeval  tv1, tv2;
    gettimeofday(&tv1, NULL);

    double reqMae = 0.25;
    char* ngraph = "Mean Average Error";
    char* dfile = "dataset/fertility_Diagnosis_Data_Group1_4-1.txt";
    char* ofile = "graph.temp";
    int numHiddenLayers = 2;
    char nodesPerLayer[64] = "3,4";
    int* nodes = NULL;

    while ((c = getopt(argc, argv, "m:i:g:o:l:n:h")) != -1)
    switch (c) {
        case 'm':
            reqMae = atof(optarg);
            break;
        case 'i':
            dfile = (char*)malloc(strlen(optarg) + 1);
            strcpy(dfile, optarg);
            break;
        case 'g':
            ngraph = (char*)malloc(strlen(optarg) + 1);
            strcpy(ngraph, optarg);
            break;
        case 'o':
            ofile = (char*)malloc(strlen(optarg) + 1);
            strcpy(ofile, optarg);
            break;
        case 'l':
            numHiddenLayers = atoi(optarg);
            break;
        case 'n':
            strncpy(nodesPerLayer, optarg, sizeof(nodesPerLayer) - 1);
            break;
        case 'h':
            help();
            return 0;
      default:
            help();
            return 1;
    }

    if (!(checkMaeArg(reqMae) && checkHiddenLayerArg(numHiddenLayers))) return 1;
    if (numHiddenLayers != 0) nodes = checkNodes(nodesPerLayer, numHiddenLayers);

    /* Print number of hidden layers */
    puts("-Layers in feed-forward neural network-");
    puts("Input Layer: 9 node(s)");
    for (int i = 0; i < numHiddenLayers; i++) {
        printf("Hidden Layer %d: %d node(s)\n", i + 1, nodes[i]);
    }
    puts("Output Layer: 1 node(s)\n");

    puts("-Required MAE to end training-");
    printf("MAE: %lf\n\n", reqMae);

    FILE* graph = fopen(ofile, "w");
    FILE * gnuplotPipe = popen("gnuplot -persistent > /dev/null 2>&1", "w");

    InputOutput_t data = openData(dfile);
    InputOutput_t* trainTest = splitData(data);

    Node_t* node = trainNetwork(numHiddenLayers, nodes, trainTest, reqMae, graph);
    printf("\n-After Training-\nMMSE Training: %f\n",
            minMeanSquareError(trainTest[0].output, node->activatedVal, TRAINING_MAX));
    printf("MMSE Testing: %f\n\n",
            minMeanSquareError(trainTest[1].output, node->activatedVal, TESTING_MAX));
    ResultPrediction_t resPredict = predict(trainTest[1], node->biasWeights);
    int* cm = confusionMatrix(trainTest[1].output, resPredict.prediction, TESTING_MAX);

    puts("-Confusion Matrix-");
    printf("True Positive: %d\n", cm[0]);
    printf("True Negative: %d\n", cm[1]);
    printf("False Positive: %d\n", cm[2]);
    printf("False Negative: %d\n", cm[3]);

    fclose(graph);
    fprintf(gnuplotPipe, "set title \"%s\"\n", ngraph);
    fprintf(gnuplotPipe, "plot '%s' with lines\n", ofile);
    fclose(gnuplotPipe);

    gettimeofday(&tv2, NULL);
    printf ("\nTotal time = %f seconds\n",
         (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec));

    /* Freeing used memory */
    for (int row = 0; row < DATA_ROWS; row++) {
        free(trainTest[0].input[row]);
    }
    free(trainTest[0].input);
    free(trainTest[0].output);
    free(trainTest);
    free(node->biasWeights.weights);
    free(node->muladd);
    free(node->activatedVal);
    free(resPredict.prediction);
    free(resPredict.result);
    free(cm);

    return 0;
}
