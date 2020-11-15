/* main.c -- Home of the main function
 * Author: Lim Chun Yu
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "node.h"
#include "dataparser.h"
#include "forwardprop.h"
#include "backprop.h"
#include "loss.h"

typedef struct Dataset_t {
    double** training;
    double** testing;
} Dataset_t;


double* initBiasWeights() {
    double* biasWeights = (double*)malloc((ATTR_COLUMNS + 1) * sizeof(double)); // Allocate space for weights +1 bias
    *biasWeights = 0;

    srand(time(NULL));
    for (int cols = 1; cols < ATTR_COLUMNS + 1; cols++) {
        *(biasWeights + cols) = (double)rand() / (double)RAND_MAX; // Randomly assign weights and bias
        printf("Random Number %i:%f\n", cols, *(biasWeights + cols));
    }
    return biasWeights;
}


Dataset_t splitData(double** data) {
    Dataset_t split;
    split.training = (double**)malloc(TRAINING_MAX * sizeof(double*));

    split.training = data;
    split.testing = data + TRAINING_MAX;

    return split;
}


int main() {
    FILE* graph = fopen("graph.temp","w");
    FILE * gnuplotPipe = popen("gnuplot -persistent", "w");

    double** data = openData("dataset/fertility_Diagnosis_Data_Group1_4-1.txt");
    Dataset_t trainTest = splitData(data);

    double* biasWeights = initBiasWeights();

    int t = 0;

    Node_h* layer1node1;
    layer1node1->lr = (double*)malloc(TRAINING_MAX * sizeof(double));
    layer1node1->activatedVal = (double*)malloc(TRAINING_MAX * sizeof(double));
    layer1node1->maeVal = 0.0;

    do {
        layer1node1->lr = linearRegression(trainTest.training, biasWeights, TRAINING_MAX);
        layer1node1->activatedVal = sigmoid(layer1node1->lr, TRAINING_MAX);
        layer1node1->maeVal = meanAbsoluteValue(trainTest.training, layer1node1->activatedVal,
                                                TRAINING_MAX);
        t++;

        fprintf(graph, "%i %lf \n", t, layer1node1->maeVal);
        if (layer1node1->maeVal > 0.25) {
            biasWeights = backwardsPropagation(trainTest.training, biasWeights,
                                               layer1node1->activatedVal, layer1node1->lr);
        }

    } while(layer1node1->maeVal > 0.25);

    fprintf(gnuplotPipe, "plot 'graph.temp' with lines\n");
    fclose(gnuplotPipe);
    fclose(graph);
    printf("MMSE Training: %f\n", minMeanSquareError(trainTest.training, layer1node1->activatedVal,
                                                     TRAINING_MAX));

    /* printf("MMSE Testing: %f\n", minMeanSquareError(testing, activatedVal, TESTING_MAX)); */

    /* Testing */
    /* double* testLR =  (double*)malloc(TESTING_MAX * sizeof(double)); */
    /* testLR = linearRegression(testing, biasWeights, TESTING_MAX); */
    /* testLR = sigmoid(testLR, TESTING_MAX); */
    /* for(int i = 0; i < TESTING_MAX; i++){ */
        /* //printf("test: %f\n", *(testLR + i)); */
        /* if ( *(testLR + i) > 0.25) { */
            /* *(testLR + i) = 1; */
        /* } else { */
            /* *(testLR + i) = 0; */
        /* } */
    /* } */

    /* char** cm = (char**)malloc(2 * 10 * sizeof(char*)); */
    /* cm = confusionMatrix(testing, testLR, 10); */
    /* printf("Origin      Predict         Res\n"); */
    /* for (int i = 0; i < TESTING_MAX; i++) { */

        /* printf("%f     %f      %s\n",testing[0+i][DATA_COLUMNS-1],testLR[0+i],cm[0+i]); */
    /* } */

    free(trainTest.training);
    free(biasWeights);
    free(layer1node1->lr);
    free(layer1node1->activatedVal);
    return 0;
}
