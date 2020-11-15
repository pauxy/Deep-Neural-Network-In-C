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


double* initBiasWeights(int inputFields) {
    double* biasWeights = (double*)malloc((inputFields + 1) * sizeof(double)); // Allocate space for weights +1 bias
    *biasWeights = 0;

    srand(time(NULL));
    for (int cols = 1; cols < inputFields + 1; cols++) {
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

    Node_h* node = (Node_h*)malloc(sizeof(Node_h));

    node->biasWeights = initBiasWeights(ATTR_COLUMNS);

    int t = 0;

    node->lr = (double*)malloc(TRAINING_MAX * sizeof(double));
    node->activatedVal = (double*)malloc(TRAINING_MAX * sizeof(double));
    node->maeVal = 0.0;

    do {
        node->lr = linearRegression(trainTest.training, node->biasWeights, TRAINING_MAX);
        node->activatedVal = sigmoid(node->lr, TRAINING_MAX);
        node->maeVal = meanAbsoluteValue(trainTest.training, node->activatedVal,
                                         TRAINING_MAX);
        t++;

        fprintf(graph, "%i %lf \n", t, node->maeVal);
        if (node->maeVal > 0.25) {
            node->biasWeights = backwardsPropagation(trainTest.training, node->biasWeights,
                                                     node->activatedVal, node->lr);
        }

    } while (node->maeVal > 0.25);

    fprintf(gnuplotPipe, "plot 'graph.temp' with lines\n");
    fclose(gnuplotPipe);
    fclose(graph);

    printf("MMSE Training: %f\n", minMeanSquareError(trainTest.training, node->activatedVal,
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
    free(node->biasWeights);
    free(node->lr);
    free(node->activatedVal);
    return 0;
}
