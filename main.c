/* main.c -- Home of the main function
 * Author: Lim Chun Yu
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dataparser.h"
#include "forwardprop.h"
#include "backprop.h"
#include "loss.h"
#include "mlp.h"

double MAE_VAL;

typedef struct Dataset_t {
    double** training;
    double** testing;
} Dataset_t;

Dataset_t splitData(double** data) {
    Dataset_t split;
    split.training = (double**)malloc(TRAINING_MAX * sizeof(double*));

    split.training = data;
    split.testing = data + TRAINING_MAX;

    return split;
}


typedef struct ResultPrediction_t {
    double* result;
    int* prediction;
} ResultPrediction_t;

ResultPrediction_t predict(double** data, double* biasWeights) {
    ResultPrediction_t resPredict;

    resPredict.result = linearRegression(data, biasWeights, TESTING_MAX);
    resPredict.result = sigmoid(resPredict.result, TESTING_MAX);

    resPredict.prediction = (int*)malloc(TESTING_MAX * sizeof(int));
    for (int i = 0; i < TESTING_MAX; i++){
        if ( *(resPredict.result + i) > 0.5)
            *(resPredict.prediction + i) = 1;
        else
            *(resPredict.prediction + i) = 0;
    }
    return resPredict;
}


int main() {
    FILE* graph = fopen("graph.temp","w");
    FILE * gnuplotPipe = popen("gnuplot -persistent", "w");

    double** data = openData("dataset/fertility_Diagnosis_Data_Group1_4-1.txt");
    Dataset_t trainTest = splitData(data);

    Node_t* node = (Node_t*)malloc(sizeof(Node_t));

    node->biasWeights = initBiasWeights(ATTR_COLUMNS);

    int t = 0;

    node->lr = (double*)malloc(TRAINING_MAX * sizeof(double));
    node->activatedVal = (double*)malloc(TRAINING_MAX * sizeof(double));
    MAE_VAL = 0.0;

    do {
        node->lr = linearRegression(trainTest.training, node->biasWeights, TRAINING_MAX);
        node->activatedVal = sigmoid(node->lr, TRAINING_MAX);
        MAE_VAL = meanAbsoluteValue(trainTest.training, node->activatedVal,
                                    TRAINING_MAX);
        t++;

        fprintf(graph, "%i %lf \n", t, MAE_VAL);
        if (MAE_VAL > 0.25) {
            node->biasWeights = backwardsPropagation(trainTest.training, node->biasWeights,
                                                     node->activatedVal, node->lr);
        }

    } while (MAE_VAL > 0.25);

    fprintf(gnuplotPipe, "plot 'graph.temp' with lines\n");
    fclose(gnuplotPipe);
    fclose(graph);

    printf("MMSE Training: %f\n", minMeanSquareError(trainTest.training, node->activatedVal,
                                                     TRAINING_MAX));
    ResultPrediction_t resPredict = predict(trainTest.testing, node->biasWeights);
    char** cm = confusionMatrix(trainTest.testing, resPredict.prediction, TESTING_MAX);

    /* printf("MMSE Testing: %f\n", minMeanSquareError(testing, activatedVal, TESTING_MAX)); */

    free(trainTest.training);
    free(node->biasWeights);
    free(node->lr);
    free(node->activatedVal);
    free(resPredict.prediction);
    free(resPredict.result);
    return 0;
}
