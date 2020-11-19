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

double*** splitData(double** data) {
    double** training = (double**)malloc(TRAINING_MAX * sizeof(double*));

    training = data;
    double** testing = data + TRAINING_MAX;

    double*** split = (double***)malloc(2 * sizeof(double**));
    split[0] = training;
    split[1] = testing;

    return split;
}


typedef struct ResultPrediction_t {
    double* result;
    int* prediction;
} ResultPrediction_t;

ResultPrediction_t predict(double** data, BiasWeights_t biasWeights) {
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
    double*** trainTest = splitData(data);

    Node_t* node = (Node_t*)malloc(sizeof(Node_t));

    node->biasWeights = initBiasWeights(ATTR_COLUMNS);

    int t = 0;

    node->lr = (double*)malloc(TRAINING_MAX * sizeof(double));
    node->activatedVal = (double*)malloc(TRAINING_MAX * sizeof(double));
    MAE_VAL = 0.0;

    do {
        node->lr = linearRegression(trainTest[0], node->biasWeights, TRAINING_MAX);
        node->activatedVal = sigmoid(node->lr, TRAINING_MAX);
        MAE_VAL = meanAbsoluteValue(trainTest[0], node->activatedVal,
                                    TRAINING_MAX);
        t++;

        fprintf(graph, "%i %lf \n", t, MAE_VAL);
        if (MAE_VAL > 0.25) {
            node->biasWeights = backwardsPropagation(trainTest[0], node->biasWeights,
                                                     node->activatedVal, node->lr);
        }

    } while (MAE_VAL > 0.25);

    fprintf(gnuplotPipe, "plot 'graph.temp' with lines\n");
    fclose(gnuplotPipe);
    fclose(graph);

    printf("MMSE Training: %f\n", minMeanSquareError(trainTest[0], node->activatedVal,
                                                     TRAINING_MAX));
    ResultPrediction_t resPredict = predict(trainTest[1], node->biasWeights);
    char** cm = confusionMatrix(trainTest[1], resPredict.prediction, TESTING_MAX);

    /* printf("MMSE Testing: %f\n", minMeanSquareError(testing, activatedVal, TESTING_MAX)); */

    free(trainTest[0]);
    free(trainTest);
    free(node->biasWeights.weights);
    free(node->lr);
    free(node->activatedVal);
    free(resPredict.prediction);
    free(resPredict.result);
    return 0;
}
