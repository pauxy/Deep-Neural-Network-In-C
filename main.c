/* main.c -- Home of the main function
 * Author: Lim Chun Yu
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "dataparser.h"
#include "forwardprop.h"
#include "backprop.h"
#include "loss.h"
#include "mlp.h"


double MAE_VAL;

InputOutput_t* splitData(InputOutput_t data) {
    InputOutput_t training;
    training.input = data.input;
    training.output = data.output;

    InputOutput_t testing;
    testing.input = data.input + TRAINING_MAX;
    testing.output = data.output + TRAINING_MAX;

    InputOutput_t* split = (InputOutput_t*)malloc(2 * sizeof(InputOutput_t));
    split[0] = training;
    split[1] = testing;

    return split;
}


typedef struct ResultPrediction_t {
    double* result;
    int* prediction;
} ResultPrediction_t;

ResultPrediction_t predict(InputOutput_t data, BiasWeights_t biasWeights) {
    ResultPrediction_t resPredict;

    resPredict.result = (double*)malloc(TESTING_MAX * sizeof(double));
    resPredict.result = linearRegression(data.input, biasWeights, resPredict.result,
                                         TESTING_MAX, ATTR_COLUMNS);
    resPredict.result = sigmoid(resPredict.result, resPredict.result, TESTING_MAX);

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
    struct timeval  tv1, tv2;
    gettimeofday(&tv1, NULL);
    FILE* graph = fopen("graph.temp","w");
    FILE * gnuplotPipe = popen("gnuplot -persistent", "w");

    InputOutput_t data = openData("dataset/fertility_Diagnosis_Data_Group1_4-1.txt");
    InputOutput_t* trainTest = splitData(data);

    Node_t* node = (Node_t*)malloc(sizeof(Node_t));
    node->connections = ATTR_COLUMNS;
    node->biasWeights = initBiasWeights(node->connections);
    node->lr = (double*)malloc(TRAINING_MAX * sizeof(double));
    node->activatedVal = (double*)malloc(TRAINING_MAX * sizeof(double));

    int t = 0;
    MAE_VAL = 0.0;

    do {
        node->lr = linearRegression(trainTest[0].input, node->biasWeights, node->lr,
                                    TRAINING_MAX, node->connections);
        node->activatedVal = sigmoid(node->lr, node->activatedVal, TRAINING_MAX);
        MAE_VAL = meanAbsoluteValue(trainTest[0].output, node->activatedVal,
                                    TRAINING_MAX);
        if(t==0){
            printf("-Before Training-\nMMSE Training: %f\n", minMeanSquareError(trainTest[0].output, node->activatedVal,TRAINING_MAX));
            printf("MMSE Testing: %f\n", minMeanSquareError(trainTest[1].output,node->activatedVal,TESTING_MAX));
        }
        t++;

        fprintf(graph, "%i %lf \n", t, MAE_VAL);
        if (MAE_VAL > 0.25) {
            node->biasWeights = backwardsPropagation(trainTest[0].input, trainTest[0].output, node->biasWeights,
                                                     node->activatedVal, node->lr,
                                                     TRAINING_MAX, node->connections);
        }

    } while (MAE_VAL > 0.25);

    fprintf(gnuplotPipe, "plot 'graph.temp' with lines\n");
    fclose(gnuplotPipe);
    fclose(graph);
    printf("\n-After Training-\nMMSE Training: %f\n", minMeanSquareError(trainTest[0].output, node->activatedVal,TRAINING_MAX));
    printf("MMSE Testing: %f\n", minMeanSquareError(trainTest[1].output, node->activatedVal,TESTING_MAX));
    ResultPrediction_t resPredict = predict(trainTest[1], node->biasWeights);
    char** cm = confusionMatrix(trainTest[1].output, resPredict.prediction, TESTING_MAX);

    gettimeofday(&tv2, NULL);
    printf ("\nTotal time = %f seconds\n",
         (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
         (double) (tv2.tv_sec - tv1.tv_sec));


    /* Freeing used memory */
    for (int row = 0; row < DATA_ROWS; row++) {
        free(trainTest[0].input[row]);
    }
    free(trainTest[0].input);
    free(trainTest[0].output);
    free(trainTest);
    free(node->biasWeights.weights);
    free(node->lr);
    free(node->activatedVal);
    free(resPredict.prediction);
    free(resPredict.result);
    free(cm);
    return 0;
}
