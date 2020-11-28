/* main.c -- Home of the main function
 * Author: Lim Chun Yu
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>


#include "dataparser.h"
#include "forwardprop.h"
#include "backprop.h"
#include "error.h"
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
void help(){
    puts("Perceptron command line input help.");
    puts("Options are:");
    puts("-a <file name> : sets input file");
    puts("-b <mae value> : sets new minimum mae value");
    puts("-c <graph name> : sets new graph name");
    puts("-d <file name> : sets new output file name");
}


int main(int argc, char **argv) {

    double reqMae = 0.25;
    char *ngraph = "perceptron";
    char *dfile = "dataset/fertility_Diagnosis_Data_Group1_4-1.txt";
    char *ofile = "graph.temp";

    int c;
    struct timeval  tv1, tv2;
    gettimeofday(&tv1, NULL);
    while ((c = getopt (argc, argv, "m:i:g:o:h")) != -1)
    switch (c) {
        case 'm':
            reqMae = atof(optarg);
            break;
        case 'i':
            dfile = optarg;
            break;
        case 'g':
            ngraph = optarg;
            break;
        case 'o':
            ofile = optarg;
            break;
        case '?':
            if (optopt == 'm' || optopt == 'i' || optopt == 'g' || optopt == 'o')
                fprintf(stderr, "Option -%c requires an argument.\n", optopt);
            else
                help();
            return 1;
      default:
            help();
            return 0;
    }

    FILE* graph = fopen(ofile, "w");
    FILE * gnuplotPipe = popen("gnuplot -persistent", "w");

    InputOutput_t data = openData(dfile);
    InputOutput_t* trainTest = splitData(data);

    Node_t* node = (Node_t*)malloc(sizeof(Node_t));
    node->connections = ATTR_COLUMNS;
    node->biasWeights = initBiasWeights(node->connections);
    node->muladd = (double*)malloc(TRAINING_MAX * sizeof(double));
    node->activatedVal = (double*)malloc(TRAINING_MAX * sizeof(double));

    int t = 0;
    MAE_VAL = 0.0;

    do {
        node->activatedVal = forwardPropagation(trainTest[0].input, node->biasWeights,
                                                node->muladd, node->activatedVal,
                                                TRAINING_MAX, node->connections);
        MAE_VAL = meanAbsoluteValue(trainTest[0].output, node->activatedVal,
                                    TRAINING_MAX);
        if(t == 0) {
            printf("-Before Training-\nMMSE Training: %f\n",
                    minMeanSquareError(trainTest[0].output, node->activatedVal, TRAINING_MAX));
            printf("MMSE Testing: %f\n",
                    minMeanSquareError(trainTest[1].output, node->activatedVal, TESTING_MAX));
        }
        t++;

        fprintf(graph, "%i %lf \n", t, MAE_VAL);
        if (MAE_VAL > 0.25) {
            node->biasWeights = backwardsPropagation(trainTest[0].input, trainTest[0].output,
                                                     node->biasWeights, node->activatedVal,
                                                     node->muladd, TRAINING_MAX, node->connections);
        }

    } while (MAE_VAL > reqMae);
    fprintf(gnuplotPipe, "set title \"%s\"\n", ngraph);
    fprintf(gnuplotPipe, "plot '%s' with lines\n", ofile);
    fclose(gnuplotPipe);
    fclose(graph);
    printf("\n-After Training-\nMMSE Training: %f\n",
            minMeanSquareError(trainTest[0].output, node->activatedVal, TRAINING_MAX));
    printf("MMSE Testing: %f\n",
            minMeanSquareError(trainTest[1].output, node->activatedVal, TESTING_MAX));
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
    free(node->muladd);
    free(node->activatedVal);
    free(resPredict.prediction);
    free(resPredict.result);
    free(cm);
    return 0;
}
