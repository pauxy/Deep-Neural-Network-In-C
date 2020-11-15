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

int main() {
    FILE* graph = fopen("graph.temp","w");
    FILE * gnuplotPipe = popen("gnuplot -persistent", "w");
    double** data = openData("dataset/fertility_Diagnosis_Data_Group1_4-1.txt");
    double** training = (double**)malloc(TRAINING_MAX * sizeof(double*));
    double** testing = (double**)malloc(TESTING_MAX * sizeof(double*));
    training = data;
    testing = data + 90;
    srand(time(NULL));
    double* biasWeights = (double*)malloc((ATTR_COLUMNS + 1) * sizeof(double));//allocate space for weights +1 bias
    *biasWeights = 0;
    for (int cols = 1; cols < ATTR_COLUMNS + 1; cols++) {
        *(biasWeights + cols) = (double)rand() / (double)RAND_MAX;//randomly assign weights and bias
        printf("Random Number %i:%f\n", cols, *(biasWeights + cols));
    }
    int t = 0;
    double* lr =  (double*)malloc(TRAINING_MAX * sizeof(double));
    double* activatedVal = (double*)malloc(TRAINING_MAX * sizeof(double));
    double maeVal = 0.0;

    do {
        lr = linearRegression(training, biasWeights, TRAINING_MAX);
        activatedVal = sigmoid(lr, TRAINING_MAX);
        maeVal = meanAbsoluteValue(training, activatedVal,TRAINING_MAX);
        t += 1;

        if (t % 10 == 0) {                              //testing
            printf("MAE value: %f\n",maeVal);
            printf("T value: %i\n",t);
            printf("sigmoid: %f\n",*(activatedVal));
        }
        fprintf(graph, "%i %lf \n", t, maeVal);
        if(maeVal>0.25){
            biasWeights = backwardsPropagation(training, biasWeights, activatedVal, lr);
        }
    } while(maeVal>0.25);
//    fprintf(gnuplotPipe, "plot 'graph.temp' with lines\n");
    printf("MMSE Training: %f\n",minMeanSquareError(training,activatedVal,TRAINING_MAX));
    printf("MMSE Testing: %f\n",minMeanSquareError(testing,activatedVal,TESTING_MAX));
    //TESTING
    double* testLR =  (double*)malloc(TESTING_MAX* sizeof(double));
    testLR=linearRegression(testing,biasWeights,TESTING_MAX);
    testLR=sigmoid(testLR,TESTING_MAX);
    for(int i=0;i<TESTING_MAX;i++){
        //printf("test: %f\n",*(testLR+i));
        if(*(testLR+i)>0.25){
            *(testLR+i)=1;//printf("1                  %f\n",training[i][DATA_COLUMNS - 1]);
        }else{
            *(testLR+i)=0;
    }
    }
    char** cm=(char**)malloc(2*10*sizeof(char*));
    cm=confusionMatrix(testing,testLR,10);
        printf("origin     predict         res\n");
    for(int i=0;i<TESTING_MAX;i++){

        printf("%f     %f      %s\n",testing[0+i][DATA_COLUMNS-1],testLR[0+i],cm[0+i]);
    }
    return 0;
}
