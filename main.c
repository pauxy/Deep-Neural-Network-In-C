#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

const int TRAINING_MAX = 90;
const int TESTING_MAX = 10;
const int DATA_COLUMNS = 10;
const int DATA_ROWS = 100;
const double LEARNING_RATE = 0.05;

double** openData(char*);
double linearRegression(double**, double*);
double sigmoid(double);
double meanAbsoluteValue(double**, double);
double* backwardsPropagation(double*, double, double**, double);

double** openData(char* filename) {
    double* val = (double*)malloc(DATA_ROWS * DATA_ROWS * sizeof(double));
    double** data = (double**)malloc(DATA_ROWS * sizeof(double*));
    FILE* filelist;
    filelist = fopen(filename, "r");
    char line[256];
    int count = 0;

    while (fgets(line, sizeof(line), filelist) != NULL) {
        data[count] = val + (count * DATA_ROWS);

        char* new = strtok(line, ",");
        for (int i = 0; i < DATA_COLUMNS; i++) {
            if (i != 0) new = strtok(NULL, ",");

            printf("%d  %f  %d \n", count, atof(new), i);
            data[count][i] = atof(new);
        }
        count += 1;
    }

    return data;
}


double linearRegression(double** data, double* biasWeights) {
    double total = 0.0;
    double bias = *biasWeights;
    for (int i = 0; i < TRAINING_MAX; i++) {
        for (int j = 0; j < DATA_COLUMNS; j++) {
            total += *(biasWeights + 1 + j) * data[i][j];
        }
        total += bias;
    }
    return total;
}


double sigmoid(double sumLR) {
    return 1.0 / (1.0 + exp(-sumLR));
}


double meanAbsoluteValue(double** training, double activatedVal) {
    double total = 0.0;
    for (int i = 0; i < TRAINING_MAX; i++) {
        total += fabs(activatedVal - training[i][DATA_COLUMNS - 1]);
    }
    return total / TRAINING_MAX;
}


double* backwardsPropagation(double* biasWeights, double activatedVal,
                             double** training, double sumLR) {
    double* newBiasWeights = (double*)malloc((DATA_COLUMNS + 1) * sizeof(double));
    double ph = exp(sumLR) / pow(1.0 + exp(sumLR), 2.0);
    double biasTotal = 0.0;
    for (int j = 0; j < DATA_COLUMNS; j++) {
        double weightTotal = 0.0;
        for (int i = 0; i < TRAINING_MAX; i++) {
            double ph1 = activatedVal - training[i][DATA_COLUMNS - 1];
            /* printf("%f %f\n", ph1, ph); */
            weightTotal += (ph * ph1 * training[i][j]);
            if (j == 0) {
                biasTotal += ph * ph1;
            }
        }
        *(newBiasWeights + 1 + j) = *(biasWeights + 1 + j) - (LEARNING_RATE * (weightTotal / TRAINING_MAX));
    }
    *newBiasWeights = *(biasWeights) - (LEARNING_RATE * (biasTotal / TRAINING_MAX));
    /* printf("diff: %f\n", biasTotal); */
    return newBiasWeights;
}


int main() {
    double** data = openData("dataset/fertility_Diagnosis_Data_Group1_4-1.txt");
    double** training = (double**)malloc(TRAINING_MAX * sizeof(double*));
    double** testing = (double**)malloc(TESTING_MAX * sizeof(double*));
    training = data;
    testing = data + 90;
    srand(time(NULL));
    double* biasWeights = (double*)malloc((DATA_COLUMNS + 1) * sizeof(double));
    *biasWeights = -1.5;
    for (int i = 1; i < DATA_COLUMNS + 1; i++) {
        *(biasWeights + i) = (double)rand() / (double)RAND_MAX;
        printf("Random Number %i:%f\n", i, *(biasWeights + i));
    }
    int t = 0;
    double sumLR = 0.0;
    double activatedVal = 0.0;
    double maeVal = 0.0;

    do {
        if (t > 0) {
            biasWeights = backwardsPropagation(biasWeights, activatedVal, training, sumLR);
            for (int i = 1; i < DATA_COLUMNS + 1; i++) {
                /* printf("Random Number %i:%f\n", i, *(biasWeights + i)); */
            }
        }
        sumLR = linearRegression(training, biasWeights);
        activatedVal = sigmoid(sumLR);
        maeVal = meanAbsoluteValue(training, activatedVal);
        t += 1;

        if (t % 10000 == 0 || t < 10000) {
            printf("Sum of LR: %f\n",sumLR);
            printf("Sigmoid value: %0.300f\n",activatedVal);
            printf("MAE value: %f\n",maeVal);
            printf("T value: %i\n",t);
        }
    } while (maeVal > 0.25);

    return 0;
}
