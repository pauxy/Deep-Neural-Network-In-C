/* error.c -- Calculates MMSE and confusion matrix
 * Author: Lim Chun Yu
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "dataparser.h"

/**
 * meanAbsoluteValue(): Calculates MAE for each iteration
 *
 * @expectedOutput:     Correct output value from dataset
 * @activatedVal:       Activated values passed from the sigmoid function
 * @batchSize:          Size of data in rows
 *
 * @total:              Absolute sum of activated values minus true label of data sample
 *
 * Part 2c
 * Calculates the mean absolute value for each iteration where the activated values is subtracted by
 * the true label of the give dataset, and averaged using formula
 * \frac{\sum_{i=1}^{I} |\hat{y_{l}}^{t} - d_{i}|}{I}
 *
 * Return: Averaged sum of activated values subtracted by true label
 */
double meanAbsoluteValue(int* expectedOutput, double* activatedVal, int batchSize) {
    double total = 0.0;
    for (int rows = 0; rows < batchSize; rows++) {
        total += fabs( *(activatedVal + rows) -
                expectedOutput[rows]); /* formula provided for MAE, calcadds everyvalue to be
                                          divided in line 32 */
    }
    return total / batchSize;
}


/**
 * minMeanSquareError(): Calculates the minimum average of the squares of errors
 *
 * @expectedOutput:      Correct output value from dataset
 * @activatedVal:        Activated values passed from the sigmoid function
 * @batchSize:           Size of data in rows
 *
 * @total:               Sum of the squares of errors
 *
 * Part 2e
 * Calculates Minimum Mean Square Error using formula
 * \frac{1}{I} \sum_{i=1}^{I} (\hat{y_{l}}^{t} - d_{i})^2
 *
 * Return: Average of the sum of squares of errors
 */
double minMeanSquareError(int* expectedOutput, double* activatedVal, int batchSize) {
    double total = 0.0;
    for (int rows = 0; rows < batchSize; rows++) {
        total += pow(*(activatedVal + rows) -
                expectedOutput[rows], 2.0); /* TODO:description */
    }
    return total / batchSize;
}


/**
 * confusionMatrix(): Evaluates prediction to true label
 *
 * @data:             Attributes from dataset
 * @res:              Result from prediction
 * @batchSize:        Size of data in rows
 *
 * Part 2e
 * Determines the four classes
 * - True Positive
 * - True Negative
 * - False Positive
 * - False Negative
 *
 * Return: Array of strings for confusion matrix
 */
char** confusionMatrix(int* expectedOutput, int* res, int batchSize) {
    char** confusion = (char**)malloc(batchSize * 2 * sizeof(char*));
    int TP = 0;
    int TN = 0;
    int FP = 0;
    int FN = 0;
    for (int i = 0; i < batchSize; i++) {
        int origin = expectedOutput[i];
        char* con = "TP";
        if (origin == res[i]) {
            TP++;
            if (origin == 0) {
                con = "TN"; /* true negative */
                TN++;
                TP--;
            }
        /* origin == 1 true positive */
        } else {
            con = "FP";
            FP++; /* false positive */
            if (origin == 0) {
                con = "FN"; /* false negative */
                FN++;
                FP--;
            }
        }
        *(confusion + i) = con;
    }
    printf("\n-Confusion Matrix-\nTrue Positive: %d\nTrue Negative: %d\nFalse Positive: %d\nFalse Negative: %d\n", TP, TN, FP, FN);
    return confusion;
}
