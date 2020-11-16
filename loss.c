/* loss.c -- Calculates MMSE and confusion matrix
 * Author: Lim Chun Yu
 */

#include <math.h>
#include <stdlib.h>

#include "dataparser.h"

/**
 * minMeanSquareError(): Calculates the minimum average of the squares of errors
 *
 * @data:                Attributes from dataset
 * @activatedVal:        Activated values passed from the sigmoid function
 * @val:                 Size of data in rows
 *
 * @total:               Sum of the squares of errors
 *
 * Part 2e
 * Calculates Minimum Mean Square Error using formula
 * \frac{1}{I} \sum_{i=1}^{I} (\hat{y_{l}}^{t} - d_{i})^2
 *
 * Return: Average of the sum of squares of errors
 */
double minMeanSquareError(double** data, double* activatedVal, int val) {
    double total = 0.0;
    for (int rows = 0; rows < val; rows++) {
        total += pow(*(activatedVal + rows) -
                data[rows][DATA_COLUMNS - 1], 2.0); /* TODO:description */
    }
    return total / val ;
}


/**
 * confusionMatrix(): Evaluates prediction to true label
 *
 * @data:             Attributes from dataset
 * @res:              Result from prediction
 * @val:              Size of data in rows
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
char** confusionMatrix(double** data, int* res, int val) {
    char** confusion = (char**)malloc(val * 2 * sizeof(char*));
    for (int i = 0; i < val; i++) {
        int origin = data[i][DATA_COLUMNS - 1];
        char* con = "TP";
        if (origin == res[i]) {
            if (origin == 0) con = "TN"; /* true negative */
                                         /* origin == 1 true positive */
        } else {
            con = "FP";                  /* false positive */
            if (origin == 0) con = "FN"; /* false negative */
        }
        *(confusion + i) = con;
    }
    return confusion;
}
