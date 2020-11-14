/* forwardprop.c -- Calculates linear regression and sigmoid function
 * Author: Lim Chun Yu
 */

#include <stdlib.h>
#include <math.h>

#include "dataparser.h"
#include "forwardprop.h"

double* linearRegression(double** data, double* biasWeights, int val){  //2a
    double* lr = (double*)malloc(val * sizeof(double));
    for (int rows = 0; rows < val; rows++) {
        *(lr + rows) = 0;
        for (int cols = 0; cols < ATTR_COLUMNS; cols++) {
            // printf("%f\n", *(lr + rows));
            *(lr + rows) += (*(biasWeights + 1 + cols) * data[rows][cols]) + *biasWeights;
        }
    }
    return lr;
}


double* sigmoid(double* lr, int val) {                                  //2b
    double* activatedVal = (double*)malloc(val * sizeof(double));
    for(int rows = 0; rows < val; rows++){
        *(activatedVal + rows) = 1.0 / (1.0 + exp(- *(lr + rows)));
    }
    return activatedVal;
}
