/* loss.c -- 
 * Author: Lim Chun Yu
 */

#include <math.h>
#include <stdlib.h>

#include "dataparser.h"

/**
 * minMeanSquareError():
 *
 * @training:
 * @activatedVal:
 * @val:
 *
 * @total:
 *
 * Return:
 */
double minMeanSquareError(double** training, double* activatedVal, int val) {
    double total = 0.0;
    for (int rows = 0; rows < val; rows++) {
        total += pow(*(activatedVal + rows) - training[rows][DATA_COLUMNS - 1],2.0);
    }
    return total / val ;
}


/**
 * confusionMatrix():
 *
 * @res:
 * @data:
 * @val:
 *
 * Return:
 */
char** confusionMatrix(double* res, double** data, int val){
    char** confusion = (char**)malloc(val * 2 * sizeof(char*));
    for (int i = 0; i < val; i++){
        int origin = data[i][DATA_COLUMNS - 1];
        int result = res[i];
        char* con = "PP";
        if (origin == result) {
            if (origin == 0) con = "FF"; // true neg
            //1=true positive
        } else {
            con = "PF"; // false positive
            if (origin == 0) con = "FP"; // false neg
        }
        //printf("%d  %d  %i\n",origin,result,con);
        *(confusion + i) = con;
    }
    return confusion;
}
