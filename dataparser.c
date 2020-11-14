/* dataparser.c -- Parses dataset from csv file into 2D array
 * Author: Lim Chun Yu 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataparser.h"

const int TRAINING_MAX = 90;
const int TESTING_MAX = 10;
const int DATA_COLUMNS = 10;
const int DATA_ROWS = 100;
const double LEARNING_RATE = 0.05;
const int ATTR_COLUMNS = DATA_COLUMNS - 1; /* columns exclusive of results */

double** openData(char* filename) {
    FILE* filelist = fopen(filename, "r");

    double* val = (double*)malloc(DATA_ROWS * DATA_ROWS * sizeof(double));
    double** data = (double**)malloc(DATA_ROWS * sizeof(double*));

    char line[256];
    int count = 0;

    while (fgets(line, sizeof(line), filelist) != NULL) {   /* while file still has lines */
        data[count] = val + (count * DATA_ROWS);            /* 2d array assign */
        char* new = strtok(line, ",");                      /* gets first data between ',' */
        for (int col = 0; col < DATA_COLUMNS; col++) {
            if (col != 0) new = strtok(NULL, ",");            /* gets remaining data between ',' */

            // printf("%d  %f  %d \n", count, atof(new), i);   /* printf for testing */
            data[count][col] = atof(new);                     /* convert string to float and assign */
        }
        count++;                                            /* counter */
    }
    return data;
}
