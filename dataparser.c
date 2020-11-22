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
const int ATTR_COLUMNS = DATA_COLUMNS - 1; /* columns exclusive of results */

/**
 * openData(): Opens dataset file and convert into 2D array
 *
 * @filename:  Name of file dataset is located
 *
 * @row:       Row in dataset
 * @data       Column/Attribute in dataset
 * @line       Buffer to read each line in file
 * @count      Counter for inserting attribute value into correct element in array
 * @token      Placeholder for using ',' as delimeter
 * @col        Column current loop is on
 *
 * Return:     2D array of attributes
 */
double** openData(char* filename) {
    FILE* filelist = fopen(filename, "r");
    /* Authored by: Germaine Wong */
    if (filelist == NULL) { /* check if file exist */
        puts("File could not be opened");
        exit(1);
    }
    /* ~ end ~ */
    double* row = (double*)malloc(DATA_ROWS * DATA_ROWS * sizeof(double));
    double** data = (double**)malloc(DATA_ROWS * sizeof(double*));

    char line[256];
    int count = 0;

    while (fgets(line, sizeof(line), filelist) != NULL) {   /* while file still has lines */
        data[count] = row + (count * DATA_ROWS);            /* 2d array assign */
        char* token = strtok(line, ",");                    /* gets first data between ',' */
        for (int col = 0; col < DATA_COLUMNS; col++) {
            if (col != 0) token = strtok(NULL, ",");        /* gets remaining data between ',' */

            // printf("%d  %f  %d \n", count, atof(token), i); /* printf for testing */
            data[count][col] = atof(token);                 /* convert string to float and assign */
        }
        count++;
    }
    return data;
}
