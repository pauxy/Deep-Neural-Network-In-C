/* dataparser.c -- Parses dataset from csv file into 2D array
 * Author: Lim Chun Yu 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataparser.h"

const int DATA_ROWS = 100;
const int TRAINING_MAX = 90;
const int TESTING_MAX = DATA_ROWS - TRAINING_MAX;
const int DATA_COLUMNS = 10;
const int ATTR_COLUMNS = DATA_COLUMNS - 1; /* columns exclusive of results */

/**
 * splitData(): splits data into 2 InputOutput_t structs training and testing
 *
 * @data:      Raw data from file
 *
 * @training   first 90 values in data 
 * @testing    last 10 values in data
 * @split      pointer of InputOutput_t that stores 2 values, training first then testing
 * 
 * Return:     pointer with 2 InputOutput_t structs
 */

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


/**
 * openData(): Opens dataset file and convert into 2D array
 *
 * @filename:  Name of file dataset is located
 *
 * @row:       Row in dataset
 * @data       Struct of separated inputs and outputs
 * @line       Buffer to read each line in file
 * @count      Counter for inserting attribute value into correct element in array
 * @token      Placeholder for using ',' as delimeter
 * @col        Column current loop is on
 *
 * Return:     Struct with 2D and 1D array of inputs and outputs
 */
InputOutput_t openData(char* filename) {
    FILE* filelist = fopen(filename, "r");
    /* Authored by: Germaine Wong */
    if (filelist == NULL) { /* check if file exist */
        fprintf(stderr, "File could not be opened\n");
        exit(1);
    }
    /* ~ end ~ */

    InputOutput_t data;
    data.output = (int*)malloc(DATA_ROWS * sizeof(int));
    data.input = (double**)malloc(DATA_ROWS * sizeof(double*));

    char line[256];
    int count = 0;
    while (fgets(line, sizeof(line), filelist) != NULL) { /* while file still has lines */
        char* token = strtok(line, ",");                    /* gets first data between ',' */
        data.input[count] = (double*)malloc((DATA_COLUMNS - 1) * sizeof(double));

        for (int col = 0; col < DATA_COLUMNS; col++) {
            if (col != 0) token = strtok(NULL, ",");        /* gets remaining data between ',' */
            if (col == DATA_COLUMNS - 1) {
                data.output[count] = atoi(token);
            } else {
                data.input[count][col] = atof(token);            /* convert string to float and assign */
            }
        }
        count++;
    }
    return data;
}
