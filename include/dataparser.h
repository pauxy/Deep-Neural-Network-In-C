/* dataparser.h
 * Author: Lim Chun Yu
 */

#ifndef DATAPARSER_H
#define DATAPARSER_H

extern const int DATA_COLUMNS;
extern const int TRAINING_MAX;
extern const int TESTING_MAX;
extern const int DATA_ROWS;
extern const int ATTR_COLUMNS;

typedef struct InputOutput_t{
    int* output;
    double** input;
} InputOutput_t;

InputOutput_t openData(char*);
InputOutput_t* splitData(InputOutput_t);

#endif // DATAPARSER_H
