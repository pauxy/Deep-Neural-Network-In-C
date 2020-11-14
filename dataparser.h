/* dataparser.h
 * Author: Lim Chun Yu
 */

#ifndef DATAPARSER_H
#define DATAPARSER_H

extern const int TRAINING_MAX;
extern const int TESTING_MAX;
extern const int DATA_COLUMNS;
extern const int DATA_ROWS;
extern const double LEARNING_RATE;
extern const int ATTR_COLUMNS;

double** openData(char*);

#endif // DATAPARSER_H
