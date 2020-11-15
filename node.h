/* node.h
 * Author: Lim Chun Yu
 */

#ifndef NODE_H
#define NODE_H

typedef struct Node_t {
    double* biasWeights;
    double* lr;
    double* activatedVal;
    double maeVal;
} Node_t;

double* initBiasWeights(int);

#endif // NODE_H
