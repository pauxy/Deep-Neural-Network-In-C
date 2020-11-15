/* node.h
 * Author: Lim Chun Yu
 */

#ifndef NODE_H
#define NODE_H

typedef struct Node_h {
    double* biasWeights;
    double* lr;
    double* activatedVal;
    double maeVal;
} Node_h;

#endif // NODE_H
