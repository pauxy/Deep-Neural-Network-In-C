/* mlp.h
 * Author: Lim Chun Yu
 */

#ifndef MLP_H
#define MLP_H

typedef struct Node_t {
    double* biasWeights;
    double* lr;
    double* activatedVal;
    double maeVal;
} Node_t;

typedef struct Layer_t {
    Node_t* nodes;
    int numOfNodes;
} Layer_t;

Layer_t genLayer(int);
double* initBiasWeights(int);

#endif // MLP_H
