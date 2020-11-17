/* mlp.h
 * Author: Lim Chun Yu
 */

#ifndef MLP_H
#define MLP_H

typedef struct BiasWeights_t {
    double* weights;
    double bias;
} BiasWeights_t;

typedef struct Node_t {
    BiasWeights_t biasWeights;
    double* lr;
    double* activatedVal;
} Node_t;

typedef struct Layer_t {
    Node_t* nodes;
    int numOfNodes;
} Layer_t;

Layer_t genLayer(int);
BiasWeights_t initBiasWeights(int);

#endif // MLP_H
