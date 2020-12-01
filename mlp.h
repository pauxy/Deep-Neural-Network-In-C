/* mlp.h
 * Author: Lim Chun Yu
 */

#ifndef MLP_H
#define MLP_H

#include "dataparser.h"
#include <stdio.h>

typedef struct BiasWeights_t {
    double* weights;
    double bias;
} BiasWeights_t;

typedef struct Node_t {
    BiasWeights_t biasWeights;
    double* muladd;
    double* activatedVal;
    int connections;
} Node_t;

typedef struct Layer_t {
    Node_t* nodes;
    int numOfNodes;
    double** layerOutput;
    struct Layer_t* prev;
} Layer_t;

Layer_t genLayer(int, int, Layer_t*);
BiasWeights_t initBiasWeights(int);
void trainNodes(Node_t* , int, InputOutput_t );
Node_t* trainNetwork(int, int*, InputOutput_t*, double ,FILE*);

#endif // MLP_H
