/* mlp.h
 * Author: Lim Chun Yu
 */

#ifndef MLP_H
#define MLP_H

#include "dataparser.h"

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
} Layer_t;

Layer_t genLayer(int , int , double, InputOutput_t );
BiasWeights_t initBiasWeights(int);
void trainNodes(Node_t* , int, double, InputOutput_t );

#endif // MLP_H
