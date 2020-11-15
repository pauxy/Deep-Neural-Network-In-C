/* mlp.h
 * Author: Lim Chun Yu
 */

#ifndef MLP_H
#define MLP_H

#include "node.h"

typedef struct Layer_t {
    Node_t* nodes;
    int numOfNodes;
} Layer_t;

Layer_t genLayer(int);

#endif // MLP_H
