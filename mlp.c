/* mlp.c -- 
 * Author: Lim Chun Yu
 */

#include <stdlib.h>

#include "mlp.h"
#include "node.h"

Layer_t genLayer(int nodes) {
    Layer_t layer;
    layer.numOfNodes = nodes;
    layer.nodes = (Node_t*)malloc(nodes * sizeof(Node_t));
    for (int i = 0; i < nodes; i++) {
        (layer.nodes + i)->biasWeights = initBiasWeights(nodes);
    }

    return layer;
}
