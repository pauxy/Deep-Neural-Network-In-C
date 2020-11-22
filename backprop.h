/* backprop.h
 * Author: Lim Chun Yu
 */

#ifndef BACKPROP_H
#define BACKPROP_H

#include "mlp.h"

BiasWeights_t backwardsPropagation(Inputoutput_t, BiasWeights_t, double*, double*, int, int);

#endif // BACKPROP_H
