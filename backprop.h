/* backprop.h
 * Author: Lim Chun Yu
 */

#ifndef BACKPROP_H
#define BACKPROP_H

#include "mlp.h"

double meanAbsoluteValue(double**, double*, int);
BiasWeights_t backwardsPropagation(double**, BiasWeights_t, double*, double*, int, int);

#endif // BACKPROP_H
