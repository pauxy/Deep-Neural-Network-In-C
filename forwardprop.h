/* forwardprop.h
 * Author: Lim Chun Yu
 */

#ifndef FORWARDPROP_H
#define FORWARDPROP_H

#include "mlp.h"

double* linearRegression(double**, BiasWeights_t, int, int);
double* sigmoid(double*, int);

#endif // FORWARDPROP_H
