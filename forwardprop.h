/* forwardprop.h
 * Author: Lim Chun Yu
 */

#ifndef FORWARDPROP_H
#define FORWARDPROP_H

#include "mlp.h"

double* linearRegression(double**, BiasWeights_t, double*, int, int);
double* sigmoid(double*, double*, int);

#endif // FORWARDPROP_H
