/* backprop.h
 * Author: Lim Chun Yu
 */

#ifndef BACKPROP_H
#define BACKPROP_H

double meanAbsoluteValue(double**, double*, int);
double* backwardsPropagation(double**, double*, double*, double*);

#endif // BACKPROP_H
