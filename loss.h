/* loss.h
 * Author: Lim Chun Yu
 */

#ifndef LOSS_H
#define LOSS_H

double meanAbsoluteValue(double**, double*, int);
double minMeanSquareError(double**, double*, int);
char** confusionMatrix(double**, int*, int);

#endif // LOSS_H
