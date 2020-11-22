/* loss.h
 * Author: Lim Chun Yu
 */

#ifndef LOSS_H
#define LOSS_H

double meanAbsoluteValue(int*, double*, int);
double minMeanSquareError(int*, double*, int);
char** confusionMatrix(int*, int*, int);

#endif // LOSS_H
