#ifndef LOSS_H
#define LOSS_H

#include "algebra/matrix.h"

class ILoss {
public:
    virtual ~ILoss() {}
    virtual float calculate(Matrix& yTrue, Matrix& yPred) = 0;
    virtual Matrix partialDerivative(Matrix& yTrue, Matrix& yPred) = 0;
};

#endif