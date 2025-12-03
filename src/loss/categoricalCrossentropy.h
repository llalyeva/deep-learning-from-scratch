#ifndef CATEGORICAL_CROSSENTROPY_H
#define CATEGORICAL_CROSSENTROPY_H

#include "algebra/matrix.h"
#include "loss/loss.h"

class CategoricalCrossentropyLoss: public ILoss {
private:
    float loss;

public:
    float calculate(Matrix& yTrue, Matrix& yPred) override;

    Matrix partialDerivative(Matrix& yTrue, Matrix& yPred) override;
};

#endif