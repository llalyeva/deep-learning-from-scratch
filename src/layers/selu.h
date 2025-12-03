#ifndef SELU_H
#define SELU_H

#include "algebra/matrix.h"
#include "layers/layer.h"

class SELUActivation : public ILayer {
private:
    Matrix input;

public:
    Matrix forward(Matrix& Y) override;

    Matrix backward(Matrix& outputGradient, float learningRate, float momentum) override;

    float getL2RegularizationLoss() override;
};


#endif