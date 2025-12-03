#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "algebra/matrix.h"
#include "layers/layer.h"

class SoftmaxActivation : public ILayer {
private:
    Matrix input;
    Matrix innerPot;

public:
    Matrix forward(Matrix& Y) override;

    Matrix backward(Matrix& outputGradient, float learningRate, float momentum) override;

    float getL2RegularizationLoss() override;
};

#endif