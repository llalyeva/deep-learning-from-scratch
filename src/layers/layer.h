#ifndef LAYER_H
#define LAYER_H

#include "algebra/matrix.h"

class ILayer {
private:
    Matrix W;
    Matrix input;

public:
    virtual ~ILayer() {}

    virtual Matrix forward(Matrix& Y) = 0;

    virtual Matrix backward(Matrix& outputGradient, float learningRate, float momentum) = 0;

    virtual float getL2RegularizationLoss() = 0;
};

#endif