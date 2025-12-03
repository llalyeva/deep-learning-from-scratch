#ifndef DENSE_H
#define DENSE_H

#include "algebra/matrix.h"
#include "layers/layer.h"

class DenseLayer : public ILayer {
private:
    Matrix W;
    Matrix input;
    Matrix wVelosity;
    float l2Lambda;

public:
    DenseLayer(size_t inputSize, size_t outputSize, float l2Lambda = 0.01f);

    Matrix forward(Matrix& Y) override;

    Matrix backward(Matrix& outputGradient, float learningRate, float momentum) override;

    float getL2RegularizationLoss() override;
};

#endif