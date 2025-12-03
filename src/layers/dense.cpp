#include "algebra/matrix.h"
#include "algebra/operations.h"
#include "layers/dense.h"
#include "layers/layer.h"

DenseLayer::DenseLayer(size_t inputSize, size_t outputSize, float l2Lambda) {
    this->W = normalLeCunInit(outputSize, inputSize);  // Normal LeCun weight initialization (suitable for SeLU)
    this->wVelosity = Matrix(outputSize, inputSize);  // Used to properly apply momentum
    this->l2Lambda = l2Lambda;
}

Matrix DenseLayer::forward(Matrix& Y) {
    /* Forward pass of the Dense layer. */

    // Inner Potential = W * Y
    this->input = Y;
    return dot(W, input);
}

Matrix DenseLayer::backward(Matrix& outputGradient, float learningRate, float momentum) {
    /* Backward pass of the Dense layer. */

    Matrix inputT = this->input.T();

    Matrix wGradient = dot(outputGradient, inputT);

    // Apply momentum:
    this->wVelosity = this->wVelosity * momentum + wGradient;

    // Gradient descent:
    this->W = this->W - this->wVelosity * learningRate;

    // Calculate gradient for the next layer:
    return dot(this->W.T(), outputGradient);
}


float DenseLayer::getL2RegularizationLoss() {
    /* Calculate L2 regularization term. */

    float l2Loss = 0.0f;

    // Sum of weight squares:
    for (size_t i = 0; i < this->W.getRows(); ++i) {
        for (size_t j = 0; j < this->W.getCols(); ++j) {
            l2Loss += this->W.get(i, j) * this->W.get(i, j);
        }
    }

    return (this->l2Lambda / 2.0f) * l2Loss;
}