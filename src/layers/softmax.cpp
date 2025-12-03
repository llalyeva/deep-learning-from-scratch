#include <cmath>

#include "algebra/matrix.h"
#include "algebra/operations.h"
#include "layers/softmax.h"
#include "layers/layer.h"

Matrix SoftmaxActivation::forward(Matrix& Y) {
    /* Forward pass of the softmax activation function. */

    this->input = Y;
    size_t rows = Y.getRows();
    size_t cols = Y.getCols();

    float denom = 0.0;
    this->innerPot = Matrix(rows, cols);

    // Sum of exponents from each value:
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            denom += exp(Y.get(i, j));
        }
    }

    // Devide exponent of the current value by some of the others:
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            this->innerPot.set(i, j, exp(Y.get(i, j)) / denom);
        }
    }

    return this->innerPot;
}

Matrix SoftmaxActivation::backward(Matrix& outputGradient, float learningRate, float momentum) {
    /* Backward pass of the softmax activation function. */

    size_t rows = this->input.getRows();
    size_t cols = this->input.getCols();

    Matrix gradInput(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float softmax_i = this->innerPot.get(i, j);

            for (size_t k = 0; k < cols; ++k) {
                float softmax_k = this->innerPot.get(i, k);
                // Jacobian matrix calculation:
                float jacobian = (j == k) ? (softmax_i * (1 - softmax_i)) : (-softmax_i * softmax_k);
                // Gradient descent:
                gradInput.set(i, j, gradInput.get(i, j) + jacobian * outputGradient.get(i, k));
            }
        }
    }

    return gradInput;
}

float SoftmaxActivation::getL2RegularizationLoss() {
    /* L2 is not applied in activation functions */
    return 0.0f;
}