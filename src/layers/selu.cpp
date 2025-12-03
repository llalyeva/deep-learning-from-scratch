#include <cmath>
#include "algebra/matrix.h"
#include "algebra/operations.h"
#include "layers/selu.h"
#include "layers/layer.h"

Matrix SELUActivation::forward(Matrix& Y) {
    /* Forward pass of the SeLU activation function. */
    
    // Constants used in the calculation of SeLU:
    const float alpha = 1.0507;
    const float lambda = 1.6733;

    this->input = Y;

    size_t rows = Y.getRows();
    size_t cols = Y.getCols();
    const std::vector<float>& inputData = Y.getData();
    std::vector<float> innerPotData(rows * cols);

    // Apply SeLU to the input matrix:
    for (size_t i = 0; i < rows * cols; ++i) {
        float value = inputData[i];
        if (value > 0) {
            innerPotData[i] = alpha * value;
        } else {
            innerPotData[i] = alpha * lambda * (std::exp(value) - 1);
        }
    }

    Matrix result(rows, cols);
    result.setData(std::move(innerPotData));  // small optimization (probably not good idea)
    return result;
}

Matrix SELUActivation::backward(Matrix& outputGradient, float learningRate, float momentum) {
    /* Backward pass of the SeLU activation function. */

    // Constants used in the calculation of SeLU:
    const float alpha = 1.0507;
    const float lambda = 1.6733;

    float epsilon = 1e-10;  // Applied for numerical stability

    size_t rows = this->input.getRows();
    size_t cols = this->input.getCols();
    const std::vector<float>& inputData = this->input.getData();
    const std::vector<float>& outputGradData = outputGradient.getData();
    std::vector<float> gradInputData(rows * cols);

    for (size_t i = 0; i < rows * cols; ++i) {
        float value = inputData[i] + epsilon;  // Add epsilon for numerical stability
        float grad = (value > 0) ? lambda : (alpha * lambda * std::exp(value));
        gradInputData[i] = grad * outputGradData[i];
    }

    Matrix result(rows, cols);
    result.setData(std::move(gradInputData)); // small optimization (probably not good idea)
    return result;
}

float SELUActivation::getL2RegularizationLoss() { return 0.0f; };