#include "algebra/matrix.h"
#include "algebra/operations.h"
#include "loss/categoricalCrossentropy.h"
#include "loss/loss.h"

float CategoricalCrossentropyLoss::calculate(Matrix& yTrue, Matrix& yPred) {
    /* Calculation of the categorical crossentropy loss */

    float epsilon = 1e-10;

    // -sum(yTrue * ln(yPred)))
    size_t p = yTrue.getRows();
    Matrix logPart = matrixLog(yPred + epsilon);  // Adds epsilon for numerical stability
    Matrix ll = yTrue.hadamardProduct(logPart);
    this->loss = -1 * (ll.sum() / p);
    return this->loss;
}

Matrix CategoricalCrossentropyLoss::partialDerivative(Matrix& yTrue, Matrix& yPred) {
    return yPred - yTrue;  // Categorical crossentropy derivative in the case of softmax activation function
}