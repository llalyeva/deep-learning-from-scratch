#include <algorithm>
#include <cmath>
#include <random>

#include "algebra/matrix.h"
#include "algebra/operations.h"

bool isColumnVector(const Matrix& vector) { return vector.getCols() == 1; }

bool isRowVector(const Matrix& vector) { return vector.getRows() == 1; }

Matrix dot(const Matrix& A, const Matrix& B) {
    /* Calculation of the dot product between matrices */

    size_t Acols = A.getCols();
    size_t Arows = A.getRows();

    if(Acols != B.getCols() || Arows != B.getRows()) {
        // Perform matrix multiplication if dot product is not applicable:
        return A * B;
    }

    // Direct access to underlying data for performance:
    const std::vector<float>& Adata = A.getData();
    const std::vector<float>& Bdata = B.getData();

    // Calculate the dot product:
    float product = 0.0f;
    size_t totalElements = Arows * Acols;

    for (size_t i = 0; i < totalElements; ++i) {
        product += Adata[i] * Bdata[i];
    }

    // Create and set the result matrix:
    Matrix productMatrix(1, 1);
    productMatrix.set(0, 0, product);

    return productMatrix;
}


Matrix matrixLog(const Matrix& A) {
    /* Calculates natural logarithm for each element of the input matrix */

    Matrix logMatrix = Matrix(A.getRows(), A.getCols());

    for (size_t i = 0; i < A.getRows(); ++i) {
        for (size_t j = 0; j < A.getCols(); ++j) {
            float value = stableLn(A.get(i, j));
            logMatrix.set(i, j, value);
        }
    }

    return logMatrix;
}

Matrix normalLeCunInit(int outputSize, int inputSize) {
    /* Noral LeCun weights initialization */

    std::random_device rd;
    std::mt19937 generator(rd());

    float standardDeviation = sqrt(1.0 / inputSize);
    Matrix randomMatrix(outputSize, inputSize);

    std::normal_distribution<float> distribution(0.0, standardDeviation);

    for (size_t r = 0; r < outputSize; ++r) {
        for (size_t c = 0; c < inputSize; ++c) {
            randomMatrix.set(r, c, distribution(generator));
        }
    }

    return randomMatrix;
}

float stableLn(float value) {
    /* Numerical stable natural logarithm. */

    float minThreshold = 1e-10f;
    float clampedValue = std::max(value, minThreshold);
    return std::log(clampedValue);
}

float computeNorm(const Matrix& gradient) {
    /* Calculation of the Euclidean norm */

    float norm = 0.0;
    for (size_t i = 0; i < gradient.getRows(); ++i) {
        for (size_t j = 0; j < gradient.getCols(); ++j) {
            norm += std::pow(gradient.get(i, j), 2);
        }
    }
    return std::sqrt(norm);
}

Matrix clipGradient(const Matrix& gradient, float maxNorm) {
    float norm = computeNorm(gradient);
    if (norm > maxNorm) {
        float scalingFactor = maxNorm / norm;
        return gradient * scalingFactor; // Scale all elements
    }
    return gradient;
}

int argmax(const Matrix& vec) {
    bool rowVector = isRowVector(vec);
    bool colVector = isColumnVector(vec);

    if (!(rowVector || colVector)) {
        throw std::invalid_argument("[operations.argmax] input should be a vector");
    }

    std::vector<float> data = vec.getData();

    return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
}