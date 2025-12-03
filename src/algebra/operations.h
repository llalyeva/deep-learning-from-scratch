#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "algebra/matrix.h"

bool isColumnVector(const Matrix& vector);

bool isRowVector(const Matrix& vector);

Matrix dot (const Matrix& A, const Matrix& B);

Matrix matrixLog(const Matrix& A);

Matrix normalLeCunInit(int outputSize, int inputSize);

float stableLn(float value);

float computeNorm(const Matrix& gradient);

Matrix clipGradient(const Matrix& gradient, float maxNorm);

int argmax(const Matrix& vector);

#endif