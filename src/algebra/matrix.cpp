#include <algorithm>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <random>

#include "algebra/matrix.h"

Matrix::Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0f) {}

size_t Matrix::index(size_t i, size_t j) const { return i * cols + j; }

size_t Matrix::getRows() const { return rows; }

size_t Matrix::getCols() const { return cols; }

float* Matrix::rawData() { return this->data.data(); }

const float* Matrix::rawData() const { return this->data.data(); }

void Matrix::setData(const std::vector<float>& newData) {
    if (newData.size() != rows * cols) {
        throw std::invalid_argument("[Matrix.setData] input data size does not match matrix dimensions");
    }
    this->data = newData;
}

void Matrix::setData(std::vector<float>&& newData) {
    if (newData.size() != rows * cols) {
        throw std::invalid_argument("[Matrix.setData] input data size does not match matrix dimensions");
    }
    this->data = std::move(newData);
}

const std::vector<float>& Matrix::getData() const { return this->data; }

Matrix Matrix::operator+(const float& scalar) const {
    /* Matrix + scalar */

    Matrix result(rows, cols);

    for (size_t i = 0; i < this->data.size(); ++i) {
        result.data[i] = this->data[i] + scalar;
    }

    return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
    /* Matrix + Matrix */

    if (other.rows == 1 && other.cols == 1) {
        float scalar = other.data[0];
        return (*this) + scalar;
    } 

    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("[Matrix.operator+] matrix dimensions must match for addition");
    }

    Matrix result(this->rows, this->cols);

    for (size_t i = 0; i < this->data.size(); ++i) {
        result.data[i] = this->data[i] + other.data[i];
    }
    return result;
}

Matrix Matrix::operator-(const float& scalar) const {
    /* Matrix - scalar */

    Matrix result(this->rows, this->cols);

    for (size_t i = 0; i < this->data.size(); ++i) {
        result.data[i] = this->data[i] - scalar;
    }

    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    /* Matrix - Matrix */

    if (other.rows == 1 && other.cols == 1) {
        float scalar = other.data[0];
        return (*this) - scalar;
    }

    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("[Matrix.operator-] matrix dimensions must match for subtraction");
    }

    Matrix result(this->rows, this->cols);

    for (size_t i = 0; i < this->data.size(); ++i) {
        result.data[i] = this->data[i] - other.data[i];
    }
    return result;
}

Matrix Matrix::operator*(const float& scalar) const {
    /* Matrix * scalar */

    Matrix result(this->rows, this->cols);

    for (size_t i = 0; i < this->data.size(); ++i) {
        result.data[i] = this->data[i] * scalar;
    }

    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    /* Optimized matrix multiplication using SIMD vectorization. */

    size_t rowsA = this->rows;
    size_t colsA = this->cols;
    size_t rowsB = other.getRows();
    size_t colsB = other.getCols();

    if (colsA != rowsB) {
        throw std::invalid_argument("[Matrix.operator*] matrix dimensions do not match for multiplication");
    }

    Matrix C(rowsA, colsB);

    const float* Adata = this->rawData();
    const float* Bdata = other.rawData();
    float* Cdata = C.rawData();

    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j) {
            float sum = 0.0f;

            size_t k = 0;
            __m256 acc = _mm256_setzero_ps();
            for (; k + 8 <= colsA; k += 8) {
                __m256 aVec = _mm256_loadu_ps(&Adata[i * colsA + k]);
                __m256 bVec = _mm256_loadu_ps(&Bdata[k * colsB + j]);
                acc = _mm256_fmadd_ps(aVec, bVec, acc);
            }

            float temp[8];
            _mm256_storeu_ps(temp, acc);
            for (int x = 0; x < 8; ++x) {
                sum += temp[x];
            }

            for (; k < colsA; ++k) {
                sum += this->get(i, k) * other.get(k, j);
            }

            Cdata[i * colsB + j] = sum;
        }
    }

    return C;
}

Matrix Matrix::T() const {
    /* Transpose matrix */

    Matrix result(this->cols, this->rows);
    float* resultData = result.data.data();

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            resultData[j * this->rows + i] = data[i * this->cols + j];
        }
    }

    return result;
}

void Matrix::set(size_t i, size_t j, float value) {
    if (i >= rows || j >= cols) {
        throw std::out_of_range("[Matrix.set] matrix index out of bounds");
    }
    data[index(i, j)] = value;
}

float Matrix::get(size_t i, size_t j) const { return data[index(i, j)]; }

Matrix Matrix::getRow(size_t i) const {
    Matrix row(cols, 1);
    std::copy(data.begin() + i * cols, data.begin() + (i + 1) * cols, row.data.begin());
    return row;
}

Matrix Matrix::getRowsSlice(size_t start, size_t end) const {
    if (start >= end) {
        throw std::invalid_argument("[Matrix.getRowsSlice] start index must be less than end index");
    }

    size_t numRows = end - start;
    Matrix slice(numRows, this->cols);

    std::copy(this->data.begin() + start * this->cols, this->data.begin() + end * this->cols, slice.data.begin());

    return slice;
}

float Matrix::sum() const {
    /* Shortcut for std::sum(matrix.getData()) */
    
    return std::reduce(data.begin(), data.end());
}

Matrix Matrix::rescale(float factor) const {
    /* Scale each element of matrix by factor. */

    Matrix rescaledMatrix(this->rows, this->cols);

    for (size_t i = 0; i < this->data.size(); ++i) {
        rescaledMatrix.data[i] = this->data[i] / factor;
    }

    return rescaledMatrix;
}

Matrix Matrix::rescaleR(float factor) const {
    /* Devide factor by each element of matrix. */

    Matrix rescaledMatrix(this->rows, this->cols);

    for (size_t i = 0; i < this->data.size(); ++i) {
        rescaledMatrix.data[i] = factor / this->data[i];
    }

    return rescaledMatrix;
}

Matrix Matrix::hadamardProduct(const Matrix& other) const {
    /* Element-wise product calculation. */

    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("[Matrix.hadamardProduct] matrix dimensions must match for element-wise multiplication");
    }

    Matrix result(this->rows, this->cols);

    for (size_t i = 0; i < this->data.size(); ++i) {
        result.data[i] = this->data[i] * other.data[i];
    }

    return result;
}