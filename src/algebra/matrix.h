#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>


class Matrix {
private:
    std::vector<float> data;
    size_t rows, cols;
    size_t index(size_t i, size_t j) const;

public:
    Matrix(size_t r = 1, size_t c = 1);

    size_t getRows() const;

    size_t getCols() const;

    void set(size_t i, size_t j, float value);

    float get(size_t i, size_t j) const;

    const std::vector<float>& getData() const;

    void setData(const std::vector<float>& newData);

    void setData(std::vector<float>&& newData);

    float* rawData();

    const float* rawData() const;

    Matrix operator+(const float& scalar) const;

    Matrix operator+(const Matrix& other) const;

    Matrix operator-(const float& scalar) const;

    Matrix operator-(const Matrix& other) const;

    Matrix operator*(const float& scalar) const;

    Matrix operator*(const Matrix& other) const;

    Matrix T() const;

    Matrix getRow(size_t i) const;

    Matrix getRowsSlice(size_t start, size_t end) const;

    float sum() const;

    Matrix rescale (float factor) const;

    Matrix rescaleR (float factor) const;

    Matrix hadamardProduct(const Matrix& other) const;
};


#endif