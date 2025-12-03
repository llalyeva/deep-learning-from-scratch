#include <algorithm>
#include <iostream>
#include <random>
#include <fstream>
#include <vector>

#include "algebra/matrix.h"
#include "dataset/dataset.h"

Matrix datasetFromFile(std::string filename, size_t samplesCount, size_t featuresCount, unsigned short int seed, bool shuffle) {
    /* Creates a dataset from the csv file as a 2D matrix. */

    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::size_t> dist(0, samplesCount - 1);

    std::ifstream file(filename);
    std::string c;
    Matrix dataset(samplesCount, featuresCount);

    if (file.fail()) {
        throw std::runtime_error("[dataset.datasetFromFile] invalid file requested");
    }

    // Convert each line of file to a vector of floats and add it to the dataset.
    int rows = 0;
    while (std::getline(file, c) && rows < samplesCount) {
        std::string numberFromLine;
        size_t targetRow = shuffle ? dist(rng) : rows;  // Shuffle or load sequentially
        size_t cols = 0;

        // Iterate each character of line to get separated by comma floats.
        for (size_t i = 0; i < c.size(); ++i) {
            if (c[i] == ',') {
                dataset.set(targetRow, cols, stof(numberFromLine));  // Convert number to float
                numberFromLine.clear();  // Clear number buffer
                cols++;
            } else {
                numberFromLine.push_back(c[i]);  // Gather digits
            }
        }
        rows++;
    }

    return dataset;
}

Matrix labelsVectorFromFile(std::string filename, size_t samplesCount, unsigned short int seed, bool shuffle) {
    /* Creates a vector from file with labels. */

    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::size_t> dist(0, samplesCount - 1);

    std::ifstream file(filename);
    std::string c;
    Matrix labelsVector(samplesCount, 1);

    // Return an empty vector if file is damaged:
    if (file.fail()) {
        throw std::runtime_error("[dataset.labelsVectorFromFile] invalid file requested");
    }

    // Convert each line of file to integer representing a label:
    size_t rows = 0;
    while (std::getline(file, c) && rows < samplesCount) {
        size_t targetRow = shuffle ? dist(rng) : rows;  // Shuffle or load sequentially
        labelsVector.set(targetRow, 0, stoi(c));  // Add label to labels vector
        rows++;
    }

    return labelsVector;
}

Matrix toCategoricalLabels (Matrix& labelsVector, size_t numOfClasses) {
    /* Applies one-hot encoding to the labels. */

    Matrix categorizedLabels(labelsVector.getRows(), numOfClasses);

    for (size_t i = 0; i < labelsVector.getRows(); ++i) {
        for (size_t j = 0; j < numOfClasses; ++j) {
            if (j == labelsVector.get(i, 0)) {
                categorizedLabels.set(i, j, 1);
            } else {
                categorizedLabels.set(i, j, 0);
            }
        }
    }

    return categorizedLabels;
}

Matrix getSlice(Matrix& dataset, float fromRate, float toRate) {
    if (fromRate >= toRate) {
        throw std::invalid_argument("[dataset.getSlice] fromRate should be less than toRate");
    }

    size_t rows = dataset.getRows();
    size_t cols = dataset.getCols();
    size_t fromIndex = rows * fromRate;
    size_t toIndex = rows * toRate - 1;
    size_t sRows = toIndex - fromIndex + 1;
    Matrix slicedDataset(sRows, cols);

    for (size_t i = 0; i < sRows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            slicedDataset.set(i, j, dataset.get(i + fromIndex, j));
        }
    }

    return slicedDataset;
}

void saveVectorToCSV(const std::vector<int>& vec, std::string filename) {
    std::ofstream outFile(filename, std::ios::trunc);
    if (outFile.fail()) {
        throw std::runtime_error("[dataset.saveVectorToCSV] invalid file requested");
    }

    // Write each element of the vector to the file
    for (const int& value : vec) {
        outFile << value << '\n';
    }

    outFile.close();
}