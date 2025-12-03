#ifndef DATASET_H
#define DATASET_H

#include <vector>

#include "algebra/matrix.h"

Matrix datasetFromFile (std::string filename, size_t samplesCount, size_t featuresCount,  unsigned short int seed = 0, bool shuffle = false);

Matrix labelsVectorFromFile (std::string filename, size_t samplesCount, unsigned short int seed = 0, bool shuffle = false);

Matrix toCategoricalLabels (Matrix& labelsVector, size_t numOfClasses);

Matrix getSlice(Matrix& dataset, float fromRate, float toRate);

void saveVectorToCSV(const std::vector<int>& vec, std::string filename);

#endif