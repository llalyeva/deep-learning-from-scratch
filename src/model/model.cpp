#include <iomanip>
#include <vector>
#include <memory>
#include <chrono>

#include "algebra/matrix.h"
#include "algebra/operations.h"
#include "layers/dense.h"
#include "layers/layer.h"
#include "loss/loss.h"
#include "model/model.h"


Model::Model(size_t epochs, size_t batchSize, float learningRate, float momentum, bool doGradClip) {
    this->epochs = epochs;
    this->batchSize = batchSize;
    this->learningRate = learningRate;
    this->momentum = momentum;
    this->doGradClip = doGradClip;
}

void Model::setLoss(std::unique_ptr<ILoss> newLoss) { this->loss = std::move(newLoss); }

void Model::addLayer(std::unique_ptr<ILayer> layer) { layers.push_back(std::move(layer)); }

void Model::fit(Matrix& xTrain, Matrix& yTrain, Matrix& xVal, Matrix& yVal) {
    /*
    Iteratively fits the model to the provided dataset.

    Envolved techniques:
    SGD optimizator, L2 regularization, momentum, gradient clipping
    */
    
    if (xTrain.getRows() != yTrain.getRows()) {
        throw std::invalid_argument("[Model.fit] xTrain and yTrain rows count must be the same");
    }
    if (xVal.getRows() != yVal.getRows()) {
        throw std::invalid_argument("[Model.fit] xVal and yVal rows count must be the same");
    }
    if (loss == nullptr) {
        throw std::invalid_argument("[Model.fit] loss function has not been set");
    }

    auto timer_start = std::chrono::high_resolution_clock::now();

    int stagnationCount = 0;
    float decayFactor = 0.5f;
    float previousLoss = std::numeric_limits<float>::max();
    size_t trainSamples = xTrain.getRows();
    size_t valSamples = xVal.getRows();
    size_t numBatches = (trainSamples + this->batchSize - 1) / this->batchSize;

    std::cout << "===[TRAIN RESULTS]===" << std::endl;
    std::cout << std::left 
              << std::setw(10) << "epoch" 
              << std::setw(20) << "trainLoss" 
              << std::setw(20) << "trainAccuracy(%)" 
              << std::setw(20) << "valLoss" 
              << std::setw(20) << "valAccuracy(%)"
              << std::setw(20) << "learningRate"
              << std::endl;

    std::cout << std::string(110, '=') << std::endl;

    // Iterate trough each epoch and fit dataset batch-by-batch
    for (size_t e = 0; e < this->epochs; ++e) {
        float trainLoss = 0.0;
        unsigned int trainPredictedAmount = 0;
        unsigned int valPredictedAmount = 0;

        for (size_t b = 0; b < numBatches; ++b) {
            float batchLoss = 0.0;
            size_t start = b * this->batchSize;
            size_t end = std::min(start + this->batchSize, trainSamples);

            Matrix xBatch = xTrain.getRowsSlice(start, end);
            Matrix yBatch = yTrain.getRowsSlice(start, end);

            for (size_t k = 0; k < xBatch.getRows(); ++k) {
                Matrix Xk = xBatch.getRow(k);  // Single image (784 pixels)
                Matrix Yk = yBatch.getRow(k);  // True category (one-hot encoded)

                // Forward pass
                Matrix output = this->predict(Xk);

                int trueArgmax = argmax(Yk);
                int predArgmax = argmax(output);
                if (trueArgmax == predArgmax) {
                    ++trainPredictedAmount;
                }

                batchLoss += this->loss->calculate(Yk, output);

                // Backpropagation loop
                Matrix gradient = this->loss->partialDerivative(Yk, output);
                for (int i = this->layers.size() - 1; i >= 0; --i) {
                    if (this->doGradClip) {
                        // Apply gradient clip to reduce exploding
                        gradient = clipGradient(gradient, 3.0f);
                    }
                    gradient = this->layers[i]->backward(gradient, this->learningRate, this->momentum);
                }
            }

            // Apply L2 regularization
            batchLoss += this->getL2RegularizationLoss();
            trainLoss += batchLoss / xBatch.getRows();
        }

        trainLoss /= numBatches;

        // Validation loop
        float valLoss = 0.0;
        for (size_t k = 0; k < valSamples; ++k) {
            Matrix Xk = xVal.getRow(k);
            Matrix Yk = yVal.getRow(k);

            Matrix output = this->predict(Xk);

            int trueArgmax = argmax(Yk);
            int predArgmax = argmax(output);
            if (trueArgmax == predArgmax) {
                ++valPredictedAmount;
            }

            valLoss += loss->calculate(Yk, output);
        }
        valLoss /= valSamples;

        float trainAccuracy = 100.0f * trainPredictedAmount / trainSamples;
        float valAccuracy = 100.0f * valPredictedAmount / valSamples;

        // Reduce learning rate when loss has stopped decreasing.
        if (valLoss < previousLoss) {
            stagnationCount = 0;
            previousLoss = valLoss;
        } else {
            stagnationCount++;
        }
        if (stagnationCount >= 3) {
            this->learningRate *= decayFactor;
            stagnationCount = 0;
        }

        std::cout << std::left 
                  << std::setw(10) << e + 1 
                  << std::setw(20) << trainLoss 
                  << std::setw(20) << trainAccuracy 
                  << std::setw(20) << valLoss 
                  << std::setw(20) << valAccuracy
                  << std::setw(20) << this->learningRate
                  << std::endl;
    }

    auto timer_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(timer_end - timer_start).count();

    std::cout << "===[TRAIN RESULTS]===" << std::endl;
    std::cout << "Total training time: " << duration << " seconds" << std::endl;
}

std::vector<int> Model::evaluate(Matrix& X, Matrix& Y) {
    /* Evaluates the model on the provided dataset and returns predictions. */

    if (X.getRows() != Y.getRows()) {
        throw std::invalid_argument("[Model.evaluate] X and Y rows count must be the same");
    }

    auto timer_start = std::chrono::high_resolution_clock::now();

    size_t samplesCount = X.getRows();
    size_t numBatches = (samplesCount + this->batchSize - 1) / this->batchSize;
    std::vector<int> predictions;

    unsigned int predictedAmount = 0;

    for (size_t b = 0; b < numBatches; ++b) {
        size_t start = b * this->batchSize;
        size_t end = std::min(start + this->batchSize, samplesCount);

        Matrix xBatch = X.getRowsSlice(start, end);
        Matrix yBatch = Y.getRowsSlice(start, end);

        for (size_t k = 0; k < xBatch.getRows(); ++k) {
            Matrix Xk = xBatch.getRow(k);
            Matrix Yk = yBatch.getRow(k);

            Matrix output = this->predict(Xk);

            int trueArgmax = argmax(Yk);
            int predArgmax = argmax(output);
            if (trueArgmax == predArgmax) {
                ++predictedAmount;
            }

            predictions.push_back(predArgmax);
        }
    }

    float accuracy = 100.0f * predictedAmount / samplesCount;

    auto timer_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(timer_end - timer_start).count();

    std::cout << "===[EVALUATION RESULTS]===" << std::endl;
    std::cout << "Accuracy(%):  " << accuracy << "| Total testing time:   " << duration << " seconds" << std::endl;

    return predictions;
}

Matrix Model::predict(const Matrix& input) {
    /* Returns the model prediction. */

    Matrix output = input;
    for (size_t i = 0; i < this->layers.size(); ++i) {
        output = this->layers[i]->forward(output);
    }
    return output;
}

float Model::getL2RegularizationLoss() {
    /* Aggregates L2 regularization term from each dense layer. */

    float totalL2Loss = 0.0f;
    for (const auto& layer : this->layers) {
        DenseLayer* denseLayer = dynamic_cast<DenseLayer*>(layer.get());
        if (denseLayer) {
            totalL2Loss += denseLayer->getL2RegularizationLoss();
        }
    }
    return totalL2Loss;
}