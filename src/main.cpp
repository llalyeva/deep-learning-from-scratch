#include <memory>
#include <vector>
#include <chrono>
#include <boost/timer/timer.hpp>

#include "algebra/matrix.h"
#include "algebra/operations.h"
#include "dataset/dataset.h"
#include "model/model.h"
#include "layers/dense.h"
#include "layers/selu.h"
#include "layers/softmax.h"
#include "loss/categoricalCrossentropy.h"

int main() {
    auto timer_start = std::chrono::high_resolution_clock::now();

    // Dataset configuration:
    unsigned short int seed = 0;
    size_t samplesCount = 60000;
    size_t testSamplesCount = 10000;
    bool shuffleDataset = true;

    // Network configuration:
    size_t epochs = 20;
    size_t featuresCount = 784;
    size_t batchSize = 32;
    size_t neurons = 64;

    // Backpropagation configuration:
    float l2Lambda = 0.02f;
    float learningRate = 0.001f;
    float momentum = 0.95f;
    bool doGradClip = true;

    // Loading dataset from csv files:
    Matrix X = datasetFromFile("data/fashion_mnist_train_vectors.csv", samplesCount, featuresCount, seed, shuffleDataset);
    Matrix labels = labelsVectorFromFile("data/fashion_mnist_train_labels.csv", samplesCount, seed, shuffleDataset);
    Matrix Y = toCategoricalLabels(labels, 10);  // Apply one-hot encoding
    X = X.rescale(255.0f);  // Scale each pixel by 255

    // Devide dataset to train(90%) and validation(10%):
    Matrix trainX = getSlice(X, 0.0f, 0.9f);
    Matrix trainY = getSlice(Y, 0.0f, 0.9f);
    Matrix valX = getSlice(X, 0.9f, 1.0f);
    Matrix valY = getSlice(Y, 0.9f, 1.0f);

    // Construct the network:
    Model model(epochs, batchSize, learningRate, momentum, doGradClip);
    model.setLoss(std::make_unique<CategoricalCrossentropyLoss>());

    model.addLayer(std::make_unique<DenseLayer>(784, neurons, l2Lambda));
    model.addLayer(std::make_unique<SELUActivation>());

    model.addLayer(std::make_unique<DenseLayer>(neurons, 10, l2Lambda));
    model.addLayer(std::make_unique<SELUActivation>());

    model.addLayer(std::make_unique<SoftmaxActivation>());

    // Start training process:
    model.fit(trainX, trainY, valX, valY);

     // Load test dataset from csv file:
    Matrix testX = datasetFromFile("data/fashion_mnist_test_vectors.csv", testSamplesCount, featuresCount, 0, false);
    Matrix labelsTest = labelsVectorFromFile("data/fashion_mnist_test_labels.csv", testSamplesCount, 0, false);
    Matrix testY = toCategoricalLabels(labelsTest, 10);
    testX = testX.rescale(255.0f);

    Matrix seqTrainX = datasetFromFile("data/fashion_mnist_train_vectors.csv", samplesCount, featuresCount, 0, false);
    Matrix seqLabelsTrain = labelsVectorFromFile("data/fashion_mnist_train_labels.csv", samplesCount, 0, false);
    Matrix seqTrainY = toCategoricalLabels(seqLabelsTrain, 10);
    seqTrainX = seqTrainX.rescale(255.0f);

    // Evaluate model:
    const std::vector<int> testPredictions = model.evaluate(testX, testY);
    const std::vector<int> trainPredictions = model.evaluate(seqTrainX, seqTrainY);

    // Save results to the csv files:
    saveVectorToCSV(testPredictions, "test_predictions.csv");
    saveVectorToCSV(trainPredictions, "train_predictions.csv");

    auto timer_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(timer_end - timer_start).count();

    std::cout << "===[TOTAL RESULTS]===" << std::endl;
    std::cout << "Total time: " << duration << " seconds" << std::endl;

    return 0;
}