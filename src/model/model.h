#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <memory>

#include "layers/layer.h"
#include "loss/loss.h"
#include "algebra/matrix.h"

class Model {
private:
    std::unique_ptr<ILoss> loss;
    size_t epochs;
    size_t batchSize;
    float learningRate;
    float momentum;
    bool doGradClip;

    float getL2RegularizationLoss();
public:
    Model(size_t epochs = 10, size_t batchSize = 1, float learningRate = 0.01f, float momentum = 0.0f, bool doGradClip = false);

    std::vector<std::unique_ptr<ILayer>> layers;

    void setLoss(std::unique_ptr<ILoss> newLoss);

    void addLayer(std::unique_ptr<ILayer> layer);

    void fit(Matrix& xTrain, Matrix& yTrain, Matrix& xVal, Matrix& yVal);

    std::vector<int> evaluate(Matrix& xTest, Matrix& yTest);

    Matrix predict(const Matrix& input);
};

#endif