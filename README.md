# Deep Learning from Scratch (C++ / Fashion-MNIST)

  This project implements a fully-connected neural network **from first principles in modern C++**.  
No deep learning frameworks are used – everything from matrix operations to backpropagation is coded manually.

The network is trained and evaluated on the **Fashion-MNIST** dataset (Zalando’s 28×28 grayscale clothing images) in CSV format.

---

## Project Goals

The original task:

- Implement a neural network in a low-level programming language  
  (**C/C++/Java/C#/Rust**) **without** advanced ML/DL libraries.
- Train it on the provided **Fashion-MNIST** dataset using **backpropagation**.
- Implement the training pipeline, including:
  - forward pass,
  - loss computation,
  - backpropagation,
  - optimization / regularization,
  - evaluation and result export.

This repository is a C++ implementation of that task.

---

## Dataset

The project uses **Fashion-MNIST**:

- 60 000 training images + 10 000 test images.
- Each image is **28×28 grayscale** → flattened to **784 features**.
- 10 classes (T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot, etc.).
- In this project, the dataset is expected as **CSV**:

```text
data/
  fashion_mnist_train_vectors.csv   # 60 000 × 784, pixel values [0, 255]
  fashion_mnist_test_vectors.csv    # 10 000 × 784, pixel values [0, 255]
  fashion_mnist_train_labels.csv    # 60 000 × 1, class index [0–9]
  fashion_mnist_test_labels.csv     # 10 000 × 1, class index [0–9]
```

During preprocessing:

* Images are **normalized** by dividing pixel values by `255.0`.
* Labels are converted to **one-hot encoded vectors** of length 10.
* Training data is split into:

  * **90% train**
  * **10% validation** (stratification isn’t enforced, simple slicing is used).

---

## High-Level Architecture

At a glance:

* `algebra/` – a small linear algebra layer:

  * `Matrix` type (row-major, contiguous memory),
  * SIMD-accelerated matrix multiplication,
  * dot product, Hadamard product, norms, etc.
* `dataset/` – data loading and preprocessing:

  * CSV parsing,
  * random shuffling,
  * one-hot encoding,
  * slicing utilities.
* `layers/` – neural network building blocks:

  * `DenseLayer` – fully connected layer,
  * `SELUActivation` – SeLU nonlinearity,
  * `SoftmaxActivation` – softmax output layer.
* `loss/` – loss functions:

  * `CategoricalCrossentropyLoss` for multi-class classification.
* `model/` – training orchestration:

  * layer stack (`std::vector<std::unique_ptr<ILayer>>`),
  * forward pass through all layers,
  * backpropagation through all layers,
  * SGD with momentum, L2 regularization, gradient clipping, basic LR decay.
* `main.cpp` – wiring everything together:

  * loads data,
  * constructs the model,
  * trains, evaluates,
  * exports predictions.

---

## Key Concepts & Implementation Details

### 1. Custom Matrix Implementation

**Files:** `src/algebra/matrix.{h,cpp}`

* `Matrix` is a simple wrapper around `std::vector<float>` with explicit `rows` and `cols`.
* Supports:

  * basic arithmetic: `+`, `-`, `*` (scalar & matrix),
  * transpose `T()`,
  * row extraction and slicing (`getRow`, `getRowsSlice`),
  * element-wise operations (Hadamard product),
  * sum and rescaling.

**Performance detail:**

* `Matrix::operator*(const Matrix& other)` uses **AVX (256-bit) intrinsics**:

  * Loads 8 floats at a time with `_mm256_loadu_ps`,
  * Fused multiply-add with `_mm256_fmadd_ps`,
  * Accumulates tail elements scalar-wise.
* This requires building with AVX2 support, e.g. `-mavx2 -mfma` with GCC/Clang.

### 2. Algebraic Operations

**Files:** `src/algebra/operations.{h,cpp}`

Features:

* `dot(A, B)`:

  * If `A` and `B` have identical shape → elementwise dot product (flattened) → scalar 1×1 matrix.
  * Otherwise, falls back to **matrix multiplication** using `A * B`.
* `matrixLog`:

  * Applies a numerically stable `ln` (`stableLn`) to each element.
* `normalLeCunInit`:

  * Initializes weights with **LeCun normal initialization**:

    * values ~ `N(0, 1 / fan_in)`
    * works well with SeLU / self-normalizing nets.
* `computeNorm` + `clipGradient`:

  * Computes **L2 norm** of a gradient matrix.
  * Optionally **clips gradients** if norm exceeds `maxNorm` (to mitigate exploding gradients).
* `argmax`:

  * Returns index of the maximal element in a vector.

### 3. Dataset Pipeline

**Files:** `src/dataset/dataset.{h,cpp}`

* `datasetFromFile`:

  * Reads CSV of features into a `Matrix(samplesCount, featuresCount)`.
  * Optionally shuffles rows using a provided seed.
* `labelsVectorFromFile`:

  * Reads labels (one integer per line) into a column vector `Matrix(samplesCount, 1)`.
  * Supports the same shuffling mechanism to keep inputs and labels aligned.
* `toCategoricalLabels`:

  * Converts label vector to **one-hot encoded** matrix `samplesCount × numClasses`.
* `getSlice`:

  * Returns a slice of the dataset according to relative rates `[fromRate, toRate]`.
  * Used for train/validation split.
* `saveVectorToCSV`:

  * Saves prediction vectors (e.g. class indices) to disk (one prediction per line).

### 4. Layers & Activations

All layers implement the same interface:

```cpp
class ILayer {
public:
    virtual ~ILayer() {}
    virtual Matrix forward(Matrix& Y) = 0;
    virtual Matrix backward(Matrix& dOut, float learningRate, float momentum) = 0;
    virtual float getL2RegularizationLoss() = 0;
};
```

This enables a simple chain of layers in `Model`.

#### DenseLayer

**Files:** `src/layers/dense.{h,cpp}`

* Parameters:

  * `W` – weight matrix (initialized with LeCun normal),
  * `wVelosity` – for **momentum** in SGD,
  * `l2Lambda` – L2 regularization coefficient.

* `forward(Y)`:

  * Stores the input.
  * Computes `W * Y`.

* `backward(outputGradient, learningRate, momentum)`:

  * Computes local weight gradient: `dW = outputGradient * input^T`.
  * Updates velocity: `v = momentum * v + dW`.
  * Gradient descent step: `W = W - learningRate * v`.
  * Returns gradient for previous layer: `dInput = W^T * outputGradient`.

* `getL2RegularizationLoss()`:

  * Returns `(λ / 2) * Σ W_ij²`.

#### SELUActivation

**Files:** `src/layers/selu.{h,cpp}`

Implements **SeLU (Scaled Exponential Linear Unit)**:

* For input `x`:

  * `x > 0` → scaled linear region
  * `x ≤ 0` → scaled exponential region
* Forward pass:

  * Applies SeLU element-wise.
* Backward pass:

  * Computes derivative per element and multiplies by `outputGradient`.
* L2 regularization:

  * Returns `0.0f` (no regularization on activations).

#### SoftmaxActivation

**Files:** `src/layers/softmax.{h,cpp}`

* Forward pass:

  * For each sample, computes normalized exponentials:

    * `softmax_i = exp(x_i) / Σ_j exp(x_j)`
  * Stores the probabilities in `innerPot`.
* Backward pass:

  * Computes the **Jacobian of softmax**:

    * Diagonal: `s_i * (1 - s_i)`
    * Off-diagonal: `- s_i * s_j`
  * Multiplies by `outputGradient` to obtain `dInput`.
* L2 regularization:

  * Returns `0.0f`.

> Note: In practice, softmax + cross-entropy has a simplified derivative, which this project also uses in the loss (see below).

### 5. Loss Function

**Files:** `src/loss/categoricalCrossentropy.{h,cpp}`, `src/loss/loss.h`

Loss interface:

```cpp
class ILoss {
public:
    virtual ~ILoss() {}
    virtual float calculate(Matrix& yTrue, Matrix& yPred) = 0;
    virtual Matrix partialDerivative(Matrix& yTrue, Matrix& yPred) = 0;
};
```

**CategoricalCrossentropyLoss**:

* `calculate(yTrue, yPred)`:

  * Uses **categorical cross-entropy**:

    * `L = -1/N Σ_n Σ_k y_true[n,k] * log(y_pred[n,k] + ε)`
  * `ε = 1e-10` for numerical stability.
* `partialDerivative(yTrue, yPred)`:

  * For softmax outputs this simplifies to:

    * `∂L/∂z = yPred - yTrue`
  * Returns `yPred - yTrue`.

---

### 6. Training Loop & Optimization

**Files:** `src/model/model.{h,cpp}`, `src/main.cpp`

**Model configuration (default in `main.cpp`):**

* Epochs: `20`
* Batch size: `32`
* Hidden units (first dense layer): `64`
* Learning rate: `0.001`
* Momentum: `0.95`
* L2 regularization (`λ`): `0.02`
* Gradient clipping: enabled (`maxNorm = 3.0f`)

**Training (`Model::fit`)**:

1. Loop over epochs.
2. For each epoch:

   * Iterate mini-batches over training set.
   * For each sample in batch:

     * Forward pass through all layers (`Model::predict`).
     * Accumulate training loss.
     * Compute loss gradient (`loss->partialDerivative`).
     * Backpropagate through layers in reverse order:

       * Optionally apply **gradient clipping** before each layer.
       * Update weights with **SGD + momentum**.
   * Add **L2 regularization** term to batch loss.
3. Evaluate on validation set:

   * Compute validation loss and accuracy.
4. **Learning rate decay**:

   * Track validation loss.
   * If loss does not improve for 3 consecutive epochs, multiply `learningRate` by `0.5`.

During training, a formatted table is printed:

* epoch number,
* train loss,
* train accuracy (%),
* validation loss,
* validation accuracy (%),
* current learning rate.

**Evaluation (`Model::evaluate`)**:

* Runs forward passes on each sample in batches.
* Computes accuracy.
* Returns vector of predicted class indices.
* Prints evaluation metrics and total evaluation time.

---

## Running the Project

### Requirements

* A C++17-capable compiler (GCC, Clang, MSVC).
* AVX2 support (for SIMD matrix multiplication) and corresponding compiler flags.
* [Boost] is included via `#include <boost/timer/timer.hpp>` in `main.cpp` (currently only `std::chrono` is used; you can remove Boost if not needed).
* Fashion-MNIST CSV files in the `data/` directory, as described above.

### Example Build (GCC / Clang)

From the project root:

```bash
# Example: build with optimization and AVX2 support
g++ -std=c++17 -O3 -mavx2 -mfma \
    -Isrc \
    $(find src -name '*.cpp') \
    -o nn_from_scratch
```

Adjust flags and include paths to match your environment or integrate this into your own CMake / build system.

### Run

```bash
./nn_from_scratch
```

What happens:

1. Training data is loaded, normalized, and split into train/validation.

2. The model is built and trained for the configured number of epochs.

3. Test data is loaded and evaluated.

4. Predictions are saved as CSV:

   ```text
   test_predictions.csv   # predictions for 10 000 test samples
   train_predictions.csv  # predictions for 60 000 training samples (non-shuffled)
   ```

5. Training, validation, and evaluation statistics are printed to stdout.

---

## Tuning & Experimentation

Most hyperparameters are configured in `src/main.cpp`:

* **Network architecture**

  * Number of hidden units: change `neurons`.
  * Add/remove layers by adjusting `model.addLayer(...)` calls.

* **Optimization**

  * `epochs`
  * `batchSize`
  * `learningRate`
  * `momentum`
  * `doGradClip` & clip threshold (in `Model::fit`)

* **Regularization**

  * `l2Lambda` in `DenseLayer` constructors.

You can experiment with:

* deeper or wider networks,
* different activation functions (ReLU, tanh, etc.),
* alternative initialization schemes,
* custom learning rate schedules.

---

## Extending the Codebase

Here are some natural next steps:

* **New layers**
  Implement a new layer by inheriting from `ILayer` and plugging it into `Model::layers`.

* **New losses**
  Implement custom loss functions by inheriting from `ILoss`.

* **Improved numerics**

  * Use per-sample softmax rather than across the entire matrix.
  * Add log-sum-exp trick for improved stability.

* **Experiment tracking**

  * Save training curves to file,
  * Log metrics to external tools.

---

## References

* Fashion-MNIST: `Xiao, Han, et al. "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms."`
* LeCun initialization and SeLU activation are inspired by the **self-normalizing neural networks** literature.
