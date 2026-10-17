# Deep Learning Architecture Implementations

PyTorch implementations of three neural network architectures for ten-class
clothing image classification: a multilayer perceptron, a standard CNN, and a
deeper CNN with batch normalization and dropout.

The work is organized in `project.ipynb`, a notebook from the Mathematical
Engineering artificial intelligence course project (`MYZ 309`). It includes
model definitions, training, learning curves, and classification error analysis.
Notebook explanations and chart labels are primarily in Turkish.

## Architectures

| Model | Main layers |
| --- | --- |
| MLP | Flatten 28 x 28 input, 128-unit ReLU hidden layer, 10 output logits |
| Standard CNN | 32- and 64-channel convolution layers with ReLU and max pooling, 128-unit dense layer, 10 output logits |
| Deep CNN | 32/32/64-channel convolutions, batch normalization, pooling, convolutional and dense dropout, 512-unit dense layer, 10 output logits |

The models return logits. Cross-entropy loss handles the classification
objective; softmax is used later when displaying prediction confidence.

## Setup and execution

Use Python 3 with PyTorch support for your platform. The saved notebook metadata
records Python 3.13.2; it does not include a locked dependency environment.
The CPU implementation requires no Azure account, API key, or GPU.

On macOS or Linux:

```sh
git clone https://github.com/civanaban2/deep-learning-example.git
cd deep-learning-example
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab project.ipynb
```

On Windows, create the environment with `py -m venv .venv` and activate it
using `.venv\Scripts\Activate.ps1` in PowerShell before installing packages.

Choose the environment's Python kernel and run the notebook from top to bottom
with the repository root as the working directory. The first cell downloads
Fashion-MNIST into `./data`; network access is required for that download.
The notebook trains all three models for ten epochs each, so a complete run
can take appreciable time on a CPU.

`requirements.txt` lists direct dependencies, not the exact versions used for
the historical saved outputs. Install a compatible PyTorch/torchvision pair;
platform-specific installation instructions are available at
[PyTorch Get Started](https://pytorch.org/get-started/locally/).

## Data and training workflow

- Dataset: 70,000 grayscale images, each 28 x 28, across ten clothing classes.
- Preprocessing: `ToTensor()` scales pixel values into `[0, 1]`.
- Split: concatenate the official training and test datasets, then randomly
  partition them into 49,000 training, 10,500 validation, and 10,500 test images.
- Batch size: 64.
- Optimizer: Adam with learning rate `0.001`.
- Objective: cross-entropy loss.
- Training: ten epochs for each model, with per-epoch training and validation
  loss and accuracy tracking.

This is a custom 70/15/15 course-project split. It does **not** preserve the
standard Fashion-MNIST train/test split, so results are not directly comparable
to benchmark scores using that official partition.

## Evaluation and visualizations

The notebook includes:

- Sample input images and model parameter summaries.
- Validation loss and accuracy curves for all three models.
- A deep-CNN test-set confusion matrix and classification report with
  precision, recall, and F1 scores.
- The most confident incorrect predictions from the deep CNN.
- Predictions from all three models on one randomly selected test image.

### Saved output snapshot

The checked-in notebook contains these validation accuracies at epoch 10:

| Model | Saved validation accuracy |
| --- | --- |
| MLP | 88.48% |
| Standard CNN | 92.50% |
| Deep CNN | 93.52% |

These values are read from existing notebook outputs; they are **not** new
measurements, guaranteed reproduction results, or test-set accuracies. The
saved deep-CNN test classification report displays accuracy rounded to `0.93`.
It does not provide full test-set reports for the other two architectures.

## Reproducibility and scope

The current notebook does not fix random seeds for dataset splitting, model
initialization, training shuffles, or random sample selection. Rerunning it can
produce different splits and metrics. Restart the kernel before a complete
run to avoid stale state from out-of-order cell execution.

Models and tensors remain on the CPU; CUDA or Apple MPS acceleration is not
implemented. The notebook does not export trained checkpoints or provide an
inference service. Training functions refer to notebook-level dataset variables
and are intended for this notebook workflow rather than as a reusable library.

## Contributors

This repository includes contributions by **Civan Rumet Arı** and **Ulaş Durak**.
The Git history retains their individual authorship.
