# MNIST Classification with PyTorch

![ML Pipeline](https://github.com/rajendrosahu/MNISTNEW/actions/workflows/test.yml/badge.svg)

A deep learning project that achieves >99% accuracy on the MNIST dataset using a CNN architecture with less than 20,000 parameters.

## Project Overview

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification with:
- Parameter count < 20,000
- Test accuracy > 99%
- Efficient architecture using BatchNorm and Dropout
- Data augmentation for better generalization

## Model Architecture

```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- tqdm
- pytest
- numpy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MNISTNEW.git
cd MNISTNEW
```

2. Create and activate virtual environment:
```bash
# Create virtual environment
python -m venv rajenv

# Activate virtual environment
# On Windows:
rajenv\Scripts\activate
# On Unix or MacOS:
source rajenv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script:
```bash
python model.py
```

The model will be automatically saved when it achieves >99% accuracy on the test set.
The saved model will be named: `mnist_model_acc{accuracy}.pth`

### Running Tests

To verify model performance and architecture:
```bash
pytest test_model.py -v
```

This will run two tests:
1. Parameter count test (must be <20,000)
2. Accuracy test (must be >99%)

## Model Features

- **Data Augmentation**: 
  - Random rotation (15 degrees)
  - Random affine transforms
  - Random erasing
  - Normalization

- **Training Optimizations**:
  - Adam optimizer with weight decay
  - OneCycleLR scheduler
  - Gradient clipping
  - Batch normalization
  - Dropout regularization

## Directory Structure

```
MNISTNEW/
├── model.py           # Model architecture and training code
├── test_model.py      # Test cases for model validation
├── requirements.txt   # Project dependencies
├── .gitignore        # Git ignore rules
└── README.md         # Project documentation
```

## GitHub Actions

The project includes automated CI/CD pipeline that:
1. Sets up Python environment
2. Installs dependencies
3. Trains the model
4. Runs validation tests
5. Saves the trained model as an artifact

## Performance Metrics

- Parameters: <20,000
- Test Accuracy: >99%
- Training Time: ~20 epochs
- Memory Efficient: Small model size



