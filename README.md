# MNIST Classification under 8K Parameters

This project demonstrates achieving 99.4% accuracy on MNIST with less than 8,000 parameters in 15 epochs or less.

## Project Overview

The goal is to create a CNN model that:
- Achieves 99.4% accuracy consistently
- Uses less than 8,000 parameters
- Trains in 15 epochs or less

## Model Evolution

### Model_1 (Baseline)
- Parameters: 7,840
- Result: 99.1% accuracy
- Key features: Basic CNN with BatchNorm
- Analysis: Showed potential but needed optimization

### Model_2
- Parameters: 7,968
- Result: ~99.2% accuracy
- Key features: Optimized channel sizes, dropout
- Analysis: Improved stability but accuracy still short

### Model_4 (Final Solution)
- Parameters: 7,996
- Result: 99.4% consistently from epoch 11-15
- Architecture highlights:
  - Three conv layers (12->24->16 channels)
  - Dilated convolution in middle layer
  - Strategic padding in convolutions
  - Progressive dropout (0.1->0.2)
  - Global Average Pooling
  - Efficient parameter utilization

## Training Configuration

- Batch size: 64
- Learning rate: 0.05
- Optimizer: SGD with momentum (0.9)
- Weight decay: 1e-4
- Scheduler: OneCycleLR
  - 20% warm-up
  - div_factor: 25
  - final_div_factor: 1000

## Results

The final model (Model_4) consistently achieves:
- Training accuracy: >99.4%
- Test accuracy: 99.4%
- Convergence: Usually by epoch 11-15
- Parameters: 7,996 (under 8K limit)

## Key Success Factors

1. Architecture Design:
   - Balanced channel progression
   - Strategic use of dilated convolution
   - Efficient parameter utilization
   - Proper regularization (BatchNorm + Dropout)

2. Training Strategy:
   - OneCycleLR scheduler
   - Moderate batch size
   - Proper weight initialization
   - Balanced regularization

## Usage

1. Install requirements:



