# MNIST Classification with CI/CD Pipeline

A PyTorch implementation of MNIST digit classification with automated testing and deployment pipeline.

## Table of Contents
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Setup and Installation](#setup-and-installation)
- [Training and Testing](#training-and-testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Model Management](#model-management)

## Requirements
### Performance Metrics
- Model parameters < 20,000
- Test accuracy > 99.4%
- Training epochs = 20

### Technical Requirements
- Python 3.8+
- PyTorch 2.0.1
- CUDA-capable GPU (recommended)

## Project Structure
```
├── model.py           # Model architecture and training code
├── test_model.py      # Test cases and validation
├── requirements.txt   # Project dependencies
├── .gitignore        # Git ignore rules (see below)
└── .github/
    └── workflows/
        └── ml-pipeline.yml  # CI/CD configuration
```

### .gitignore Configuration
Create a `.gitignore` file in your project root with the following content:
```
# Python virtual environment
rajenv/

# Python cache files
__pycache__/
```

## Model Architecture
### Network Components
- 3 Convolutional layers with batch normalization
- Dropout layers (0.1) for regularization
- Global Average Pooling (GAP)
- Single fully connected layer

### Training Configuration
- Batch size: 128
- Optimizer: Adam
- Learning rate scheduler: OneCycleLR
- Data augmentation: Random rotation and affine transforms

## Setup and Installation
1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training and Testing
### Local Training
```bash
python model.py
```
- Progress bars show real-time training status
- Logs display accuracy and loss metrics
- Models automatically saved when accuracy ≥99%

### Running Tests
```bash
pytest test_model.py -v
```
Test cases verify:
1. Parameter count (<20,000)
2. Model accuracy (>99.4%)

## CI/CD Pipeline
### Workflow Steps
1. Environment setup
2. Dependency installation
3. Model training
4. Validation testing
5. Model artifact upload

### GitHub Actions Configuration
- Triggered on every push
- Runs on Ubuntu latest
- Uses Python 3.8
- Automated testing and validation

## Model Management
### Saving Convention
- Format: `mnist_model_acc{accuracy}.pth`
- Only saves models with ≥99% accuracy
- Includes accuracy in filename for easy reference

### Deployment Process
1. Train model locally
2. Verify tests pass
3. Commit and push:
```bash
git add .
git commit -m "Initial commit with working model"
git push origin main
```

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.
```

</rewritten_file>