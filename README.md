# Neural Network From Scratch

A NumPy implementation of a feedforward neural network with backpropagation, binary cross-entropy loss, and early stopping.  
The model is applied to the Breast Cancer dataset to demonstrate training and validation.

## Features
- Implemented purely in **NumPy** (no TensorFlow/PyTorch).
- Supports multiple hidden layers and sigmoid activations.
- Includes:
  - Weight/bias initialization
  - Feedforward computation
  - Backpropagation
  - Gradient descent updates
  - Binary cross-entropy loss
  - Train/validation split
  - Early stopping
  - Accuracy evaluation

## Repository Structure
- neuralnet.py # Core framework (training logic)
- breast_cancer_demo.py # Example training script with Breast Cancer dataset
- equirements.txt # Python dependencies
- README.md # Project documentation


## Installation
```bash
# Clone the repository
git clone https://github.com/<your-username>/nn-from-scratch.git
cd nn-from-scratch

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage 
```bash
python breast_cancer_demo.py
epoch    0 | train_loss=0.69 | val_loss=0.68 | train_acc=0.62 | val_acc=0.66
...
epoch  500 | train_loss=0.09 | val_loss=0.12 | train_acc=0.98 | val_acc=0.96
```

## Results 
Final results on the Breast Cancer dataset:

- Train accuracy: ~0.98
- Validation accuracy: ~0.96