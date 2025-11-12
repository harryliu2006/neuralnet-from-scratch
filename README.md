# Neural Network From Scratch

A NumPy implementation of a feedforward neural network with backpropagation, binary cross-entropy loss, and early stopping.  
The model is applied to the Breast Cancer dataset to demonstrate training and validation.

## Features
- Implemented purely in **NumPy** (no TensorFlow/PyTorch).
- 2 hidden layers, uses sigmoid activations for output layer and ReLU for  hidden layers.
- Includes:
  - Weight/bias initialization
  - Feedforward computation
  - Backpropagation
  - Gradient descent updates
  - Binary cross-entropy loss
  - Train/validation split
  - Early stopping
  - Accuracy evaluation


## Results 
Final results on the Breast Cancer dataset:

- Train accuracy: ~0.98
- Validation accuracy: ~0.94
