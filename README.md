# Handwritten Digits Classifier with PyTorch
This project involves developing a handwritten digits classifier using PyTorch, trained on the MNIST dataset. The goal is to build, train, and evaluate a neural network to recognize handwritten digits with high accuracy.

## Project Overview
As part of the Udacity deep learning course, this project demonstrates the implementation of a neural network for optical character recognition (OCR). The MNIST dataset, a standard dataset for image classification tasks, is used to train and validate the model.

## Features
**Data Loading and Preprocessing:**
Utilized torchvision to load and preprocess the MNIST dataset, including transformations like tensor conversion and normalization.

**Neural Network Architecture:**
Built a neural network with multiple hidden layers using PyTorch's nn.Module.

**Model Training:**
Implemented the training loop with an Adam optimizer and CrossEntropyLoss to optimize model weights.

**Evaluation:**
Achieved over 95% accuracy on the test set and saved the trained model.

**Visualization:**
Visualized the dataset and training progress using matplotlib.


## Getting Started
Prerequisites
Python 3.11
PyTorch
torchvision
matplotlib
numpy


## Installation
Clone the repository: git clone https://github.com/NarendraKhadayat/handwritten-digits-classifier.git
cd handwritten-digits-classifier


**Save the trained model:**
torch.save(net.state_dict(), 'mnist_net.pth')


## Results
The model achieved an accuracy of over 95% on the MNIST test set.
Further tuning and experimenting with hyperparameters can push the accuracy even higher.


## Acknowledgments
**Udacity** for providing the course and project structure.
**Yann LeCun's MNIST Database** for the dataset.

## Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the **LICENSE** file for details.
