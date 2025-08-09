# Rock-Paper-Scissors Neural Network (RPSNN)
This repository contains a simple demo neural network for predicting moves in Rock-Paper-Scissors (RPS). The model (rpsnn.pth) is a pretrained example—feel free to train your own or modify it as you like!

## About
The neural network is a small fully connected PyTorch model designed to learn patterns in sequences of RPS moves.

The input is a history of 5 previous moves, one-hot encoded.

The network outputs probabilities for the next move (Rock, Paper, or Scissors).

## Usage
To use the pretrained model, load rpsnn.pth into your PyTorch RPSNet model.

This model is just a demo — training your own model with your data is recommended to improve performance and adapt to your style.

Scripts for training and evaluation can be placed inside the scripts/ folder.

## Requirements
Python 3.x
PyTorch
``python -m venv venv``
``pip install torch torchvision``

## Video Demo (Done in google colab)

Feel free to explore, train, and experiment! If you have questions or improvements, email me at kiamehr13922014@gmail.com

