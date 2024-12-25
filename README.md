# XOR neural network classifier from scratch
a numpy implementation of a neural network solving the xor problem from scratch.

## overview
this project implements a neural network to solve the xor classification problem without using any deep learning frameworks. the xor problem is non-linearly separable, requiring a hidden layer and non-linear activation functions to solve - making it a great example for understanding why we need multi-layer neural networks.

## implementation details
- built using only numpy for computations and matplotlib for visualization
- custom implementation of:
  - feed-forward logic
  - backpropagation algorithm
  - sigmoid activation function
  - loss calculation
- total trainable parameters: 9
- modular design for clear separation of components

## results
training progression:
![training plot](https://github.com/user-attachments/assets/adeda2ec-8db0-4dc6-a33c-b74de5bdac7f)

initial predictions:
![initial state](https://github.com/user-attachments/assets/590e11be-331c-43c9-aa23-e4ef236e9c80)

final predictions:
![final state](https://github.com/user-attachments/assets/488211cb-f3bb-4bb5-8619-87670aa3d080)

## getting started
requirements:
- numpy
- matplotlib

usage:
```bash
python model.py
```
note: due to random weight initialization, you might need to run the model multiple times to avoid local minima.

the model classifies the following xor truth table:
```
(0,0) -> 0
(1,0) -> 1
(0,1) -> 1
(1,1) -> 0
```
