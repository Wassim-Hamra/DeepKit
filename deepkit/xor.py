"""
A function that cannot be expressed as a linear combination of its inputs, is XOR
"""

from deepkit.train import train
from deepkit.nn import NeuralNet
from deepkit.layers import Linear, Tanh
import numpy as np


inputs = np.array(
    [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]],
)

targets = np.array(
    [[0],
     [1],
     [1],
     [0]],
)

net = NeuralNet([
    Linear(input_size=2, output_size=1),
])

train(net, inputs, targets)
for x,y in zip(inputs, targets):
    predicted = net.forward(x)
    print(f"Input: {x}, Predicted: {predicted}, Target: {y}")
