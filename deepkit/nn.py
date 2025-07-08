"""
A neural network is a collection of layers that are connected together.
"""
from numpy import ndarray

from deepkit.layers import Layer
from deepkit.tensor import Tensor
from typing import Sequence

class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass through the network.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> ndarray:
        """
        Backward pass through the network.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad