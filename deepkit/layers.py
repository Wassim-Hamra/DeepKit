"""
Neural nets are built from layers, which are the building blocks of the network.
Each layer needs to pass the input to the next layer,
and propagate the gradients back to the previous layer.
"""

import numpy as np
from deepkit.tensor import Tensor
from typing import Dict, Callable


class Layer:
    def __init__(self,) -> None:
        self.params: Dict[str, Tensor] = {}  # learnable parameters of the layer
        self.grads: Dict[str, Tensor] = {}

    def forward(self,inputs: Tensor) -> Tensor:
        """
        Produce the outputs of the layer given the inputs.
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward pass through the layer.
        """
        raise NotImplementedError


class Linear(Layer):
    """
    computes the linear transformation of the input: input * weights + bias
    where weights and bias are learnable parameters.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self,inputs: Tensor) -> Tensor:
        """
        Forward pass through the layer.
        outputs = inputs * weights + bias
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]


    def backward(self,grad: Tensor) -> Tensor:
        """
        Backward pass through the layer.
        grad = grad * weights^T
        """
        # grad is the gradient of the loss with respect to the outputs of this layer
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad

        return grad @ self.params["w"].T


F = Callable[[Tensor], Tensor]


class Activation(Layer):
    """
    Base class for activation functions.
    """

    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime


    def forward(self,inputs: Tensor) -> Tensor:
        """
        Forward pass through the layer.
        """
        self.inputs = inputs
        return self.f(inputs)

    def backward(self,grad: Tensor) -> Tensor:
        """
        Backward pass through the layer.
        grad = grad * f_prime(inputs)
        """
        return grad * self.f_prime(self.inputs)



def tanh(x: Tensor) -> Tensor:
    """
    Tanh activation function.
    """
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    """
    Derivative of the tanh activation function.
    """
    return 1 - tanh(x) ** 2


class Tanh(Activation):
    """
    Tanh activation function layer.
    """

    def __init__(self) -> None:
        super().__init__(f=tanh, f_prime=tanh_prime)
