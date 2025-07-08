"""
optimizers are used to update the parameters of a model based on the gradients computed during backpropagation.
"""
from deepkit.nn import NeuralNet


class Optimizer:
    def step(slef, net: NeuralNet) -> None:
        raise NotImplementedError



class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    Updates parameters using the gradients computed during backpropagation.
    """
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def step(self, net: NeuralNet) -> None:
        for layer in net.layers:
            for param_name, param in layer.params.items():
                if param_name in layer.grads:
                    param -= self.learning_rate * layer.grads[param_name]