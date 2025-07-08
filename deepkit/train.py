"""
Here we define the training loop for the nn.
"""

from deepkit.tensor import Tensor
from deepkit.nn import NeuralNet
from deepkit.optimizers import Optimizer, SGD
from deepkit.data import DataIterator, Batch, BatchIterator
from deepkit.loss import Loss, MSE


def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          optimizer: Optimizer = SGD(),
          loss_fn: Loss = MSE(),
          data_iterator: DataIterator = BatchIterator(),
          epochs: int = 5000) -> None:
    """
    Train the neural network using the provided inputs and targets.

    Args:
        net (NeuralNet): The neural network to train.
        inputs (Tensor): Input data for training.
        targets (Tensor): Target data for training.
        optimizer (Optimizer): Optimizer to use for updating parameters.
        loss_fn (Loss): Loss function to evaluate the model's performance.
        data_iterator (DataIterator): Iterator to yield batches of data.
        epochs (int): Number of epochs to train the model.
    """
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in data_iterator(inputs, targets):
            # Forward pass
            predicted = net.forward(batch.inputs)
            epoch_loss += loss_fn.loss(predicted, batch.targets)
            grad = loss_fn.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(f"Epoch {epoch + 1}/{epochs}: Loss: {epoch_loss}")