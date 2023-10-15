from skynet_ml.nn.regularizers.base import BaseRegularizer
import numpy as np


class L2(BaseRegularizer):
    """
    L2 Regularizer (also known as Ridge regularization).
    
    L2 regularization adds a penalty equivalent to the square of the magnitude of coefficients. 
    This results in shrinking the coefficients and it helps to prevent multicollinearity. 
    L2 will not result in the elimination of coefficients, unlike L1 which can eliminate some coefficients.
    
    Given a weight matrix `W`, the L2 regularization term is:
    
    L2 = lambda * sum(W^2)
    
    where:
    - `lambda` is the regularization coefficient, determining the strength of the regularization.
    - `sum(W^2)` is the sum of the squared values of the weights.

    Args:
        lambda_val (float, optional): Regularization coefficient. Default value is set to 0.0001.

    Attributes:
        lambda_val (float): Regularization coefficient.
        name (str): Name of the regularizer with its lambda value, useful for representation purposes.
    """


    def __init__(self, lambda_val: float = 0.0001):
        """
        Initializes the L2 regularizer with a given regularization coefficient.

        Args:
            lambda_val (float, optional): Regularization coefficient.
        """
        super().__init__(lambda_val)
        self.name = f"l2_{str(lambda_val)}"


    def forward(self, weights: np.array) -> np.array:
        """
        Compute the L2 regularization cost for the given weights.

        Args:
            weights (np.array): Weights of a neural network layer.

        Returns:
            float: L2 regularization cost for the given weights.
        """
        return self.lambda_val * np.sum(np.square(weights))

    
    def backward(self, weights: np.array) -> np.array:
        """
        Compute the gradient of the L2 regularization cost with respect to the given weights.

        Args:
            weights (np.array): Weights of a neural network layer.

        Returns:
            np.array: Gradient of the L2 regularization cost with respect to the given weights. It provides a scaled version of the weights by the regularization coefficient.
        """
        return self.lambda_val * 2 * weights
