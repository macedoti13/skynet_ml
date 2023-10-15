from skynet_ml.nn.regularizers.base import BaseRegularizer
import numpy as np


class L1(BaseRegularizer):
    """
    L1 Regularizer (also known as Lasso regularization).
    
    L1 regularization adds a penalty equivalent to the absolute value of the magnitude of coefficients. 
    It can lead to zero coefficients i.e. some of the features are completely neglected for the evaluation of output. 
    Thus, L1 can also be seen as a method for feature selection.
    
    Given a weight matrix `W`, the L1 regularization term is:
    
    L1 = lambda * sum(|W|)
    
    where:
    - `lambda` is the regularization coefficient, determining the strength of the regularization.
    - `sum(|W|)` is the sum of the absolute values of the weights.

    Args:
        lambda_val (float, optional): Regularization coefficient. Default value is set to 0.0001.

    Attributes:
        lambda_val (float): Regularization coefficient.
        name (str): Name of the regularizer with its lambda value, useful for representation purposes.
    """


    def __init__(self, lambda_val: float = 0.0001):
        """
        Initializes the L1 regularizer with a given regularization coefficient.

        Args:
            lambda_val (float, optional): Regularization coefficient.
        """
        super().__init__(lambda_val)
        self.name = f"l1_{str(lambda_val)}"


    def forward(self, weights: np.array) -> np.array:
        """
        Compute the L1 regularization cost for the given weights.

        Args:
            weights (np.array): Weights of a neural network layer.

        Returns:
            float: L1 regularization cost for the given weights.
        """
        return self.lambda_val * np.sum(np.abs(weights))

    
    def backward(self, weights: np.array) -> np.array:
        """
        Compute the gradient of the L1 regularization cost with respect to the given weights.

        Args:
            weights (np.array): Weights of a neural network layer.

        Returns:
            np.array: Gradient of the L1 regularization cost with respect to the given weights. It provides the sign of weights scaled by the regularization coefficient.
        """
        return self.lambda_val * np.sign(weights)
