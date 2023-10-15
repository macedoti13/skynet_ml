from abc import ABC, abstractmethod
import numpy as np


class BaseRegularizer(ABC):
    """
    Abstract base class for regularization techniques in neural networks.

    Regularization is a technique used to prevent overfitting by adding an additional cost to the loss function, 
    which discourages certain complex models. This cost is typically associated with the magnitude (norm) of the 
    model parameters (weights and biases). Common regularizers include L1 (Lasso), L2 (Ridge), and a combination of both (Elastic Net).

    Args:
        lambda_val (float, optional): Regularization coefficient. It determines the strength of the regularization. 
                                      A higher value of lambda_val results in stronger regularization. Default value is set to 0.0001.

    Attributes:
        lambda_val (float): Regularization coefficient.
    """
    
    def __init__(self, lambda_val: float = 0.0001):
        """
        Initializes the base regularizer with a regularization coefficient.

        Args:
            lambda_val (float, optional): Regularization coefficient.
        """
        self.lambda_val = lambda_val
        
        
    @abstractmethod
    def forward(self, weights: np.array) -> float:
        """
        Compute the regularization loss for the given weights.

        This method is meant to be overridden by specific regularization techniques to provide the appropriate regularization cost.

        Args:
            weights (np.array): Weights of a neural network layer.

        Returns:
            float: Regularization cost for the given weights.
        """
        pass
        
        
    @abstractmethod
    def backward(self, weights: np.array) -> np.array:
        """
        Compute the gradient of the regularization loss with respect to the given weights.

        This method is meant to be overridden by specific regularization techniques to provide the gradient of the regularization cost.

        Args:
            weights (np.array): Weights of a neural network layer.

        Returns:
            np.array: Gradient of the regularization loss with respect to the given weights.
        """
        pass
