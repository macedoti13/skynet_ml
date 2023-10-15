from skynet_ml.nn.regularizers.base import BaseRegularizer
import numpy as np


class L1L2(BaseRegularizer):
    """
    Elastic Net Regularizer which is a combination of L1 and L2 regularization.
    
    Elastic Net regularization combines penalties from both L1 (Lasso) and L2 (Ridge) regularizations. 
    It tends to inherit some of Lasso's ability to exclude useless variables and also Ridge's 
    ability to include all variables in a model but with potentially none of them having high coefficients.
    
    Given a weight matrix `W`, the L1L2 regularization term is:
    
    L1L2 = lambda * (alpha * sum(|W|) + (1 - alpha) * sum(W^2))
    
    where:
    - `lambda` is the regularization coefficient, determining the strength of the regularization.
    - `alpha` is the mixing parameter between L1 and L2 regularization. If alpha = 1, it's Lasso. If alpha = 0, it's Ridge.
    - `sum(|W|)` is the sum of the absolute values of the weights (L1 regularization component).
    - `sum(W^2)` is the sum of the squared values of the weights (L2 regularization component).

    Args:
        lambda_val (float, optional): Regularization coefficient. Default value is set to 0.0001.
        alpha (float, optional): Mixing parameter for L1 and L2. Default value is set to 0.5.

    Attributes:
        lambda_val (float): Regularization coefficient.
        alpha (float): Mixing parameter between L1 and L2 regularization components.
        name (str): Name of the regularizer with its parameters, useful for representation purposes.
    """


    def __init__(self, lambda_val: float = 0.0001, alpha: float = 0.5):
        """
        Initializes the Elastic Net regularizer with given parameters.

        Args:
            lambda_val (float, optional): Regularization coefficient.
            alpha (float, optional): Mixing parameter for L1 and L2.
        """
        super().__init__(lambda_val)
        self.alpha = alpha
        self.name = f"l1l2_{str(lambda_val)}_{str(alpha)}"


    def forward(self, weights: np.array) -> np.array:
        """
        Compute the Elastic Net regularization cost for the given weights.

        Args:
            weights (np.array): Weights of a neural network layer.

        Returns:
            float: Elastic Net regularization cost for the given weights.
        """
        l1_component = self.alpha * np.sum(np.abs(weights))
        l2_component = (1 - self.alpha) * np.sum(np.square(weights))
        
        return self.lambda_val * (l1_component + l2_component)


    def backward(self, weights: np.array) -> np.array:
        """
        Compute the gradient of the Elastic Net regularization cost with respect to the given weights.

        Args:
            weights (np.array): Weights of a neural network layer.

        Returns:
            np.array: Gradient of the Elastic Net regularization cost with respect to the given weights.
        """
        l1_grad = self.lambda_val * 2 * self.alpha * np.sign(weights)
        l2_grad = self.lambda_val * (1 - self.alpha) * weights
        
        return l1_grad + l2_grad
