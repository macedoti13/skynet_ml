from skynet_ml.nn.regularizers.regularizer import Regularizer
import numpy as np


class L1L2(Regularizer):
    """
    L1L2 Regularizer Class
    
    The L1L2 Regularizer implements a combination of L1 (Lasso) and L2 (Ridge) regularization. 
    This type of regularization encourages sparsity (like L1) while also discouraging large weights 
    (like L2), which can be beneficial in different contexts.

    The combination of the two types of regularization is controlled by the `alpha` parameter, 
    which determines the weighting between L1 and L2 regularization in the combined regularization term.

    Parameters
    ----------
    lambda_val : float, optional
        The overall regularization factor, default is set in the base `Regularizer` class.
    alpha : float, optional
        Weighting factor determining the balance between L1 and L2 regularization, default is 0.5.

    Methods
    -------
    forward(weights: np.array) -> float:
        Compute the L1L2 regularization term for the given weights.
    
    backward(weights: np.array) -> np.array:
        Compute and return the gradient of the L1L2 regularization term with respect to the weights.
    
    get_config() -> dict:
        Returns a dictionary containing the configuration of the regularizer.

    Example
    -------
    >>> l1l2_reg = L1L2(lambda_val=0.01, alpha=0.5)
    >>> weights = np.array([1.0, -2.0, 3.0])
    >>> print(l1l2_reg.forward(weights))  # Should output the combined regularization term
    >>> print(l1l2_reg.backward(weights)) # Should output the gradient of the combined regularization term
    """


    def __init__(self, lambda_val: float = 0.0001, alpha: float = 0.5):
        """
        Initialize the L1L2 regularizer with the given regularization factor and alpha.

        Parameters
        ----------
        lambda_val : float
            The overall regularization factor.
        alpha : float
            Weighting factor determining the balance between L1 and L2 regularization.
        """
        super().__init__(lambda_val)
        self.alpha = alpha
        
        
    def forward(self, weights: np.array) -> np.array:
        """
        Compute the L1L2 regularization term for the given weights.

        Parameters
        ----------
        weights : np.array
            Weights of the model for which the L1L2 regularization term is computed.

        Returns
        -------
        float
            The L1L2 regularization term for the given weights.
        """
        l1_component = self.alpha * np.sum(np.abs(weights))
        l2_component = (1 - self.alpha) * np.sum(np.square(weights))
        return self.lambda_val * (l1_component + l2_component)

    def backward(self, weights: np.array) -> np.array:
        """
        Compute the gradient of the L1L2 regularization term with respect to the weights.

        Parameters
        ----------
        weights : np.array
            Weights of the model for which the gradient is computed.

        Returns
        -------
        np.array
            Gradient of the L1L2 regularization term with respect to the given weights.
        """
        l1_grad = self.lambda_val * 2 * self.alpha * np.sign(weights)
        l2_grad = self.lambda_val * (1 - self.alpha) * weights
        return l1_grad + l2_grad


    def get_config(self) -> dict:
        """
        Returns the configuration of the L1L2 regularizer.

        Returns
        -------
        dict
            A dictionary containing the regularization factor lambda and the weighting factor alpha.

        Example
        -------
        >>> l1l2_reg = L1L2(lambda_val=0.01, alpha=0.5)
        >>> config = l1l2_reg.get_config()
        >>> print(config)
        {'lambda_val': 0.01, 'alpha': 0.5}
        """
        return {"lambda_val": self.lambda_val, "alpha": self.alpha}
