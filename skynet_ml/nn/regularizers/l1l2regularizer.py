from skynet_ml.nn.regularizers.regularizer import Regularizer
import numpy as np

class L1L2Regularizer(Regularizer):
    """
    Combined L1 and L2 Regularizer (Elastic Net).

    The L1L2Regularizer combines the effects of both L1 (Lasso) and L2 (Ridge) regularization. 
    It is useful for reducing the magnitude of parameters and producing sparse parameters simultaneously, 
    potentially offering a balance between feature selection and small coefficient values.

    The regularization term added to the loss function is:
        lambda_val * [alpha * sum(|weights|) + (1 - alpha) * sum(weights^2)]
    where lambda_val is the regularization strength, |weights| denotes absolute values of weights,
    and weights^2 is the squared value of the weights.

    The gradient of this regularization term with respect to the weights is a combination of the
    gradients from L1 and L2 terms.

    Parameters
    ----------
    lambda_val : float, optional
        The regularization strength, by default 0.0001.
    alpha : float, optional
        The mixing parameter between L1 and L2 regularization. 
        If alpha=1, it's equivalent to L1 regularization. If alpha=0, it's L2 regularization. 
        By default 0.5, indicating an equal combination.

    Methods
    -------
    forward(weights: np.array) -> np.array:
        Compute the combined L1 and L2 regularization term for the provided weights.

    backward(weights: np.array) -> np.array:
        Compute the gradient of the combined L1 and L2 regularization term with respect to the weights.

    """

    def __init__(self, lambda_val: float = 0.0001, alpha: float = 0.5):
        super().__init__(lambda_val)
        self.alpha = alpha
        
        
    def forward(self, weights: np.array) -> np.array:
        """
        Compute the combined L1 and L2 regularization term for the provided weights.

        Parameters
        ----------
        weights : np.array
            The weights of a particular layer in the neural network.

        Returns
        -------
        np.array
            Combined L1 and L2 regularization term based on the provided weights.
        """
        l1_component = self.alpha * np.sum(np.abs(weights))
        l2_component = (1 - self.alpha) * np.sum(np.square(weights))
        return self.lambda_val * (l1_component + l2_component)


    def backward(self, weights: np.array) -> np.array:
        """
        Compute the gradient of the combined L1 and L2 regularization term with respect to the weights.

        Parameters
        ----------
        weights : np.array
            The weights of a particular layer in the neural network.

        Returns
        -------
        np.array
            Gradient of the combined L1 and L2 regularization term with respect to the provided weights.
        """
        l1_grad = self.lambda_val * 2 * self.alpha * np.sign(weights)
        l2_grad = self.lambda_val * (1 - self.alpha) * weights
        return l1_grad + l2_grad
