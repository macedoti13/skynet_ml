from skynet_ml.nn.regularizers.regularizer import Regularizer
import numpy as np

class L2Regularizer(Regularizer):
    """
    L2 Regularizer.

    The L2 regularizer, often termed as Ridge regularization, adds a penalty equal 
    to the sum of the squared values of the weights to the loss function. This tends 
    to drive weights to small, but non-zero values, leading to a more evenly distributed 
    set of weights.

    The regularization term added to the loss function is:
        lambda_val * sum(weights^2)
    where lambda_val is the regularization strength and weights^2 is the squared value of the weights.

    The gradient of this regularization term with respect to the weights is 2 times the weights.

    Methods
    -------
    forward(weights: np.array) -> np.array:
        Compute the L2 regularization term for the provided weights.

    backward(weights: np.array) -> np.array:
        Compute the gradient of the L2 regularization term with respect to the weights.

    """

    def forward(self, weights: np.array) -> np.array:
        """
        Compute the L2 regularization term for the provided weights.

        Parameters
        ----------
        weights : np.array
            The weights of a particular layer in the neural network.

        Returns
        -------
        np.array
            L2 regularization term based on the provided weights.
        """
        return self.lambda_val * np.sum(np.square(weights))
    
    
    def backward(self, weights: np.array) -> np.array:
        """
        Compute the gradient of the L2 regularization term with respect to the weights.

        Parameters
        ----------
        weights : np.array
            The weights of a particular layer in the neural network.

        Returns
        -------
        np.array
            Gradient of the L2 regularization term with respect to the provided weights.
        """
        return self.lambda_val * 2 * weights
