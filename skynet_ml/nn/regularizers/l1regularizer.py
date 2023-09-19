from skynet_ml.nn.regularizers.regularizer import Regularizer
import numpy as np

class L1Regularizer(Regularizer):
    """
    L1 Regularizer.

    The L1 regularizer, also known as Lasso regularization, penalizes the absolute 
    value of the weights. This encourages sparsity in the neural network, meaning 
    many weights can be driven to zero. This sparsity can lead to a more interpretable 
    model, as less important features will have their corresponding weights shrunk to zero.

    The regularization term added to the loss function is:
        lambda_val * sum(|weights|)
    where lambda_val is the regularization strength and |weights| is the absolute value of the weights.

    The gradient of this regularization term with respect to the weights is simply the sign of the weights.

    Methods
    -------
    forward(weights: np.array) -> np.array:
        Compute the L1 regularization term for the provided weights.

    backward(weights: np.array) -> np.array:
        Compute the gradient of the L1 regularization term with respect to the weights.

    """

    def forward(self, weights: np.array) -> np.array:
        """
        Compute the L1 regularization term for the provided weights.

        Parameters
        ----------
        weights : np.array
            The weights of a particular layer in the neural network.

        Returns
        -------
        np.array
            L1 regularization term based on the provided weights.
        """
        return self.lambda_val * np.sum(np.abs(weights))
    
    
    def backward(self, weights: np.array) -> np.array:
        """
        Compute the gradient of the L1 regularization term with respect to the weights.

        Parameters
        ----------
        weights : np.array
            The weights of a particular layer in the neural network.

        Returns
        -------
        np.array
            Gradient of the L1 regularization term with respect to the provided weights.
        """
        return self.lambda_val * np.sign(weights)
