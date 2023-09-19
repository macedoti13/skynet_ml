from skynet_ml.nn.loss_functions.loss import Loss
import numpy as np

class MeanSquaredErrorLoss(Loss):
    """
    Loss function that calculates the mean squared error (MSE) between predicted and true labels.

    The MSE quantifies the average of the squares of the differences between predicted and 
    true values. It is widely used in regression problems.

    Attributes
    ----------
    None

    Methods
    -------
    compute(yhat: np.array, ytrue: np.array) -> float:
        Compute the MSE value for given predictions and true labels.

    gradient(yhat: np.array, ytrue: np.array) -> np.array:
        Compute the gradient of the MSE with respect to the predictions for given 
        predictions and true labels.
    """

    def compute(self, yhat: np.array, ytrue: np.array) -> float:
        """
        Compute the MSE value for given predictions and true labels.

        Parameters
        ----------
        yhat : np.array
            Predicted labels or outputs.
        
        ytrue : np.array
            True labels.

        Returns
        -------
        float
            The computed MSE value.

        """
        self._check_shape(yhat, ytrue)
        return np.mean((ytrue - yhat)**2)

        
    def gradient(self, yhat: np.array, ytrue: np.array) -> np.array:
        """
        Compute the gradient of the MSE with respect to the predictions.

        Parameters
        ----------
        yhat : np.array
            Predicted labels or outputs.
        
        ytrue : np.array
            True labels.

        Returns
        -------
        np.array
            The gradient of the MSE with respect to the predictions.

        """
        self._check_shape(yhat, ytrue)
        return -2 * (ytrue - yhat)
