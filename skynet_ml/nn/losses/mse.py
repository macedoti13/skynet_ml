from skynet_ml.nn.losses.loss import Loss
import numpy as np


class MeanSquaredError(Loss):
    """
    Mean Squared Error (MSE) Loss Class

    This class represents the MSE loss, often used in regression tasks. It computes the mean of the squared differences 
    between the true and predicted values. The `compute` method calculates the loss, while the `gradient` method computes 
    the gradient of the loss with respect to the predictions.

    Methods
    -------
    compute(y_true: np.array, y_hat: np.array) -> float:
        Computes the mean squared error loss.

    gradient(y_true: np.array, y_hat: np.array) -> np.array:
        Computes the gradient of the mean squared error loss with respect to the predictions.
    """


    def compute(self, y_true: np.array, y_hat: np.array) -> float:
        """
        Compute the mean squared error loss.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target values.
        y_hat : np.array
            Estimated targets as returned by a model.
            
        Returns
        -------
        float
            The computed mean squared error loss value.
        """
        self._check_shape(y_true, y_hat)
        return np.mean((y_true - y_hat)**2)


    def gradient(self, y_true: np.array, y_hat: np.array) -> np.array:
        """
        Compute the gradient of the mean squared error loss.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target values.
        y_hat : np.array
            Estimated targets as returned by a model.
            
        Returns
        -------
        np.array
            The gradient of the mean squared error loss with respect to predictions.
        """
        self._check_shape(y_true, y_hat)
        return -2 * (y_true - y_hat)
