from skynet_ml.nn.losses.base import BaseLoss
import numpy as np


class MeanSquaredError(BaseLoss):
    """
    Implements the Mean Squared Error (MSE) loss function.

    MSE is commonly used for regression tasks. It calculates the average of the squared differences 
    between the predicted and actual values.

    Mathematically, for true values y_true and predicted values y_pred, it is defined as:

        MSE = (1/n) * Σ (y_true - y_pred)^2

    where 'n' is the number of samples.

    Notes:
        - MSE will be high if the predicted values are far from the true values and low if they are close.
        - Its minimum value is 0, which indicates perfect predictions.
    """
    
    
    def __init__(self) -> None:
        """
        Initializes the loss function with its name.
        """
        self.name = "mean_squared_error"


    def compute(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Computes the Mean Squared Error for the given true and predicted values.

        Args:
            y_true (np.array): Ground truth labels. Expected to be a 2D array with shape (batch_size, n_classes).
            y_pred (np.array): Predicted labels from the model. Expected to have the same shape as y_true.

        Returns:
            float: The computed Mean Squared Error.
        """
        self._check_shape(y_true, y_pred)
        return np.mean((y_true - y_pred)**2)


    def gradient(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
        Computes the gradient of the Mean Squared Error with respect to the predicted values.

        Args:
            y_true (np.array): Ground truth labels. Expected to be a 2D array with shape (batch_size, n_classes).
            y_pred (np.array): Predicted labels from the model. Expected to have the same shape as y_true.

        Returns:
            np.array: Gradient of the loss with respect to the predicted values. Expected to have the same shape as y_pred.
        """
        self._check_shape(y_true, y_pred)
        return -2 * (y_true - y_pred)
