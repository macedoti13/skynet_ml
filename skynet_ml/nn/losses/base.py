from abc import ABC, abstractmethod
import numpy as np


class BaseLoss(ABC):
    """
    Abstract base class for defining custom loss functions.

    A loss function (or cost function) quantifies how well the predictions of a model
    match the true values. It computes the difference between the predicted values 
    and the actual values for a given set of input data. The main goal during training 
    is to minimize this difference.

    Derived classes must implement the `compute` and `gradient` methods.
    """


    @abstractmethod
    def compute(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Computes the loss value for the given true and predicted values.

        Args:
            y_true (np.array): Ground truth labels. Expected to be a 2D array with shape (batch_size, n_classes).
            y_pred (np.array): Predicted labels from the model. Expected to have the same shape as y_true.

        Returns:
            float: The computed loss value.
        """
        pass


    @abstractmethod
    def gradient(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
        Computes the gradient of the loss with respect to the predicted values.

        Args:
            y_true (np.array): Ground truth labels. Expected to be a 2D array with shape (batch_size, n_classes).
            y_pred (np.array): Predicted labels from the model. Expected to have the same shape as y_true.

        Returns:
            np.array: Gradient of the loss with respect to the predicted values. Expected to have the same shape as y_pred.
        """
        pass


    @classmethod
    def _check_shape(cls, y_true: np.array, y_pred: np.array) -> None:
        """
        Utility method to ensure the shapes of the true and predicted values are compatible.

        Args:
            y_true (np.array): Ground truth labels.
            y_pred (np.array): Predicted labels from the model.

        Raises:
            ValueError: If y_true and y_pred are not 2D arrays of the same size.
        """
        if (y_pred.shape != y_true.shape) or (y_pred.ndim != 2):
            raise ValueError(f"y_pred and y_true must be 2D arrays of the same size, got {y_pred.shape} and {y_true.shape}")
