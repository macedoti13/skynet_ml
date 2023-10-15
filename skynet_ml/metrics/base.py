from abc import ABC, abstractmethod
import numpy as np


class BaseMetric(ABC):
    """
    Abstract Base Class for defining metrics in machine learning models.

    This class should be subclassed when defining custom metrics.
    """


    @abstractmethod
    def compute(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Compute the metric given the true labels and the predicted labels.

        Args:
            y_true (np.array): Ground truth (correct) target values.
            y_pred (np.array): Estimated target values.

        Returns:
            float: The computed metric value.
        """
        pass


    @classmethod
    def check_shape(cls, y_true: np.array, y_pred: np.array) -> None:
        """
        Check the shapes of the true labels and predicted labels.

        Ensures that both `y_true` and `y_pred` are two-dimensional arrays
        and have the same shape.

        Args:
            y_true (np.array): Ground truth (correct) target values.
            y_pred (np.array): Estimated target values.

        Raises:
            ValueError: If the shapes of the input arrays are not the same or not 2D.
        """
        if (y_pred.shape != y_true.shape) or (y_pred.ndim != 2):
            raise ValueError(f"y_true and y_pred must have the same shape and be 2D arrays. y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
