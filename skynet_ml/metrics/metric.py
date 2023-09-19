from abc import ABC, abstractmethod
import numpy as np

class Metric(ABC):
    """
    Base abstract class for metrics in machine learning tasks.
    
    This class provides a blueprint for all the metric subclasses and ensures they implement the `compute` method.
    It also provides a helper method to check the shape consistency between predicted and true labels.

    Attributes
    ----------
    None

    Methods
    -------
    compute(yhat: np.array, ytrue: np.array) -> float:
        Abstract method. Subclasses should provide their implementation to compute the metric value.

    _check_shape(yhat: np.array, ytrue: np.array) -> None:
        Helper method to validate the shapes of predicted and true labels.
    """

    @abstractmethod
    def compute(self, yhat: np.array, ytrue: np.array) -> float:
        """
        Abstract method to compute the metric value for given predictions and true labels.

        Subclasses should provide their own implementation of this method.

        Parameters
        ----------
        yhat : np.array
            Predicted labels or outputs.
        
        ytrue : np.array
            True labels.

        Returns
        -------
        float
            The computed metric value.
        """
        pass
    
    
    @classmethod
    def _check_shape(cls, yhat: np.array, ytrue: np.array) -> None:
        """
        Validate the shapes of predicted and true labels.

        This method checks whether the predicted and true labels are 2D arrays and have the same shape.
        Raises an error if they don't match the expected format.

        Parameters
        ----------
        yhat : np.array
            Predicted labels or outputs.
        
        ytrue : np.array
            True labels.

        Raises
        ------
        ValueError:
            If the shapes of yhat and ytrue don't match or aren't 2D arrays.
        """
        if (yhat.shape != ytrue.shape) or (yhat.ndim != 2):
            raise ValueError(f"yhat and ytrue must be 2D arrays of the same size, got {yhat.shape} and {ytrue.shape}")
