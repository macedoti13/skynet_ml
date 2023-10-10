from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    """
    Loss Base Class

    The Loss class is an abstract base class (ABC) that provides a blueprint for loss function classes. It enforces 
    the implementation of `compute` and `gradient` methods within child classes. These methods are essential 
    for calculating the loss and its gradient with respect to predictions, respectively. 

    The `_check_shape` class method ensures that the ground-truth labels and predicted values have matching 
    shapes and are 2-dimensional before further computations, throwing a ValueError otherwise. 
    
    Methods
    -------
    compute(y_true: np.array, y_hat: np.array) -> float:
        Abstract method. Should compute and return the loss based on true and predicted values.
        
    gradient(y_true: np.array, y_hat: np.array) -> np.array:
        Abstract method. Should compute and return the gradient of the loss with respect to predictions.

    _check_shape(y_true: np.array, y_hat: np.array) -> None:
        Class method to ensure input arrays have correct and matching shapes.
    """
    
    
    @abstractmethod
    def compute(self, y_true: np.array, y_hat: np.array) -> float:
        """
        Abstract method to compute the loss.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target values.
        y_hat : np.array
            Estimated targets as returned by a model.
            
        Returns
        -------
        float
            The computed loss value.
        """
        pass


    @abstractmethod
    def gradient(self, y_true: np.array, y_hat: np.array) -> np.array:
        """
        Abstract method to compute the gradient of the loss.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target values.
        y_hat : np.array
            Estimated targets as returned by a model.
            
        Returns
        -------
        np.array
            The computed gradient of the loss with respect to predictions.
        """
        pass
    
    
    @classmethod
    def get_config(self) -> dict:
        """
        Abstract method to get the configuration of the loss.

        Returns
        -------
        dict
            The configuration of the loss.
        """
        return {}
    
    
    @classmethod
    def _check_shape(cls, y_true: np.array, y_hat: np.array) -> None:
        """
        Check if y_true and y_hat have the correct and matching shapes.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target values.
        y_hat : np.array
            Estimated targets as returned by a model.
            
        Raises
        ------
        ValueError
            If the shape of y_hat is not equal to the shape of y_true or if they are not 2-dimensional.
        """
        if (y_hat.shape != y_true.shape) or (y_hat.ndim != 2):
            raise ValueError(f"y_hat and y_true must be 2D arrays of the same size, got {y_hat.shape} and {y_true.shape}")
