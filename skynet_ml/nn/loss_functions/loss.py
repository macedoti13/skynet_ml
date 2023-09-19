from abc import ABC, abstractmethod
from typing import Optional, Union
from skynet_ml.nn.regularizers.regularizer import Regularizer
import numpy as np

class Loss(ABC):
    """
    Base class for neural network loss functions.

    Loss functions evaluate how well the model's predictions match the true labels.
    They are central to training neural networks, as optimizing the loss function
    helps improve the model's accuracy.

    Subclasses should implement the `compute` method to calculate the value of the 
    loss for given predictions and true labels, and the `gradient` method to calculate 
    the gradient of the loss with respect to the predictions.

    Attributes
    ----------
    None

    Methods
    -------
    compute(yhat: np.array, ytrue: np.array) -> float:
        Compute the loss value for given predictions and true labels.

    gradient(yhat: np.array, ytrue: np.array) -> np.array:
        Compute the gradient of the loss with respect to the predictions for given 
        predictions and true labels.
    """

    @abstractmethod
    def compute(self, yhat: np.array, ytrue: np.array) -> float:
        """
        Compute the loss value for given predictions and true labels.

        Parameters
        ----------
        yhat : np.array
            Predicted labels or outputs.
        
        ytrue : np.array
            True labels.

        Returns
        -------
        float
            The computed loss value.

        """
        pass


    @abstractmethod
    def gradient(self, yhat: np.array, ytrue: np.array) -> np.array:
        """
        Compute the gradient of the loss with respect to the predictions.

        Parameters
        ----------
        yhat : np.array
            Predicted labels or outputs.
        
        ytrue : np.array
            True labels.

        Returns
        -------
        np.array
            The gradient of the loss with respect to the predictions.

        """
        pass
    
    
    @classmethod
    def _check_shape(cls, yhat: np.array, ytrue: np.array) -> np.array:
        """
        Internal method to validate the shapes of the input arrays yhat and ytrue.

        Parameters
        ----------
        yhat : np.array
            Predicted labels or outputs.
        
        ytrue : np.array
            True labels.

        Raises
        ------
        ValueError
            If yhat and ytrue do not have the same shape or if they are not 2D arrays.
        
        """
        if (yhat.shape != ytrue.shape) or (yhat.ndim != 2):
            raise ValueError(f"yhat and ytrue must be 2D arrays of the same size, got {yhat.shape} and {ytrue.shape}")
