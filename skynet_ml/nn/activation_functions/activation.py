from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    """
    Base class for neural network activation functions.

    Activation functions introduce non-linearity into the model, enabling it to
    learn more complex functions. They transform the summed weighted input from 
    the node into the activation of the node or output for that input.

    Subclasses should implement the `compute` method to calculate the activation 
    value for a given input and the `gradient` method to calculate the derivative 
    of the activation function.

    Attributes
    ----------
    None

    Methods
    -------
    compute(z: np.array) -> np.array:
        Compute the activation function for a given input.

    gradient(z: np.array) -> np.array:
        Compute the derivative of the activation function for a given input.

    _check_shape(z: np.array) -> None:
        Internal method to validate the shape of the input array.

    """

    @abstractmethod
    def compute(self, z: np.array) -> np.array:
        """
        Compute the activation function for a given input.

        Parameters
        ----------
        z : np.array
            Input data, typically the weighted sum of inputs and weights.

        Returns
        -------
        np.array
            Output after applying the activation function.

        """
        pass
    
    
    @abstractmethod
    def gradient(self, z: np.array) -> np.array:
        """
        Compute the derivative of the activation function for a given input.

        Parameters
        ----------
        z : np.array
            Input data, typically the weighted sum of inputs and weights.

        Returns
        -------
        np.array
            Derivative of the activation function for each input value.

        """
        pass


    @classmethod
    def _check_shape(cls, z: np.array) -> None:
        """
        Internal method to validate the shape of the input array. Raises an 
        error if the input is not a 2D array.

        Parameters
        ----------
        z : np.array
            Input data to be checked.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If input data `z` is not a 2D array.

        """
        if z.ndim != 2:
            raise ValueError(f"z must be a 2D array, got {len(z.shape)}D array instead")
