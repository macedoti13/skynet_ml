from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    """
    Base Activation class as an interface for all activation functions.
    
    This abstract class defines the skeleton for activation functions, which are essential components 
    in a neural network to introduce non-linear properties to the system. Subclasses should implement 
    the `compute`, `gradient`, and `get_config` methods.

    The `compute` method should define the forward computation of activation, while the `gradient` method
    should provide the gradient of the activation function with respect to its input, which is crucial during
    the backpropagation step of training neural networks.

    Attributes
    ----------
    None

    Methods
    -------
    compute(z: np.array) -> np.array
        Abstract method to compute the forward pass of the activation function.
    gradient(z: np.array) -> np.array
        Abstract method to compute the gradient of the activation function with respect to its input.
    get_config() -> dict
        Abstract method to retrieve the configuration of the activation function.
    _check_shape(z: np.array) -> None
        Class method to ensure that the input to the activation function is a 2D array.

    Example
    -------
    # Example should be provided in the concrete subclasses implementing this interface.
    """

    
    @abstractmethod
    def compute(self, z: np.array) -> np.array:
        """
        Abstract method to compute the forward pass of the activation function.

        Parameters
        ----------
        z : np.array
            The input to the activation function, typically the weighted sum of inputs for a neuron.
        
        Returns
        -------
        np.array
            The output of the activation function.

        Raises
        ------
        NotImplementedError
            This method needs to be implemented by the subclass.
        """
        pass

    
    @abstractmethod
    def gradient(self, z: np.array) -> np.array:
        """
        Abstract method to compute the gradient of the activation function with respect to its input.

        Parameters
        ----------
        z : np.array
            The input to the activation function.

        Returns
        -------
        np.array
            The gradient of the activation function with respect to its input.

        Raises
        ------
        NotImplementedError
            This method needs to be implemented by the subclass.
        """
        pass

    
    @classmethod
    def get_config(self):
        """
        Get the configuration of the activation function.

        Returns
        -------
        dict
            A dictionary containing the configuration parameters of the activation function.
            This should allow the activation function to be reconstructed.

        Raises
        ------
        NotImplementedError
            This method needs to be implemented by the subclass.
        """
        return {}


    @classmethod
    def _check_shape(cls, z: np.array) -> None:
        """
        Class method to ensure the input to the activation function is a 2D array.
        
        Parameters
        ----------
        z : np.array
            The input array to be checked.

        Raises
        ------
        ValueError
            If `z` is not a 2D array.
        """
        if z.ndim != 2:
            raise ValueError(f"z must be a 2D array, got {z.ndim}D array instead")
