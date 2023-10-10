from abc import ABC, abstractmethod
import numpy as np


class Initializer(ABC): 
    """
    Abstract base class used to implement weight initializers.

    This class defines the skeleton for weight initializers. Concrete implementations
    must provide specific strategies for initializing weights and a method to retrieve
    the configuration of the initializer.

    Methods
    -------
    initialize_weights(input_dim: int, n_units: int) -> np.array
        Abstract method. Must be implemented by subclasses to provide a strategy for initializing weights.
    
    get_config() -> dict
        Abstract method. Must be implemented by subclasses to provide a method for retrieving the configuration
        of the initializer.

    initialize_bias(n_units: int) -> np.array
        Class method that initializes bias to zeros. 

    Examples
    --------
    To create a custom initializer, subclass Initializer and implement the required methods:

    class MyInitializer(Initializer):
        def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
            # Custom strategy for weight initialization
            return np.random.randn(input_dim, n_units)

        def get_config(self) -> dict:
            # Custom configuration retrieval
            return {"description": "My custom weight initializer"}
    """
     
     
    @abstractmethod
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Abstract method to initialize weights.

        Subclasses should implement this method to provide a specific strategy for initializing weights.

        Parameters
        ----------
        input_dim : int
            The dimensionality of the input data.
        n_units : int
            The number of units in the layer for which the weights are initialized.

        Returns
        -------
        np.array
            The initialized weights.
        """
        pass
    
    
    @classmethod
    def get_config(self) -> dict:
        """
        Abstract method to get the configuration of the initializer.

        Subclasses should implement this method to provide a specific way to retrieve the configuration
        of the initializer, typically in the form of a dictionary.

        Returns
        -------
        dict
            A dictionary containing the configuration of the initializer.
        """
        {}

    
    @classmethod
    def initialize_bias(cls, n_units: int) -> np.array:
        """
        Initialize bias values to zero.

        This method provides a default way to initialize bias values, although subclasses might offer
        different strategies if needed.

        Parameters
        ----------
        n_units : int
            The number of units in the layer for which the bias is initialized.

        Returns
        -------
        np.array
            An array of zeros with shape (1, n_units).
        """
        return np.zeros((1, n_units))
