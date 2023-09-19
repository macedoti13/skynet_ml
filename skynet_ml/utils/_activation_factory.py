from skynet_ml.nn.activation_functions.activation import Activation
from skynet_ml.nn.activation_functions import ACTIVATIONS_MAP
from typing import Union

class ActivationFactory:
    """
    Factory class for managing and creating instances of `Activation` objects.

    Activation functions introduce non-linearity into the model. This factory ensures
    that for a given type of activation function, only one instance is created and reused.
    It provides methods to retrieve or create activation instances based on string names 
    or direct `Activation` objects.

    The class follows the Singleton design pattern, ensuring that only a single instance
    of the factory exists throughout the runtime.

    Attributes
    ----------
    _instance : ActivationFactory
        Holds the single instance of ActivationFactory.
        
    _activations : dict
        Dictionary mapping activation names or object identifiers to their respective 
        instances.

    Methods
    -------
    get_activation(activation: Union[str, Activation]) -> Activation:
        Retrieves or creates and caches the appropriate `Activation` instance based 
        on the given input.

    _create_activation(name: str) -> Activation:
        Creates an instance of the specified activation function based on its name.

    """
    
    _instance = None
    _activations = {}
    
    
    def __new__(cls) -> "ActivationFactory":
        """Creates or retrieves the single instance of the factory."""
        if cls._instance is None:
            cls._instance = super(ActivationFactory, cls).__new__(cls)
        return cls._instance
    
    
    def get_activation(self, activation: Union[str, Activation]) -> Activation:
        """
        Retrieve or create the appropriate `Activation` instance.

        This method checks the type of the provided activation argument. If it's a string, 
        the method looks up the respective activation instance or creates a new one. If it's 
        an `Activation` object, the method caches it and returns it.

        Parameters
        ----------
        activation : Union[str, Activation]
            Activation function name as a string or an instance of the `Activation` class.

        Returns
        -------
        Activation
            An instance of the specified activation function.

        Raises
        ------
        TypeError
            If the provided activation argument is neither a string nor an `Activation` object.
        """
        # If activation is a string, then we need to create the activation
        if isinstance(activation, str):
            key = activation 
            if key not in self._activations:
                self._activations[key] = self._create_activation(key)
            return self._activations[key]
        
        # If activation is an object, then we put it in the dictionary
        elif isinstance(activation, Activation):
            key = str(activation) + str(id(activation))
            self._activations[key] = activation
            return activation
        
        # If activation is neither a string nor an object, then we raise an error
        else:
            raise TypeError("Invalid activation type provided. Must be either string or BaseActivation object.")
        
        
    def _create_activation(self, name: str) -> Activation:
        """
        Create an instance of the specified activation function based on its name.

        This is a helper method used to instantiate a new activation function 
        if it does not already exist in the `_activations` cache.

        Parameters
        ----------
        name : str
            Name of the activation function to instantiate.

        Returns
        -------
        Activation
            A new instance of the specified activation function.

        Raises
        ------
        ValueError
            If the provided activation name does not map to any known activation functions.
        """
        if name not in ACTIVATIONS_MAP:
            raise ValueError(f"Unknown activation: {name}")
        
        return ACTIVATIONS_MAP[name]()
