from skynet_ml.nn.initialization_methods.initializer import Initializer
from skynet_ml.nn.initialization_methods import INITIALIZERS_MAP, ACTIVATIONS_INITIALIZER_MAP
from typing import Union, Optional

class InitializerFactory:
    """
    Singleton factory class to generate and manage weight initializers for neural network layers.

    This class handles the creation and caching of initializer objects. It also offers
    functionality to fetch initializers based on the activation function of a layer, ensuring 
    that layers with specific activations get their recommended initializers.

    Attributes
    ----------
    _instance : InitializerFactory
        Singleton instance of the class.
    
    _initializers : dict
        Dictionary to cache created initializer objects.

    Methods
    -------
    get_initializer(initializer: Optional[Union[str, Initializer]] = None, activation: Optional[str] = None) -> Initializer:
        Returns an initializer based on the provided initializer name, initializer object, 
        or activation function.

    _create_initializer(name: str) -> Initializer:
        Create an initializer object based on the given name.

    """
    
    _instance = None
    _initializers = {}
    
    def __new__(cls) -> "InitializerFactory":
        """Creates or retrieves the single instance of the factory."""
        if cls._instance is None:
            cls._instance = super(InitializerFactory, cls).__new__(cls)
        return cls._instance
    
    
    def get_initializer(self, initializer: Optional[Union[str, Initializer]] = None, activation: Optional[str] = None) -> Initializer:
        """
        Fetch or create an initializer based on the provided parameters.

        Parameters
        ----------
        initializer : Union[str, Initializer], optional
            The name of the initializer or the initializer object. Default is None.
        
        activation : str, optional
            The name of the activation function for which the initializer is sought. Default is None.

        Returns
        -------
        Initializer
            The fetched or created initializer object.

        Raises
        ------
        TypeError
            If the provided arguments do not match the expected type or combination.
        
        """
        # If initializer is a str, we need to create the initialier object and cache it.
        if isinstance(initializer, str):
            key = initializer
            if key not in self._initializers:
                self._initializers[key] = self._create_initializer(key)
            return self._initializers[key]
        
        # If initializer is an object, then we put it in the dictionary
        elif isinstance(initializer, Initializer):
            key = str(initializer) + str(id(initializer))
            self._initializers[key] = initializer
            return initializer
        
        # If initializer is None, then we need to create the initializer object based on the activation function
        elif initializer is None and activation:
            initializer_name = ACTIVATIONS_INITIALIZER_MAP[activation]
            return self.get_initializer(initializer_name)
        
        # If initializer is neither a string nor an object and activation str is not present, then we raise an error
        else:
            raise TypeError(f"Invalid initializer type provided. Must be either string, Initializer object, or None with an activation string.")
    
    
    def _create_initializer(self, name: str) -> Initializer:
        """
        Create an initializer object based on the given name.

        Parameters
        ----------
        name : str
            The name of the initializer to be created.

        Returns
        -------
        Initializer
            The created initializer object.

        Raises
        ------
        ValueError
            If the provided initializer name is not found in the INITIALIZERS_MAP.

        """
        if name not in INITIALIZERS_MAP:
            raise ValueError(f"Unknown initializer: {name}")
        
        return INITIALIZERS_MAP[name]()
