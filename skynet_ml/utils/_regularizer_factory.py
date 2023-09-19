from skynet_ml.nn.regularizers.regularizer import Regularizer
from skynet_ml.nn.regularizers import REGULARIZERS_MAP
from typing import Union, Optional

class RegularizerFactory:
    """
    Factory class to create and cache regularizer objects.

    The RegularizerFactory class provides a mechanism to create regularizer objects based on 
    their string names, and caches them to avoid unnecessary creation of regularizer objects 
    that have been previously created.

    It follows the Singleton pattern to ensure that there's a single instance of the factory 
    across the program, and every request for a regularizer object goes through this single 
    instance.

    Attributes
    ----------
    _instance : RegularizerFactory
        The single instance of the factory class.
    
    _regularizers : dict
        Dictionary to cache created regularizer objects.

    Methods
    -------
    get_regularizer(regularizer: Optional[Union[str, Regularizer]] = None) -> Union[Regularizer, None]:
        Retrieve or create the requested regularizer object.
    
    _create_regularizer(name: str) -> Regularizer:
        Create a regularizer object based on its string name.

    """

    _instance = None
    _regularizers = {}
    
    def __new__(cls) -> "RegularizerFactory":
        """Creates or retrieves the single instance of the factory."""
        if cls._instance is None:
            cls._instance = super(RegularizerFactory, cls).__new__(cls)
        return cls._instance
    
    
    def get_regularizer(self, regularizer: Optional[Union[str, Regularizer]] = None) -> Union[Regularizer, None]:
        """
        Retrieve or create the requested regularizer object.

        Parameters
        ----------
        regularizer : Optional[Union[str, Regularizer]], optional
            Name of the regularizer as a string or the regularizer object itself.
            Default is None.

        Returns
        -------
        Regularizer or None
            The requested regularizer object if found or created, else None.

        Raises
        ------
        TypeError
            If an invalid type is provided for the regularizer.
        """
        # If a regularizer is provided as a string, we need to create the regularizer object and cache it.
        if isinstance(regularizer, str):
            key = regularizer
            if key not in self._regularizers:
                self._regularizers[key] = self._create_regularizer(key)
            return self._regularizers[key]
        
        # If a regularizer is provided as an object, then we put it in the dictionary
        elif isinstance(regularizer, Regularizer):
            key = str(regularizer) + str(id(regularizer))
            self._regularizers[key] = regularizer
            return regularizer
        
        # If a regularizer is None, then we return None (no regularizer or will be provided in the model class)
        elif regularizer is None:
            return None
        
        else:
            raise TypeError("Invalid regularizer type provided.")
        
        
    def _create_regularizer(self, name: str) -> Regularizer:
        """
        Create a regularizer object based on its string name.

        Parameters
        ----------
        name : str
            Name of the regularizer as a string.

        Returns
        -------
        Regularizer
            The created regularizer object.

        Raises
        ------
        ValueError
            If an invalid name is provided for the regularizer.
        """
        if name not in REGULARIZERS_MAP:
            raise ValueError(f"Invalid regularizer {name}")
    
        return REGULARIZERS_MAP[name]()
