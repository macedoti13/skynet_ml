from skynet_ml.utils.factories.factory import Factory
from skynet_ml.nn.initializers.initializer import Initializer
from skynet_ml.nn.initializers import INITIALIZERS_MAP, ACTIVATIONS_INITIALIZER_MAP
from typing import Optional, Union, Dict, Any


class InitializersFactory(Factory):
    """
    Factory class for creating and retrieving Initializer objects.

    InitializersFactory is a Factory design pattern class used for creating and retrieving
    instances of Initializer objects. It supports the creation of Initializer objects through
    names (strings) or directly using instances of Initializer classes.

    Attributes
    ----------
    _OBJECT_MAP : Dict
        A dictionary mapping from object names (strings) to Initializer classes.

    Methods
    -------
    get_object(object_name: Optional[Union[str, Initializer]] = None, **kwargs: Any) -> Initializer:
        Retrieves or creates an Initializer object.

    Examples
    --------
    >>> factory = InitializersFactory()
    >>> initializer = factory.get_object('NormalInitializer', mean=0, std=1)
    """
    
    
    _OBJECT_MAP: Dict = INITIALIZERS_MAP
    
    
    def get_object(self, object_name: Optional[Union[str, Initializer]] = None,  activation: Optional[str] = None, **kwargs: Any) -> Initializer:    
        """
        Retrieves or creates an Initializer object.

        Parameters
        ----------
        object_name : Union[str, Initializer], optional
            Name of the initializer (string) or an instance of an Initializer class. If provided a string,
            the method will create an object with a name matching the string. If provided an Initializer instance,
            the method returns the instance directly. Default is None.
        **kwargs : Any
            Additional keyword arguments for creating the Initializer object.

        Returns
        -------
        Initializer
            The retrieved or newly created Initializer object.

        Raises
        ------
        TypeError
            If object_name is not a string, an Initializer instance, or None.

        Examples
        --------
        >>> factory = InitializersFactory()
        >>> initializer = factory.get_object('NormalInitializer', mean=0, std=1)
        """
        
        # If the object_name is a string, create a key for it (including configurations) and store it in the cache.
        if isinstance(object_name, str):
            key = object_name + str(kwargs)
            if key not in self._cache:
                self._cache[key] = self.create_object(object_name, **kwargs)
            return self._cache[key]
        
        # If the provided object_name is a Initilizer instance
        elif isinstance(object_name, Initializer):
            key = str(type(object_name)) + str(object_name.get_config())  
            if key not in self._cache:
                # Cache the provided instance if no instance with this configuration is already cached
                self._cache[key] = object_name
            return self._cache[key]  # Return the cached instance with this configuration
        
        # If the provided object_name is None, create an Initializer object with the provided activation
        elif object_name is None and activation:
            initializer_name = ACTIVATIONS_INITIALIZER_MAP[activation]
            return self.get_object(initializer_name)
        
        else:
            raise TypeError("The provided object_name must be either a string or a Initializer instance.")
