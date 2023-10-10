from skynet_ml.utils.factories.factory import Factory
from skynet_ml.nn.activations.activation import Activation
from skynet_ml.nn.activations import ACTIVATIONS_MAP
from typing import Optional, Union, Dict, Any


class ActivationsFactory(Factory):
    """
    Factory class to handle the instantiation of activation functions in the neural network.

    ActivationsFactory allows for the easy retrieval and instantiation of activation 
    function objects, providing a mechanism to obtain these objects through a 
    string identifier, or directly passing an activation object.

    Attributes
    ----------
    _OBJECT_MAP : Dict
        A mapping between activation function names (str) and their corresponding 
        class objects. The mapping is expected to be provided by the ACTIVATIONS_MAP.

    Methods
    -------
    get_object(object_name: Union[str, Activation], **kwargs: Any) -> Activation
        Retrieves or creates an Activation object based on the provided object_name parameter.

    Example
    -------
    >>> activations_factory = ActivationsFactory()
    >>> sigmoid_activation = activations_factory.get_object("Sigmoid")
    >>> custom_activation = CustomActivation()
    >>> cached_custom_activation = activations_factory.get_object(custom_activation)
    """
    
    _OBJECT_MAP: Dict = ACTIVATIONS_MAP
    
    
    def get_object(self, object_name: Optional[Union[str, Activation]] = None, **kwargs: Any) -> Activation:    
        """
        Retrieves or creates a Activation object based on the provided object_name parameter.
        
        If object_name is a string, the method looks up _OBJECT_MAP for a corresponding Activation class,
        instantiates it (if not already in cache), caches and returns it.
        
        If object_name is a Activation instance, the method returns the instance directly.
        
        Parameters
        ----------
        object_name : Union[str, Activation], optional
            The name of the Activation (str) or a Activation instance. If it's a string, the method attempts
            to create a Activation object with a name matching the string. Default is None.
            
        Returns
        -------
        Activation
            The retrieved or newly created Activation object.
            
        Raises
        ------
        TypeError
            If object_name is not a string, a Activation instance, or None.
        """
        
        # If the object_name is a string, create a key for it (including configurations) and store it in the cache.
        if isinstance(object_name, str):
            key = object_name + str(kwargs)
            if key not in self._cache:
                self._cache[key] = self.create_object(object_name, **kwargs)
            return self._cache[key]
        
        # If the provided object_name is a Activation instance
        elif isinstance(object_name, Activation):
            key = str(type(object_name)) + str(object_name.get_config())  
            if key not in self._cache:
                # Cache the provided instance if no instance with this configuration is already cached
                self._cache[key] = object_name
            return self._cache[key]  # Return the cached instance with this configuration
        
        # If the provided object_name is not a string or a Activation instance, raise a TypeError.
        else:
            raise TypeError(f"The provided object_name must be either a string or a Activation instance, got {type(object_name)}.")
