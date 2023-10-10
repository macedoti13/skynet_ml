from skynet_ml.utils.factories.factory import Factory
from skynet_ml.nn.regularizers.regularizer import Regularizer
from skynet_ml.nn.regularizers import REGULARIZERS_MAP
from typing import Optional, Union, Dict, Any


class RegularizersFactory(Factory):
    """
    RegularizersFactory Class

    The RegularizersFactory is responsible for creating and managing instances of Regularizers. It supports
    the dynamic creation of Regularizer instances, identified by a given string (their name), or the retrieval
    of previously created instances. The factory leverages caching to efficiently manage and reuse Regularizer 
    instances.

    Parameters
    ----------
    _OBJECT_MAP : Dict
        A mapping from string identifiers to corresponding Regularizer classes.

    Methods
    -------
    get_object(object_name: Optional[Union[str, Regularizer]] = None, **kwargs: Any) -> Regularizer:
        Retrieves or creates a Regularizer instance based on the provided object_name and kwargs.

    Example
    -------
    >>> reg_factory = RegularizersFactory()
    >>> l1_reg = reg_factory.get_object('L1', lambda_val=0.01)
    >>> print(l1_reg)  # Should output an instance of L1 Regularizer with lambda_val set to 0.01.
    """
    
    
    _OBJECT_MAP: Dict = REGULARIZERS_MAP
    
    
    def get_object(self, object_name: Optional[Union[str, Regularizer]] = None, **kwargs: Any) -> Regularizer:    
        """
        Retrieves or creates a Regularizer object based on the provided parameters.
        
        If object_name is a string, this method looks up _OBJECT_MAP for a corresponding Regularizer class,
        instantiates it (if not already in cache), caches, and returns it.
        
        If object_name is a Regularizer instance, the method returns the instance directly.

        Parameters
        ----------
        object_name : Union[str, Regularizer], optional
            The name of the Regularizer (str) or a Regularizer instance. Default is None.
        kwargs : dict
            Additional keyword arguments to be passed to the Regularizer constructor.
            
        Returns
        -------
        Regularizer
            The retrieved or newly created Regularizer object.
            
        Raises
        ------
        TypeError
            If object_name is not a string, a Regularizer instance, or None.
        """
        
        # If the object_name is a string, create a key for it (including configurations) and store it in the cache.
        if isinstance(object_name, str):
            key = object_name + str(kwargs)
            if key not in self._cache:
                self._cache[key] = self.create_object(object_name, **kwargs)
            return self._cache[key]
        
        # If the provided object_name is a Regularizer instance
        elif isinstance(object_name, Regularizer):
            key = str(type(object_name)) + str(object_name.get_config())  
            if key not in self._cache:
                # Cache the provided instance if no instance with this configuration is already cached
                self._cache[key] = object_name
            return self._cache[key]  # Return the cached instance with this configuration
        
        elif object_name is None:
            return None
        
        # If the provided object_name is not a string or a Regularizer instance, raise a TypeError.
        else:
            raise TypeError("The provided object_name must be either a string or a Regularizer instance.")
