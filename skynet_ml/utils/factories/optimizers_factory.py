from skynet_ml.utils.factories.factory import Factory
from skynet_ml.nn.optimizers.optimizer import Optimizer
from skynet_ml.nn.optimizers import OPTIMIZERS_MAP
from typing import Optional, Union, Dict, Any


class OptimizersFactory(Factory):
    """
    Singleton factory class to generate and manage Optimizer instances for training of neural network models.

    This class handles the creation and caching of Optimizer objects. It serves to provide an efficient way 
    to retrieve Optimizer instances without necessarily creating new ones every time, adhering to the Singleton pattern.

    Attributes
    ----------
    _OBJECT_MAP : Dict
        A predefined mapping of string identifiers to Optimizer classes available for creation. The actual mapping
        is imported from skynet_ml.optimizers (OPTIMIZERS_MAP).
    
    _cache : Dict
        Dictionary to cache created Optimizer objects for efficient retrieval, preventing unnecessary instantiation.

    Methods
    -------
    get_object(object_name: Optional[Union[str, Optimizer]] = None, **kwargs: Any) -> Optimizer:
        Retrieves or creates an Optimizer object based on the provided object_name parameter. If the parameter is a string, 
        it tries to find or create an Optimizer with a corresponding name. If the parameter is an Optimizer instance, it simply 
        returns the instance. Raises a TypeError if the parameter type is not valid.

    Examples
    --------
    >>> optim_factory = OptimizersFactory()
    >>> sgd_optimizer = optim_factory.get_object("SGD", learning_rate=0.01)
    >>> type(sgd_optimizer)
    <class 'skynet_ml.nn.optimizers.sgd.SGD'>
    """
    
    
    _OBJECT_MAP: Dict = OPTIMIZERS_MAP
    
    
    def get_object(self, object_name: Optional[Union[str, Optimizer]] = None, **kwargs: Any) -> Optimizer:    
        """
        Retrieves or creates an Optimizer object based on the provided object_name parameter.

        If object_name is a string, this method looks up `_OBJECT_MAP` for a corresponding Optimizer class,
        instantiates it (if not already in cache) with the provided kwargs, caches, and returns it.
        
        If object_name is an Optimizer instance, the method returns the instance directly after caching it
        if an instance with the same configuration is not already cached.

        Parameters
        ----------
        object_name : Optional[Union[str, Optimizer]]
            The name of the Optimizer (str) or an Optimizer instance. If it's a string, the method attempts
            to create an Optimizer object with a name matching the string. Default is None.
        **kwargs : Any
            Arbitrary keyword arguments that will be passed to the initializer of the Optimizer object
            if it needs to be created.

        Returns
        -------
        Optimizer
            The retrieved or newly created Optimizer object.

        Raises
        ------
        TypeError
            If object_name is neither a string, an Optimizer instance, nor None.

        Examples
        --------
        >>> optim_factory = OptimizersFactory()
        >>> sgd_optimizer = optim_factory.get_object("SGD", learning_rate=0.01)
        >>> type(sgd_optimizer)
        <class 'skynet_ml.nn.optimizers.sgd.SGD'>
        
        >>> sgd_instance = SGD(learning_rate=0.01)
        >>> retrieved_instance = optim_factory.get_object(sgd_instance)
        >>> retrieved_instance == sgd_instance  # Will return True since it retrieves the same instance
        True
        """
        # If the object_name is a string, create a key for it (including configurations) and store it in the cache.
        if isinstance(object_name, str):
            key = object_name + str(kwargs)
            if key not in self._cache:
                self._cache[key] = self.create_object(object_name, **kwargs)
            return self._cache[key]
        
        # If the provided object_name is a Loss instance
        elif isinstance(object_name, Optimizer):
            key = str(type(object_name)) + str(object_name.get_config())  
            if key not in self._cache:
                # Cache the provided instance if no instance with this configuration is already cached
                self._cache[key] = object_name
            return self._cache[key]  # Return the cached instance with this configuration
        
        else:
            raise TypeError("The provided object_name must be either a string or a Optimizer instance.")
