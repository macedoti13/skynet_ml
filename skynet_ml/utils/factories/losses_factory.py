from skynet_ml.utils.factories.factory import Factory
from skynet_ml.nn.losses.loss import Loss
from skynet_ml.nn.losses import LOSSES_MAP
from typing import Optional, Union, Dict, Any


class LossesFactory(Factory):
    """
    Singleton factory class to generate and manage Loss instances for evaluation and training of neural network models.

    This class handles the creation and caching of Loss objects. Its purpose is to provide an efficient way 
    to retrieve Loss instances without necessarily creating new ones every time, adhering to the Singleton pattern.

    Attributes
    ----------
    _OBJECT_MAP : Dict
        A predefined mapping of string identifiers to Loss classes available for creation. The actual mapping
        is imported from skynet_ml.nn.losses (LOSSES_MAP).

    _cache : Dict
        Dictionary to cache created Loss objects for efficient retrieval, preventing unnecessary instantiation.

    Methods
    -------
    get_object(object_name: Optional[Union[str, Loss]] = None, **kwargs: Any) -> Loss:
        Retrieves or creates a Loss object based on the provided object_name parameter. If the parameter is a string, 
        it tries to find or create a Loss with a corresponding name using optional keyword arguments. If the parameter 
        is a Loss instance, it simply returns the instance. Raises a TypeError if the parameter type is not valid.
    """
    
    
    _OBJECT_MAP: Dict = LOSSES_MAP
    
    
    def get_object(self, object_name: Optional[Union[str, Loss]] = None, **kwargs: Any) -> Loss:    
        """
        Retrieves or creates a Loss object based on the provided object_name parameter.
        
        Parameters
        ----------
        object_name : Optional[Union[str, Loss]]
            - If str: The name of the Loss function. The method looks up _OBJECT_MAP for 
                      a corresponding Loss class, instantiates it (if not already in cache), 
                      caches and returns it.
            - If Loss instance: The method returns this instance, caches it if not already cached.
            - If None: A TypeError is raised.
        **kwargs : Any
            Additional keyword arguments to be passed when initializing the loss object.
        
        Returns
        -------
        Loss
            The retrieved or newly created Loss object.
            
        Raises
        ------
        TypeError
            If object_name is not a string, a Loss instance, or None.
        """
        
        # If the object_name is a string, create a key for it (including configurations) and store it in the cache.
        if isinstance(object_name, str):
            key = object_name + str(kwargs)
            if key not in self._cache:
                self._cache[key] = self.create_object(object_name, **kwargs)
            return self._cache[key]
        
        # If the provided object_name is a Loss instance
        elif isinstance(object_name, Loss):
            key = str(type(object_name)) + str(object_name.get_config())  
            if key not in self._cache:
                # Cache the provided instance if no instance with this configuration is already cached
                self._cache[key] = object_name
            return self._cache[key]  # Return the cached instance with this configuration
        
        else:
            raise TypeError("The provided object_name must be either a string or a Loss instance.")
