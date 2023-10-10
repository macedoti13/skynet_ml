from skynet_ml.utils.factories.factory import Factory
from skynet_ml.metrics.metric import Metric
from skynet_ml.metrics import METRICS_MAP
from typing import Optional, Union, Dict, Any


class MetricsFactory(Factory):
    """
    Singleton factory class to generate and manage Metric instances for evaluation of neural network models.

    This class handles the creation and caching of Metric objects. Its purpose is to provide an efficient way 
    to retrieve Metric instances without necessarily creating new ones every time, adhering to the Singleton pattern.

    Attributes
    ----------
    _OBJECT_MAP : Dict
        A predefined mapping of string identifiers to Metric classes available for creation. The actual mapping
        is imported from skynet_ml.metrics (METRICS_MAP).
    
    _cache : Dict
        Dictionary to cache created Metric objects for efficient retrieval, preventing unnecessary instantiation.

    Methods
    -------
    get_object(object_name: Optional[Union[str, Metric]] = None) -> Metric:
        Retrieves or creates a Metric object based on the provided object_name parameter. If the parameter is a string, 
        it tries to find or create a Metric with a corresponding name. If the parameter is a Metric instance, it simply 
        returns the instance. Raises a TypeError if the parameter type is not valid.

    """
    
    _OBJECT_MAP: Dict = METRICS_MAP
    
    
    def get_object(self, object_name: Optional[Union[str, Metric]] = None, **kwargs: Any) -> Metric:    
        """
        Retrieves or creates a Metric object based on the provided object_name parameter.
        
        If object_name is a string, the method looks up _OBJECT_MAP for a corresponding Metric class,
        instantiates it (if not already in cache), caches and returns it.
        
        If object_name is a Metric instance, the method returns the instance directly.
        
        Parameters
        ----------
        object_name : Union[str, Metric], optional
            The name of the Metric (str) or a Metric instance. If it's a string, the method attempts
            to create a Metric object with a name matching the string. Default is None.
            
        Returns
        -------
        Metric
            The retrieved or newly created Metric object.
            
        Raises
        ------
        TypeError
            If object_name is not a string, a Metric instance, or None.
        """
        
        # If the object_name is a string, create a key for it (including configurations) and store it in the cache.
        if isinstance(object_name, str):
            key = object_name + str(kwargs)
            if key not in self._cache:
                self._cache[key] = self.create_object(object_name, **kwargs)
            return self._cache[key]
        
        # If the provided object_name is a Metric instance
        elif isinstance(object_name, Metric):
            key = str(type(object_name)) + str(object_name.get_config())  
            if key not in self._cache:
                # Cache the provided instance if no instance with this configuration is already cached
                self._cache[key] = object_name
            return self._cache[key]  # Return the cached instance with this configuration
        
        # If the provided object_name is not a string or a Metric instance, raise a TypeError.
        else:
            raise TypeError("The provided object_name must be either a string or a Metric instance.")
