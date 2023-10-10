from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Type, Any
import inspect


class Factory(ABC):
    """
    Singleton factory base class to generate and manage objects for neural network components.

    This class handles the creation and caching of objects like initializers, activations, optimizers, etc.
    The actual behavior and logic for creating and retrieving these objects should be implemented in the derived 
    factories. This base class mainly provides the structure and common patterns, such as singleton instantiation 
    and object caching.

    Attributes
    ----------
    _instance : Dict[Type[Factory], Factory]
        Dictionary holding singleton instances of derived factory classes.
        
    _cache : dict
        Dictionary to cache created objects for efficient retrieval.

    _OBJECT_MAP : Dict[str, Type[Any]]
        Mapping of string names to actual classes or functions. This should be overridden by derived classes
        to provide the appropriate mappings for their specific objects.

    Methods
    -------
    get_object(obj_name: Optional[Union[str, Any]] = None, **kwargs: Any) -> Any:
        Abstract method to fetch or create an object based on provided parameters. This should be implemented 
        by derived factory classes.

    _create_object(name: str) -> Any:
        Create an object based on the given name using the _OBJECT_MAP. Raises a ValueError if the name is 
        not found in the map.
    """

    
    _instances: Dict["Factory", "Factory"] = {}
    _cache = {}
    _OBJECT_MAP: Dict[str, Type[Any]] = {}
    
    
    def __new__(cls) -> "Factory":
        """
        Creates or retrieves the single instance of the factory.
        """
        
        if cls not in cls._instances:
            cls._instances[cls] = super(Factory, cls).__new__(cls)
        return cls._instances[cls]
       
       
    @abstractmethod
    def get_object(self, object_name: Optional[Union[str, Any]] = None, **kwargs: Any) -> Any:
        """
        Fetch or create an object based on the provided parameters.

        Parameters
        ----------
        obj_name : Union[str, Any], optional
            The name or instance of the object to be fetched or created. Depending on the specific factory, this might
            be an initializer, an activation function, an optimizer, etc. Default is None.

        **kwargs : Any
            Additional keyword arguments that may be relevant for specific derived factories.

        Returns
        -------
        Any
            The fetched or created object.

        Raises
        ------
        TypeError
            If the provided arguments do not match the expected type or combination for the specific derived factory.

        ValueError
            If the provided obj_name is not found in the _OBJECT_MAP for the specific derived factory.
        """
        
        pass
    
    
    def create_object(self, name: str, **kwargs: Any) -> Any:
        """
        Create an object based on the given name.

        Parameters
        ----------
        name : str
            The name of the object to be created.

        Returns
        -------
        Any
            The created object.

        Raises
        ------
        ValueError
            If the provided name is not found in the _OBJECT_MAP.
        """
        
        if name not in self._OBJECT_MAP:
            raise ValueError(f"Object with name {name} not found")
        
        cls = self._OBJECT_MAP[name]
        
        constructor_args = inspect.signature(cls.__init__).parameters.keys()
        kwargs_for_constructor = {k: v for k, v in kwargs.items() if k in constructor_args}
        
        return cls(**kwargs_for_constructor)
