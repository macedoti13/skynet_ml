from typing import Optional, Union, Dict, Type, Any
from abc import ABC, abstractmethod


class BaseFactory(ABC):
    """
    BaseFactory is an abstract base class implementing the Singleton and Factory patterns. 
    This class ensures that only one instance of a specific subclass is created, and provides 
    a method to instantiate objects from registered subclasses based on their string name.

    Args:
        ABC (ABC): Inherits properties and methods from the ABC (Abstract Base Class).
    """

    _instances: Dict["BaseFactory", "BaseFactory"] = {}
    _cache = {}
    _object_map: Dict[str, Type[Any]] = {}


    def __new__(cls) -> "BaseFactory":
        """
        Ensure a single instance is created for the factory subclass.

        Returns:
            BaseFactory: Instance of the specific factory subclass.
        """        
        if cls not in cls._instances:
            cls._instances[cls] = super(BaseFactory, cls).__new__(cls)
        return cls._instances[cls]


    @abstractmethod
    def get_object(self, object_name: Optional[Union[str, Any]]) -> Any:
        """
        Abstract method to get or create an instance of the desired object.

        Args:
            object_name (Optional[Union[str, Any]]): Name or type of the object to get or create.

        Returns:
            Any: Instance of the desired object.
        """
        pass


    @classmethod
    def create_from_str(cls, name: str) -> Any:
        """
        Create or retrieve a cached instance of an object based on its string name.

        Args:
            name (str): Name of the object to be created or retrieved.

        Raises:
            ValueError: Raised when the desired object name is not found in the registered object map.

        Returns:
            Any: Instance of the object with the given name.
        """
        if name not in cls._object_map:
            raise ValueError(f"Object with name {name} not found in objects map!")
        
        obj = cls._object_map[name]()
        
        if obj.name not in cls._cache:
            cls._cache[obj.name] = obj
        return cls._cache[obj.name]
