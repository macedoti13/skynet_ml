from skynet_ml.nn.initializers.base import BaseInitializer
from skynet_ml.nn.initializers import initializers_map
from skynet_ml.utils.factories.base import BaseFactory
from typing import Optional, Union, Dict


class InitializersFactory(BaseFactory):
    """
    InitializersFactory is a concrete factory class for creating and retrieving instances of weight initializers.

    It follows the Singleton and Factory patterns using BaseFactory, ensuring only one instance of a particular 
    weight initializer is created and reused.

    Attributes:
        _object_map (Dict): A mapping of initializer names to their respective class types.

    Args:
        BaseFactory (BaseFactory): Inherits the Singleton and Factory properties and methods from BaseFactory.
    """
    _object_map: Dict = initializers_map


    def get_object(self, object_name: Optional[Union[str, BaseInitializer]]) -> BaseInitializer:
        """
        Get or create an instance of the desired weight initializer.

        Args:
            object_name (Optional[Union[str, BaseInitializer]]): Name or instance of the weight initializer.

        Raises:
            TypeError: Raised when the provided object_name is neither a string nor a BaseInitializer instance.

        Returns:
            BaseInitializer: Instance of the desired weight initializer.
        """

        if isinstance(object_name, str):
            return self.create_from_str(object_name)
        elif isinstance(object_name, BaseInitializer):
            if object_name.name not in self._cache:
                self._cache[object_name.name] = object_name
            return self._cache[object_name.name]  
        else:
            raise TypeError("The provided object_name must be either a string or a BaseInitializer instance.")
