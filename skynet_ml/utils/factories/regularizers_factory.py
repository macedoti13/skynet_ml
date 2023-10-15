from skynet_ml.nn.regularizers.base import BaseRegularizer
from skynet_ml.utils.factories.base import BaseFactory
from skynet_ml.nn.regularizers import regularizers_map
from typing import Optional, Union, Dict


class RegularizersFactory(BaseFactory):
    """
    RegularizersFactory is a concrete factory class responsible for creating and retrieving instances of regularizers.

    This class makes use of the Singleton and Factory patterns as provided by BaseFactory, ensuring that only one
    instance of a particular regularizer is created and reused throughout the application's lifecycle.

    Attributes:
        _object_map (Dict): A dictionary that maps regularizer names to their respective class implementations.

    Args:
        BaseFactory (BaseFactory): Inherits the Singleton and Factory behaviors and methods from BaseFactory.
    """
    
    _object_map: Dict = regularizers_map


    def get_object(self, object_name: Optional[Union[str, BaseRegularizer]]) -> BaseRegularizer:
        """
        Retrieve or create an instance of the desired regularizer.

        Args:
            object_name (Optional[Union[str, BaseRegularizer]]): The name or instance of the regularizer.

        Raises:
            TypeError: If the provided object_name is neither a string nor a BaseRegularizer instance.

        Returns:
            BaseRegularizer: An instance of the requested regularizer.
        """
        
        if isinstance(object_name, str):
            return self.create_from_str(object_name)
        elif isinstance(object_name, BaseRegularizer):
            if object_name.name not in self._cache:
                self._cache[object_name.name] = object_name
            return self._cache[object_name.name]
        else:
            raise TypeError("The provided object_name must be either a string or a BaseRegularizer instance.")
