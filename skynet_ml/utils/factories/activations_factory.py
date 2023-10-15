from skynet_ml.nn.activations.base import BaseActivation
from skynet_ml.utils.factories.base import BaseFactory
from skynet_ml.nn.activations import activations_map
from typing import Optional, Union, Dict


class ActivationsFactory(BaseFactory):
    """
    ActivationsFactory is a concrete factory class for creating and retrieving instances of activation functions.

    It follows the Singleton and Factory patterns using BaseFactory, ensuring only one instance of a particular activation 
    function is created and reused.

    Args:
        BaseFactory (BaseFactory): Inherits the Singleton and Factory properties and methods from BaseFactory.
    """
    
    _object_map: Dict = activations_map
    
    
    def get_object(self, object_name: Optional[Union[str, BaseActivation]]) -> BaseActivation:    
        """
        Get or create an instance of the desired activation function.

        Args:
            object_name (Optional[Union[str, BaseActivation]]): Name or instance of the activation function.

        Raises:
            TypeError: Raised when the provided object_name is neither a string nor a BaseActivation instance.

        Returns:
            BaseActivation: Instance of the desired activation function.
        """
        
        if isinstance(object_name, str):
            return self.create_from_str(object_name)
        elif isinstance(object_name, BaseActivation):
            if object_name.name  not in self._cache:
                self._cache[object_name.name] = object_name
            return self._cache[object_name.name]  
        else:
            raise TypeError("The provided object_name must be either a string or a BaseActivation instance.")
