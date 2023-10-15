from skynet_ml.nn.optimizers.base import BaseOptimizer
from skynet_ml.utils.factories.base import BaseFactory
from skynet_ml.nn.optimizers import optimizers_map
from typing import Optional, Union, Dict


class OptimizersFactory(BaseFactory):
    """
    OptimizersFactory is a concrete factory class responsible for creating and retrieving instances of optimizers.

    This class leverages the Singleton and Factory patterns using BaseFactory, ensuring only one instance of a 
    particular optimizer is created and reused.

    Attributes:
        _object_map (Dict): A dictionary that maps optimizer names to their respective class implementations.

    Args:
        BaseFactory (BaseFactory): Inherits the Singleton and Factory behaviors and methods from BaseFactory.
    """

    _object_map: Dict = optimizers_map


    def get_object(self, object_name: Optional[Union[str, BaseOptimizer]]) -> BaseOptimizer:
        """
        Get or create an instance of the desired optimizer.

        Args:
            object_name (Optional[Union[str, BaseOptimizer]]): The name or instance of the optimizer.

        Raises:
            TypeError: If the provided object_name is neither a string nor a BaseOptimizer instance.

        Returns:
            BaseOptimizer: An instance of the requested optimizer.
        """

        if isinstance(object_name, str):
            return self.create_from_str(object_name)
        elif isinstance(object_name, BaseOptimizer):
            if object_name.name not in self._cache:
                self._cache[object_name.name] = object_name
            return self._cache[object_name.name]
        else:
            raise TypeError("The provided object_name must be either a string or a BaseOptimizer instance.")
