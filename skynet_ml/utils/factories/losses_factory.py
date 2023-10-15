from skynet_ml.utils.factories.base import BaseFactory
from skynet_ml.nn.losses.base import BaseLoss
from skynet_ml.nn.losses import losses_map
from typing import Optional, Union, Dict


class LossesFactory(BaseFactory):
    """
    LossesFactory is a concrete factory class for creating and retrieving instances of loss functions.

    This class follows the Singleton and Factory patterns using BaseFactory, ensuring only one instance of a 
    particular loss function is created and reused.

    Attributes:
        _object_map (Dict): A mapping of loss function names to their respective class types.

    Args:
        BaseFactory (BaseFactory): Inherits the Singleton and Factory properties and methods from BaseFactory.
    """

    _object_map: Dict = losses_map


    def get_object(self, object_name: Optional[Union[str, BaseLoss]]) -> BaseLoss:
        """
        Get or create an instance of the desired loss function.

        Args:
            object_name (Optional[Union[str, BaseLoss]]): Name or instance of the loss function.

        Raises:
            TypeError: Raised when the provided object_name is neither a string nor a BaseLoss instance.

        Returns:
            BaseLoss: Instance of the desired loss function.
        """

        if isinstance(object_name, str):
            return self.create_from_str(object_name)
        elif isinstance(object_name, BaseLoss):
            if object_name.name not in self._cache:
                self._cache[object_name.name] = object_name
            return self._cache[object_name.name]  
        else:
            raise TypeError("The provided object_name must be either a string or a BaseLoss instance.")
