from skynet_ml.utils.factories.base import BaseFactory
from skynet_ml.metrics.base import BaseMetric
from skynet_ml.metrics import metrics_map
from typing import Optional, Union, Dict


class MetricsFactory(BaseFactory):
    """
    MetricsFactory is a concrete factory class responsible for creating and retrieving instances of metric functions.

    This class leverages the Singleton and Factory patterns using BaseFactory, ensuring only one instance of a 
    particular metric function is created and reused.

    Attributes:
        _object_map (Dict): A dictionary that maps metric names to their respective class implementations.

    Args:
        BaseFactory (BaseFactory): Inherits the Singleton and Factory behaviors and methods from BaseFactory.
    """

    _object_map: Dict = metrics_map


    def get_object(self, object_name: Optional[Union[str, BaseMetric]]) -> BaseMetric:
        """
        Get or create an instance of the desired metric function.

        Args:
            object_name (Optional[Union[str, BaseMetric]]): The name or instance of the metric function.

        Raises:
            TypeError: If the provided object_name is neither a string nor a BaseMetric instance.

        Returns:
            BaseMetric: An instance of the requested metric function.
        """

        if isinstance(object_name, str):
            return self.create_from_str(object_name)
        elif isinstance(object_name, BaseMetric):
            if object_name.name not in self._cache:
                self._cache[object_name.name] = object_name
            return self._cache[object_name.name]
        else:
            raise TypeError("The provided object_name must be either a string or a BaseMetric instance.")
