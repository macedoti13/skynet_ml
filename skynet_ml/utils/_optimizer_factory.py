from skynet_ml.nn.optimizers.optimizer import Optimizer
from skynet_ml.nn.optimizers import OPTIMIZERS_MAP
from typing import Union, Optional

class OptimizerFactory:
    """
    Factory class to manage and retrieve instances of optimizers.
    
    This factory ensures that only one instance of an optimizer is maintained in memory and reused as needed. 
    It supports retrieval and caching of optimizers via both their name (string) or direct instance. Additionally, 
    it provides a mechanism to set a custom learning rate when creating an optimizer from its name.

    Attributes
    ----------
    _instance : OptimizerFactory or None
        Singleton instance of the factory.
    _optimizers : dict
        Cache for optimizer instances.

    Methods
    -------
    get_optimizer(optimizer, learning_rate=None)
        Retrieves or caches an optimizer based on the given input and optionally sets its learning rate.
    _create_optimizer(name, learning_rate=None)
        Internal method to instantiate a new optimizer based on its name and optionally set its learning rate.
    """
    
    _instance = None
    _optimizers = {}
    
    
    def __new__(cls) -> "OptimizerFactory":
        """Creates or retrieves the single instance of the factory."""
        if cls._instance is None:
            cls._instance = super(OptimizerFactory, cls).__new__(cls)
        return cls._instance
    
    
    def get_optimizer(self, optimizer: Union[str, Optimizer], learning_rate: Optional[float] = None) -> Optimizer:
        """
        Retrieves or caches an optimizer based on the given input and optionally sets its learning rate.
        
        This method can handle both string representations and instances of optimizers. If a string is 
        provided, the method looks up the associated optimizer in the cache or creates a new one, and then 
        optionally sets its learning rate. If an instance of an optimizer is provided, it caches and returns it.

        Parameters
        ----------
        optimizer : Union[str, Optimizer]
            The name of the desired optimizer or its direct instance.
        learning_rate : Optional[float], default=None
            The learning rate to set for the optimizer if created from its name.

        Returns
        -------
        Optimizer
            Instance of the requested optimizer.

        Raises
        ------
        TypeError
            If the provided `optimizer` argument is neither a string nor an Optimizer object.
        """
        # If the optimizer is a string, look it up in the cache or create a new instance.
        if isinstance(optimizer, str):
            key = optimizer
            if key not in self._optimizers:
                self._optimizers[key] = self._create_optimizer(key, learning_rate)
            return self._optimizers[key]
        
        # If the optimizer is an object, put it in the cache and return it.
        elif isinstance(optimizer, Optimizer):
            key = str(optimizer) + str(id(optimizer))
            self._optimizers[key] = optimizer
            return optimizer
        
        else:
            raise TypeError("optimizer must be a string or an Optimizer object.")
        
        
    def _create_optimizer(self, name: str, learning_rate: Optional[float] = None) -> Optimizer:
        """
        Internal method to instantiate a new optimizer based on its name and optionally set its learning rate.
        
        Given the name of an optimizer, it looks up the correct class in the OPTIMIZERS_MAP, instantiates it, 
        and then optionally sets its learning rate.

        Parameters
        ----------
        name : str
            Name of the desired optimizer.
        learning_rate : Optional[float], default=None
            The learning rate to set for the optimizer.

        Returns
        -------
        Optimizer
            New instance of the requested optimizer.

        Raises
        ------
        ValueError
            If the provided name does not match any known optimizer.
        """
        if name not in OPTIMIZERS_MAP:
            raise ValueError(f"Unknown optimizer: {name}")
        
        optimizer = OPTIMIZERS_MAP[name]()
        
        if learning_rate is not None:
            optimizer.learning_rate = learning_rate
            
        return optimizer