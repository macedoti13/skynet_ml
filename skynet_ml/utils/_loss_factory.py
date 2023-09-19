from skynet_ml.nn.loss_functions.loss import Loss
from skynet_ml.nn.loss_functions import LOSS_MAP
from typing import Union

class LossFactory:
    """
    Factory class to manage and retrieve instances of loss functions.
    
    This factory ensures that only one instance of a loss function is maintained in memory and reused as needed. 
    It supports retrieval and caching of loss functions via both their name (string) or direct instance.

    Attributes
    ----------
    _instance : LossFactory or None
        Singleton instance of the factory.
    _losses : dict
        Cache for loss function instances.

    Methods
    -------
    get_loss(loss)
        Retrieves or caches a loss function based on the given input.
    _create_loss(name)
        Internal method to instantiate a new loss function based on its name.
    """
    
    _instance = None
    _losses = {}
    
    
    def __new__(cls) -> "LossFactory":
        """Creates or retrieves the single instance of the factory."""
        if cls._instance is None:
            cls._instance = super(LossFactory, cls).__new__(cls)
        return cls._instance
    
    
    def get_loss(self, loss: Union[str, Loss]) -> Loss:
        """
        Retrieves or caches a loss function based on the given input.
        
        This method can handle both string representations and instances of loss functions. If a string is 
        provided, the method looks up the associated loss function in the cache or creates a new one. If an 
        instance of a loss function is provided, it caches and returns it.

        Parameters
        ----------
        loss : Union[str, Loss]
            The name of the desired loss function or its direct instance.

        Returns
        -------
        Loss
            Instance of the requested loss function.

        Raises
        ------
        TypeError
            If the provided `loss` argument is neither a string nor a Loss object.
        """
        # If the loss is a string, look it up in the cache or create a new instance.
        if isinstance(loss, str):
            key = loss
            if key not in self._losses:
                self._losses[key] = self._create_loss(key)
            return self._losses[key]
        
        # If the loss is an object, put it in the cache and return it.
        elif isinstance(loss, Loss):
            key = str(loss) + str(id(loss))
            self._losses[key] = loss
            return loss
        
        # If the loss is neither a string nor an object, raise an error.
        else:
            raise TypeError("loss must be a string or a Loss object.")
        
        
    def _create_loss(self, name: str) -> Loss:
        """
        Internal method to instantiate a new loss function based on its name.
        
        Given the name of a loss function, it looks up the correct class in the LOSS_MAP and instantiates it.

        Parameters
        ----------
        name : str
            Name of the desired loss function.

        Returns
        -------
        Loss
            New instance of the requested loss function.

        Raises
        ------
        ValueError
            If the provided name does not match any known loss function.
        """
        if name not in LOSS_MAP:
            raise ValueError(f"Unknown loss function: {name}")
        
        return LOSS_MAP[name]()
