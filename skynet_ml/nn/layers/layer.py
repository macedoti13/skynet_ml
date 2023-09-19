from skynet_ml.nn.regularizers.regularizer import Regularizer
from skynet_ml.nn.activation_functions.activation import Activation
from skynet_ml.nn.initialization_methods.initializer import Initializer
from skynet_ml.utils._activation_factory import ActivationFactory
from skynet_ml.utils._initializer_factory import InitializerFactory
from skynet_ml.utils._regularizer_factory import RegularizerFactory
from typing import Union, Optional
from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    """
    Abstract base class representing a layer in a neural network.

    This class is intended to be subclassed by other specific types of layers, 
    providing a standardized interface for forward and backward operations, 
    weight management, and configuration retrieval.

    Attributes
    ----------
    n_units : int
        Number of units (neurons) in the layer.
    
    activation : Activation
        Activation function for the layer.
    
    initializer : Initializer
        Initialization method to set the weights and biases for the layer.
    
    regularizer : Regularizer, optional
        Regularization method for the layer's weights.
    
    has_bias : bool
        Whether the layer should have bias terms.
    
    input_dim : int, optional
        Dimensionality of the input data for the layer.
    
    is_initialized : bool
        Indicates if the layer's weights and biases have been initialized.
    
    Methods
    -------
    initialize():
        Initialize the layer's weights and biases.
    
    initialize_m():
        Initialize the momentum terms for the weights and biases.
    
    initialize_v():
        Initialize the velocity terms for the weights and biases.
    
    forward(x: np.array) -> np.array:
        Abstract method for the forward propagation.
    
    backward(dLda: np.array) -> np.array:
        Abstract method for the backward propagation.
    
    get_weights() -> dict:
        Abstract method to retrieve the layer's weights and biases.
    
    set_weights(weights: dict):
        Abstract method to set the layer's weights and biases.
    
    get_config() -> dict:
        Abstract method to retrieve the layer's configuration details.
    """

    def __init__(self, 
                 n_units: int, 
                 activation: Union[str, Activation] = "linear", 
                 initializer: Optional[Union[str, Initializer]] = None,
                 regularizer: Optional[Union[str, Regularizer]] = None,
                 has_bias: bool = True,
                 input_dim: int = None,
                ) -> None:
        """
        Initialize the Layer with the given parameters.

        Parameters
        ----------
        n_units : int
            Number of units (neurons) in the layer.
        
        activation : Union[str, Activation], optional
            Activation function's name or the function itself. Defaults to "linear".
        
        initializer : Optional[Union[str, Initializer]], optional
            Initialization method's name or the method itself. Defaults to None.
        
        regularizer : Optional[Union[str, Regularizer]], optional
            Regularization method's name or the method itself. Defaults to None.
        
        has_bias : bool, optional
            If set to True, the layer will have bias terms. Defaults to True.
        
        input_dim : int, optional
            Dimensionality of the input data for the layer. Defaults to None.
        """
        self.n_units = n_units
        self.activation = ActivationFactory().get_activation(activation)
        self.initializer = InitializerFactory().get_initializer(initializer, activation)
        self.regularizer = RegularizerFactory().get_regularizer(regularizer)
        self.has_bias = has_bias
        self.input_dim = input_dim
        self.is_initialized = False
        
        
    def initialize(self) -> None:
        """
        Initialize the layer's weights and biases using the provided initializer.
        """
        if not self.is_initialized:
            self.weights = self.initializer.initialize_weights(self.input_dim, self.n_units)
            self.biases = self.initializer.initialize_bias(self.n_units)
            self.is_initialized = True
    
    
    def initialize_m(self) -> None:
        """
        Initialize the m terms for the weights and biases to zeros.
        """
        self.mweights = np.zeros_like(self.weights)
        if self.has_bias:
            self.mbiases = np.zeros_like(self.biases)


    def initialize_v(self) -> None:
        """
        Initialize the v terms for the weights and biases to zeros.
        """
        self.vweights = np.zeros_like(self.weights)
        if self.has_bias:
            self.vbiases = np.zeros_like(self.biases)


    @abstractmethod
    def forward(self, x: np.array) -> np.array:
        """
        Abstract method for forward propagation through the layer.

        Parameters
        ----------
        x : np.array
            Input data to the layer.
        
        Returns
        -------
        np.array
            Output after processing the input data.
        """
        pass


    @abstractmethod
    def backward(self, dLda: np.array) -> np.array:
        """
        Abstract method for backward propagation through the layer.

        Parameters
        ----------
        dLda : np.array
            Gradient of the loss with respect to the layer's output.
        
        Returns
        -------
        np.array
            Gradient of the loss with respect to the layer's input.
        """
        pass


    @abstractmethod
    def get_weights(self) -> dict:
        """
        Abstract method to retrieve the layer's weights and biases.

        Returns
        -------
        dict
            Dictionary containing the layer's weights and biases.
        """
        pass


    @abstractmethod
    def set_weights(self, weights: dict) -> None:
        """
        Abstract method to set the layer's weights and biases.

        Parameters
        ----------
        weights : dict
            Dictionary containing the weights and biases to be set for the layer.
        """
        pass


    @abstractmethod
    def get_config(self) -> dict:
        """
        Abstract method to retrieve the layer's configuration details.

        Returns
        -------
        dict
            Dictionary containing the configuration details of the layer.
        """
        pass
