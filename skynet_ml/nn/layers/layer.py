from skynet_ml.utils.factories import ActivationsFactory, RegularizersFactory, InitializersFactory
from skynet_ml.nn.regularizers.regularizer import Regularizer
from skynet_ml.nn.initializers.initializer import Initializer
from skynet_ml.nn.activations.activation import Activation
from typing import Union, Optional
from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    """
    Abstract base class for defining neural network layers.

    This class should be inherited by any layer that intends to be used within a neural network model. 
    It contains method signatures for initialization, forward pass computations, backward pass gradient calculations,
    and weight management, which must be implemented by any concrete subclass.

    Attributes
    ----------
    n_units : int
        Number of units or neurons in the layer.
    activation : Activation
        Activation function instance used for transforming layer output.
    initializer : Initializer
        Initializer instance used to set the initial weights of the layer.
    regularizer : Regularizer
        Regularizer instance used to apply regularization to the layer's weights.
    has_bias : bool
        Flag indicating whether the layer should have bias terms.
    input_dim : int, optional
        Dimensionality of input data received by the layer. Required for input layer.
    is_initialized : bool
        Flag indicating whether the layer's weights have been initialized.

    Methods
    -------
    initialize() -> None:
        Initializes the layer's weights and biases using the provided initializer. Must be called before the layer is used.

    initialize_momentum() -> None:
        Initializes momentum terms for the layer's weights and biases. Required for optimization algorithms using momentum.

    initialize_velocity() -> None:
        Initializes velocity terms for the layer's weights and biases. Required for optimization algorithms using velocity.

    forward(x: np.array) -> np.array:
        Computes the layer's output given its input. To be implemented by subclass.

    backward(dl_da: np.array) -> np.array:
        Computes the gradient of the loss with respect to the layer's input, given the gradient of the loss with respect
        to the layer's output. To be implemented by subclass.

    get_weights() -> dict:
        Retrieves the layer's weights and biases as a dictionary. To be implemented by subclass.

    set_weights(weights: dict) -> None:
        Sets the layer's weights and biases from a dictionary. To be implemented by subclass.

    get_config() -> dict:
        Retrieves a dictionary containing the configuration of the layer. To be implemented by subclass.
    """
    
    
    def __init__(self, 
                 n_units: int, 
                 activation: Union[str, Activation],
                 initializer: Optional[Union[str, Initializer]] = None,
                 regularizer: Optional[Union[str, Regularizer]] = None,
                 has_bias: Optional[bool] = True,
                 input_dim: Optional[int] = None,
                ) -> None:
        
        self.n_units = n_units
        self.activation = ActivationsFactory().get_object(activation)
        self.initializer = InitializersFactory().get_object(initializer, activation)
        self.regularizer = RegularizersFactory().get_object(regularizer)
        self.has_bias = has_bias
        self.input_dim = input_dim
        self.is_initialized = False
        
        
    def initialize(self) -> None:
        """
        Initializes the layer's weights and biases using the designated initializer.
        This method must be called before the layer is used for the first time.
        """
        
        if not self.is_initialized:
            self.weights = self.initializer.initialize_weights(self.input_dim, self.n_units)
            self.bias = self.initializer.initialize_bias(self.n_units)
            self.is_initialized = True
            
            
    def initialize_momentum(self) -> None:
        """
        Initializes momentum terms for the layer's weights and biases to zero.
        This is essential for optimization algorithms that utilize momentum.
        """
        self.m_weights = np.zeros_like(self.weights)
        if self.has_bias:
            self.m_bias = np.zeros_like(self.bias)
            
            
    def initialize_velocity(self) -> None:
        """
        Initializes velocity terms for the layer's weights and biases to zero.
        Necessary for optimization algorithms that use velocity, like RMSprop or Adam.
        """
        self.v_weights = np.zeros_like(self.weights)
        if self.has_bias:
            self.v_bias = np.zeros_like(self.bias)
            
            
    @abstractmethod
    def forward(self, x: np.array) -> np.array:
        """
        Computes and returns the output of the layer for a given input.
        This abstract method must be implemented by all subclasses.

        Parameters:
        x (np.array): Input data.

        Returns:
        np.array: Output of the layer.
        """
        pass
    
    
    @abstractmethod
    def backward(self, dl_da: np.array) -> np.array:
        """
        Computes the gradient of the loss with respect to the layer's input.
        This abstract method must be implemented by all subclasses.

        Parameters:
        dl_da (np.array): Gradient of the loss with respect to the layer's output.

        Returns:
        np.array: Gradient of the loss with respect to the layer's input.
        """
        pass
    
    
    @abstractmethod
    def get_weights(self) -> dict:
        """
        Retrieves the layer's weights and biases in a dictionary.
        This abstract method must be implemented by all subclasses.

        Returns:
        dict: Dictionary containing the layer's weights and biases.
        """
        pass
    
    
    @abstractmethod
    def set_weights(self, weights: dict) -> None:
        """
        Sets the layer's weights and biases from a provided dictionary.
        This abstract method must be implemented by all subclasses.

        Parameters:
        weights (dict): Dictionary containing the layer's weights and biases to be set.
        """
        pass
    
    
    @classmethod
    def get_config(self) -> dict:
        """
        Retrieves a dictionary containing the configuration of the layer.
        This method must be implemented by all subclasses to facilitate saving and loading models.

        Returns:
        dict: Dictionary containing the configuration of the layer.
        """
        return {}
