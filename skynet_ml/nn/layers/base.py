from skynet_ml.utils.factories import ActivationsFactory, RegularizersFactory, InitializersFactory
from skynet_ml.nn.initializers import activations_initializers_map
from skynet_ml.nn.regularizers.base import BaseRegularizer
from skynet_ml.nn.initializers.base import BaseInitializer
from skynet_ml.nn.activations.base import BaseActivation
from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np


class BaseLayer(ABC):
    """
    Abstract class for layers in a neural network.
    
    This class serves as a blueprint for other layer classes, offering a base structure and functionality
    that is common to all layers.
    
    Args:
        n_units (int): Number of units in the layer.
        activation (Union[str, BaseActivation], optional): Activation function to be used in the layer. Can be
                                                            given as a string or a BaseActivation object. Defaults to 'linear'.
        initializer (Union[str, BaseInitializer], optional): Weights initializer for the layer. Can be
                                                            given as a string or a BaseInitializer object.
                                                            Defaults to an initializer mapped to the activation function.
        regularizer (Union[str, BaseRegularizer], optional): Regularizer for the layer. Can be given as a
                                                            string or a BaseRegularizer object. Defaults to None.
        has_bias (bool, optional): Whether the layer has a bias term. Defaults to True.
        input_dim (int, optional): Number of input features for the layer. To be set for the first layer in a model. Defaults to None.

    Attributes:
        n_units (int): Number of units in the layer.
        activation (BaseActivation): Activation function object for the layer.
        initializer (BaseInitializer): Weights initializer object for the layer.
        regularizer (BaseRegularizer or None): Regularizer object for the layer.
        has_bias (bool): Indicates if the layer has a bias term.
        input_dim (int or None): Number of input features for the layer.
        is_initialized (bool): Flag indicating if the layer's weights and biases are initialized.
        weights (np.array): Weights matrix of the layer.
        bias (np.array): Bias vector of the layer.
    """


    def __init__(self, 
                 n_units: int, 
                 activation: Optional[Union[str, BaseActivation]] = "linear",
                 initializer: Optional[Union[str, BaseInitializer]] = None,
                 regularizer: Optional[Union[str, BaseRegularizer]] = None,
                 has_bias: Optional[bool] = True,
                 input_dim: Optional[int] = None,
                ) -> None:
        """
        Initializes the BaseLayer class.
        """        
        
        self.n_units = n_units
        self.activation = ActivationsFactory().get_object(activation)
        
        # If initializer is not given, use the initializer mapped to the activation function
        if initializer is None:
            initializer = activations_initializers_map[self.activation.name]
            
        self.initializer = InitializersFactory().get_object(initializer)
        self.regularizer = RegularizersFactory().get_object(regularizer) if regularizer is not None else None
        self.has_bias = has_bias
        self.input_dim = input_dim
        self.is_initialized = False
        
        
    def initialize(self) -> None:
        """
        Initialize the layer's weights and biases.
        
        This method uses the provided initializer to set the initial values of the layer's weights and biases.
        It checks if the layer's parameters are already initialized to avoid reinitialization.
        """
        if not self.is_initialized:
            self.weights = self.initializer.initialize_weights(self.input_dim, self.n_units)
            self.bias = self.initializer.initialize_bias(self.n_units)
            self.is_initialized = True
            
            
    def initialize_momentum(self) -> None:
        """
        Initialize momentum for the layer's weights and biases.
        
        This is particularly useful for certain optimization algorithms like Momentum and Nesterov Accelerated Gradient.
        Initializes the momentum to a matrix of zeros with the same shape as the layer's weights and biases.
        """
        self.m_weights = np.zeros_like(self.weights)
        if self.has_bias:
            self.m_bias = np.zeros_like(self.bias)
            
            
    def initialize_velocity(self) -> None:
        """
        Initialize velocity for the layer's weights and biases.
        
        This is useful for optimizers like Adam which require a velocity term.
        Initializes the velocity to a matrix of zeros with the same shape as the layer's weights and biases.
        """
        self.v_weights = np.zeros_like(self.weights)
        if self.has_bias:
            self.v_bias = np.zeros_like(self.bias)
            
            
    @abstractmethod
    def forward(self, x: np.array) -> np.array:
        """
        Forward pass computation for the layer.
        
        Args:
            x (np.array): Input data to the layer.
            
        Returns:
            np.array: The output of the layer after applying weights, biases, and activation function.
        """
        pass
    
    
    @abstractmethod
    def backward(self, dl_da: np.array) -> np.array:
        """
        Backward pass computation for the layer.
        
        Args:
            dl_da (np.array): Gradient of the loss with respect to the output of the layer.
            
        Returns:
            np.array: Gradient of the loss with respect to the input of the layer.
        """
        pass
    
    
    @abstractmethod
    def get_weights(self) -> dict:
        """
        Retrieve the weights and biases of the layer.
        
        Returns:
            dict: A dictionary containing the weights ('weights' key) and biases ('bias' key) of the layer.
        """
        pass
    
    
    @abstractmethod
    def set_weights(self, weights: dict) -> None:
        """
        Set the weights and biases of the layer.
        
        Args:
            weights (dict): A dictionary containing the weights ('weights' key) and biases ('bias' key) to be set for the layer.
        """
        pass
