from skynet_ml.nn.layers.base import BaseLayer
import numpy as np


class Dense(BaseLayer):
    """
    A dense (fully connected) layer for neural networks.
    
    This layer implements the operation: output = activation(dot(input, weights) + bias), where 
    `dot` is the dot product, `weights` are the learned weight matrix, and `bias` is the learned bias vector.

    Attributes:
        n_units (int): Number of neurons (units) in the layer.
        activation (BaseActivation): Activation function applied to the output.
        initializer (BaseInitializer): Weight and bias initialization strategy.
        regularizer (BaseRegularizer, optional): Regularizer function applied to the weights matrix (default is None).
        has_bias (bool, optional): Whether to include a bias vector in the layer computation (default is True).
        input_dim (int, optional): Number of input features to the layer (default is None).
    """
    
    
    def forward(self, x: np.array) -> np.array:
        """
        Compute the forward pass for the dense layer.
        
        Args:
            x (np.array): Input data to the layer.
            
        Returns:
            np.array: The output of the dense layer after applying weights, biases, and activation function.
        """

        self.input_vector = x
        
        # Compute the linear combination of the input vector and the weights
        self.z = np.dot(x, self.weights) + self.bias
        
        # Compute the activation of the linear combination 
        self.a = self.activation.compute(self.z) 
        
        return self.a
    
    
    def backward(self, dl_da: np.array) -> np.array:
        """
        Compute the backward pass for the dense layer.
        
        Args:
            dl_da (np.array): Gradient of the loss with respect to the output of the layer.
            
        Returns:
            np.array: Gradient of the loss with respect to the input of the layer.
        """

        # Compute the delta of the layer: partial derivative of the loss with respect to the linear combination
        da_dz = self.activation.gradient(self.z)
        self.delta = np.multiply(dl_da, da_dz)
        
        # Compute the partial derivative of the loss with respect to the weights
        self.d_weights = np.dot(self.input_vector.T, self.delta) 
        
        # Compute the partial derivative of the loss with respect to the biases
        self.d_bias = np.sum(self.delta, axis=0, keepdims=True) if self.has_bias else None 
        
        # Compute the partial derivative of the loss with respect to the activation of previous layer (dl_da of previous layer)
        dl_da_previous = np.dot(self.delta, self.weights.T) 
        
        return dl_da_previous
    
    
    def get_weights(self) -> dict:
        """
        Retrieve the weights and biases of the dense layer.
        
        Returns:
            dict: A dictionary containing the weights ('weights' key) and biases ('bias' key) of the dense layer.
        """
        return {"weights": self.weights, "bias": self.bias}
    
    
    def set_weights(self, weights: dict) -> None:
        """
        Set the weights and biases of the dense layer.
        
        Args:
            weights (dict): A dictionary containing the weights ('weights' key) and biases ('bias' key) to be set for the dense layer.
        """
        self.weights = weights["weights"]
        self.bias = weights["bias"]
    
    
    def get_config(self) -> dict:
        """
        Get the configuration of the dense layer.
        
        Returns:
            dict: A dictionary containing the configuration parameters of the dense layer.
        """
        
        return {
            'name': "Dense",
            "n_units": self.n_units,
            "activation": self.activation.name, 
            "initializer": self.initializer.name,
            "regularizer": self.regularizer.name if self.regularizer is not None else None,
            "input_dim": self.input_dim,
            "has_bias": self.has_bias,
        }
