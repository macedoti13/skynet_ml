from skynet_ml.nn.layers.layer import Layer
import numpy as np


class Dense(Layer):
    """
    Fully connected neural network layer (Dense Layer).

    The Dense layer is a standard layer type that is used in many types of neural networks. It's a fully connected layer,
    meaning it's “densely connected.” Each neuron in a Dense layer receives input from all neurons of the previous layer.

    Attributes:
    n_units (int): Number of units (neurons) in the layer.
    activation (str or Activation): Activation function applied to the layer. Can be either a string identifier or an Activation instance.
    initializer (str or Initializer, optional): Weight initializer for the layer. Can be either a string identifier or an Initializer instance.
    regularizer (str or Regularizer, optional): Regularizer function applied to the layer's weights. Can be either a string identifier or a Regularizer instance.
    has_bias (bool, optional): Determines whether the layer uses bias. Default is True.
    input_dim (int, optional): Dimensionality of the input space.

    Methods:
    - forward(x: np.array) -> np.array: Computes and returns the output of the Dense layer for a given input.
    - backward(dl_da: np.array) -> np.array: Computes and returns the gradient of the loss with respect to the layer's input.
    - calculate_delta(dl_da: np.array) -> np.array: Computes and returns the delta of the layer, used during the backward pass.
    - get_weights() -> dict: Retrieves the layer's weights and biases in a dictionary.
    - set_weights(weights: dict) -> None: Sets the layer's weights and biases from a provided dictionary.
    - get_config() -> dict: Retrieves a dictionary containing the configuration of the Dense layer.
    """
    
    def forward(self, x: np.array) -> np.array:
        """
        Computes and returns the output of the Dense layer for a given input.

        Parameters:
        x (np.array): Input data of shape (batch_size, input_dim).

        Returns:
        np.array: Output of the layer of shape (batch_size, n_units).
        """
        self.input_vector = x
        self.z = np.dot(x, self.weights) + self.bias # Compute the linear combination of the input vector and the weights
        self.a = self.activation.compute(self.z) # Compute the activation of the linear combination
        
        return self.a
    
    
    def backward(self, dl_da: np.array) -> np.array:
        """
        Computes the gradient of the loss with respect to the layer's input.
        
        Parameters:
        dl_da (np.array): Gradient of the loss with respect to the layer's output.

        Returns:
        np.array: Gradient of the loss with respect to the layer's input.
        """
        self.delta = self.calculate_delta(dl_da) # Compute the delta of the layer: partial derivative of the loss with respect to the linear combination
        self.d_weights = np.dot(self.input_vector.T, self.delta) # Compute the partial derivative of the loss with respect to the weights
        self.d_bias = np.sum(self.delta, axis=0, keepdims=True) if self.has_bias else None # Compute the partial derivative of the loss with respect to the biases
        
        dl_da_previous = np.dot(self.delta, self.weights.T) # Compute the partial derivative of the loss with respect to the activation of previous layer (dl_da of previous layer)
        return dl_da_previous
    
    
    def calculate_delta(self, dl_da: np.array) -> np.array:
        """
        Computes the delta of the layer, which is used during the backward pass.
        
        Parameters:
        dl_da (np.array): Gradient of the loss with respect to the layer's output.

        Returns:
        np.array: The delta of the layer.
        """
        da_dz = self.activation.gradient(self.z)
        return np.multiply(dl_da, da_dz)
    
    
    def get_weights(self) -> dict:
        """
        Retrieves the Dense layer's weights and biases in a dictionary.
        
        Returns:
        dict: Dictionary containing the layer's weights and biases.
        """
        return {"weights": self.weights, "bias": self.bias}
    
    
    def set_weights(self, weights: dict) -> None:
        """
        Sets the Dense layer's weights and biases from a provided dictionary.

        Parameters:
        weights (dict): Dictionary containing the layer's weights and biases to be set.
        """
        self.weights = weights["weights"]
        self.bias = weights["bias"]
    
    
    def get_config(self) -> dict:
        """
        Retrieves a dictionary containing the configuration of the Dense layer.

        Returns:
        dict: Dictionary containing the configuration of the layer, including name, number of units, activation function, etc.
        """
        return {
            'name': "Dense",
            "n_units": self.n_units,
            "activation": self.activation.name, 
            "initializer": self.initializer.name,
            "input_dim": self.input_dim,
            "has_bias": self.has_bias,
        }
