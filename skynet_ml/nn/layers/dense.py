import numpy as np
from skynet_ml.nn.initializers import initialize_weights, initialize_bias
from skynet_ml.nn.activations import get_activation_function, get_activation_derivative
from skynet_ml.nn.optimizers import get_optimizer

class Dense:
    """
    Represents a fully connected (dense) neural network layer. Each neuron in this layer is connected to every neuron in the previous layer.
    
    Attributes:
        weights (np.array): Weights matrix of the layer.
        bias (np.array): Bias vector for the layer. Set to zeros if `has_bias=False`.
        activation (function): Activation function for neurons.
        activation_derivative (function): Derivative of the activation function.
        has_bias (bool): Indicator for including bias.
    """    


    def __init__(self, input_size: int, output_size: int, activation: str = "sigmoid", has_bias: bool = True, initialization: str = "random") -> None:
        """
        Initializes the Dense layer.

        Args:
            input_size (int): Number of input features or neurons from the previous layer.
            output_size (int): Number of neurons in this layer.
            activation (str, optional): Activation function's name. Defaults to "sigmoid".
            has_bias (bool, optional): Indicator for bias units. Defaults to True.
            initialization (str, optional): Method to initialize weights and biases. Defaults to "random".
        """
        self.weights = initialize_weights(initialization, input_size, output_size)
        self.bias = initialize_bias(has_bias, initialization, output_size)
        self.activation = get_activation_function(activation)
        self.activation_derivative = get_activation_derivative(activation)
        self.has_bias = has_bias


    def forward(self, input_vector: np.array) -> np.array:
        """
        Computes the forward pass through the layer.

        Args:
            input_vector (np.array): Layer inputs.

        Returns:
            np.array: Activations after linear transformation and activation function.
        """
        # Store input for backpropagation
        self.input_vector = input_vector 
        # Linear transformation (Wx + b)
        self.z = np.dot(self.weights, self.input_vector) + self.bias
        # Activation(Wx + b)
        self.a = self.activation(self.z)
        
        return self.a


    def backward(self, d_output: np.array) -> np.array:
        """
        Computes the gradients using backpropagation.

        Args:
            d_output (np.array): Gradient of the loss with respect to the outputs of this layer. It's the first term of the delta for this layer.

        Returns:
            np.array: Gradient w.r.t. layer inputs.
        """
        # Calculate delta: combining upstream gradient and gradient of the activation (d_L/d_a * d_a/d_z)
        self.delta = np.multiply(d_output, self.activation_derivative(self.z)) 
        # Gradient w.r.t. weights
        self.d_weights = np.dot(self.delta, self.input_vector.T)
        # Gradient w.r.t. bias (if present)
        self.d_bias = np.sum(self.delta, axis=1, keepdims=True) if self.has_bias else None
        # Gradient w.r.t. layer inputs for backpropagation to the previous layer (d_L/d_a for next layer)
        d_input = np.dot(self.weights.T, self.delta)
        
        return d_input


    def update_parameters(self, optimizer: str = "sgd", learning_rate: float = 0.01) -> None:
        """
        Updates parameters (weights and biases) using the provided optimizer.

        Args:
            optimizer (str, optional): Optimization method. Defaults to "sgd".
            learning_rate (float, optional): Step size for optimizer. Defaults to 0.01.
        """
        # Fetch optimizer function
        optimizer_function = get_optimizer(optimizer)
        
        # Update weights
        self.weights = optimizer_function(self.weights, self.d_weights, learning_rate)
        # Update bias if present
        if self.has_bias:
            self.bias = optimizer_function(self.bias, self.d_bias, learning_rate)
