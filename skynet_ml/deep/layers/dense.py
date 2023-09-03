import numpy as np
from skynet_ml.deep.initializers import initializers_map
from skynet_ml.deep.layers.activations import activations_map, d_activations_map

class Dense:
    """
    Represents a fully connected (dense) neural network layer. Each neuron in this layer is connected to every neuron in the previous layer.
    
    Attributes:
        weights (np.array): Weights matrix of the layer.
        bias (np.array): Bias vector for the layer. Set to zeros if `has_bias=False`.
        activation (function): Activation function for neurons.
        d_activation (function): Derivative of the activation function.
        has_bias (bool): Indicator for including bias.
    """       
    
    
    def __init__(self, input_dim: int, output_dim: int, activation: str=None, has_bias: bool=True, initializer="random"):
        """
        Initializes the Dense layer.

        Args:
            input_dim (int): Number of input features or neurons from the previous layer.
            output_dim (int): Number of neurons in this layer.
            activation (str, optional): Activation function's name. Defaults to None.
            has_bias (bool, optional): Indicator for bias units. Defaults to True.
            initializer (str, optional): Method to initialize weights and biases. Defaults to "random".
        """   
        
        # asserts
        assert isinstance(input_dim, int) and input_dim > 0, "Input dimension must be a positive integer."
        assert isinstance(output_dim, int) and output_dim > 0, "Output dimension must be a positive integer."
        assert initializer in initializers_map or hasattr(initializer, "initialize_weights"), "Invalid initializer type."

        # get's the initializer object, either directly or from string
        if isinstance(initializer, str):
            self.initializer = initializers_map[initializer]
        else:
            self.initializer = initializer
            
        # initializes the weights and biases with the initializer
        self.weights = self.initializer.initialize_weights(input_dim, output_dim)
        self.bias = self.initializer.initialize_bias(output_dim, has_bias)
        self.has_bias = has_bias
        
        # initializes the activation function and it's derivative
        self.activation = activations_map[activation]
        self.d_activation = d_activations_map[activation]
        
        
    def forward(self, input_vector: np.array) -> np.array:
        """
        Computes the forward pass through the layer. 

        Args:
            input_vector (np.array): Inputs for the layer.

        Returns:
            np.array: Activations after linear transformation and activation function.
        """      
        
        # assert vector shapes are correct
        assert input_vector.shape[0] == self.weights.shape[1], "Input shape mismatch. Expected shape: ({},) but got: {}.".format(self.weights.shape[1], input_vector.shape)
  
        # saves the input vector for backprop
        self.input_vector = input_vector
        
        # linear transformation (Wx + b)
        self.z = np.dot(self.weights, self.input_vector) + self.bias
        
        # calculates the activation(Wx + b)
        self.a = self.activation(self.z)
        
        return self.a
    
    
    def backward(self, d_output: np.array) -> np.array:
        """
        Computes the gradients using backpropagation. It receives d_output and uses it to calculate it's delta. 
        Then it uses the delta to calculate the gradient w.r.t the weights and biases. In the end, calculates 
        the first term of the delta for the next layer (d_input).

        Args:
            d_output (np.array): Gradient of the loss with respect to the outputs of this layer. It's the first term of the delta for this layer.

        Returns:
            np.array: Gradient w.r.t. layer inputs.
        """        
        
        # asserts 
        assert hasattr(self, 'z'), "Forward method must be called before backward."
        assert d_output.shape == self.a.shape, "d_output shape mismatch. Expected shape: {} but got: {}.".format(self.a.shape, d_output.shape)

        # calculates the delta for this layer: 
        self.delta = np.multiply(d_output, self.d_activation(self.z))
        
        # computes the gradient of the loss w.r.t weights
        self.d_weights = np.dot(self.delta, self.input_vector.T)
        
        # computes the gradient of the loss w.r.t biases (if present)
        self.d_bias = np.sum(self.delta, axis=1, keepdims=True) if self.has_bias else None
        
        # computes the first term of the delta for the layer that comes before, gradient w.r.t layer inputs
        d_input = np.dot(self.weights.T, self.delta)
        
        return d_input
    
    
    def initialize_momentum(self):
        """
        Initializes the momentum values for the layer's weights and biases.
        
        For optimization techniques that leverage momentum (e.g., SGD with momentum),
        this method ensures that the initial momentum values are set to zeros 
        with the same shape as the layer's weights and biases.
        """    
        # asserts 
        assert hasattr(self, 'weights'), "Weights must be initialized before initializing momentum or velocity."
        assert hasattr(self, 'bias'), "Bias must be initialized before initializing momentum or velocity."

        self.m_weights = np.zeros_like(self.weights)
        if self.has_bias:
            self.m_bias = np.zeros_like(self.bias)
            
            
    def initialize_velocity(self):
        """
        Initializes the velocity values for the layer's weights and biases.
        
        For optimization techniques that use adaptive learning rates or 
        second-order information (e.g., Adam), this method ensures that the 
        initial velocity or moving average of past squared gradients values 
        are set to zeros with the same shape as the layer's weights and biases.
        """   
        # asserts
        assert hasattr(self, 'weights'), "Weights must be initialized before initializing momentum or velocity."
        assert hasattr(self, 'bias'), "Bias must be initialized before initializing momentum or velocity."

        self.v_weights = np.zeros_like(self.weights)
        if self.has_bias:
            self.v_bias = np.zeros_like(self.bias)
