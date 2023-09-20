from skynet_ml.nn.layers.layer import Layer
import numpy as np

class Dense(Layer):
    """
    A Dense (fully connected) layer for a neural network.
    
    A dense layer is a layer where every neuron receives input from every element of the previous layer. It's 
    the most basic and common type of layer used in neural networks. This layer performs a linear transformation 
    on the input data by using weights and biases. The result can then be passed through an activation function 
    to introduce non-linearity to the model.

    Attributes
    ----------
    input_vector : np.array
        The input data for the layer during the forward pass.
    z : np.array
        The weighted input (linear transformation of the input data).
    a : np.array
        The activated output after applying the activation function.
    weights : np.array
        The weights for the layer.
    biases : np.array
        The biases for the layer.
    delta : np.array
        The delta value used in the backward pass.
    dweights : np.array
        The partial derivative of the loss with respect to the weights.
    dbiases : np.array
        The partial derivative of the loss with respect to the biases.
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
    forward(x)
        Computes the forward pass.
    backward(dLda)
        Computes the backward pass, returning the gradient of the loss with respect to the input of this layer.
    _calculate_delta(dLda)
        Computes the delta for this layer (partial derivative of loss with respect to z).
    get_weights()
        Returns the weights and biases of this layer.
    set_weights(weights)
        Sets the weights and biases of this layer.
    get_config()
        Returns the configuration of this layer.
    """
    
    def forward(self, x: np.array) -> np.array:
        """
        Computes the forward pass for the dense layer.

        Parameters
        ----------
        x : np.array
            Input data for the layer.

        Returns
        -------
        np.array
            Activated output after applying the activation function.
        """
        self.input_vector = x
        self.z = np.dot(x, self.weights) + self.biases # compute weighted input (Wx + b)
        self.a = self.activation.compute(self.z) # compute activated output (a = g(z))
        
        return self.a 
    
    
    def backward(self, dLda: np.array) -> np.array:
        """
        Computes the backward pass for the dense layer.

        Parameters
        ----------
        dLda : np.array
            Gradient of the loss with respect to the output of this layer.

        Returns
        -------
        np.array
            Gradient of the loss with respect to the input of this layer.
        """
        self.delta = self._calculate_delta(dLda) # compute delta for this layer (partial derivative of loss with respect to z)
        self.dweights = np.dot(self.input_vector.T, self.delta)  # compute partial derivative of loss with respect to weights
        self.dbiases = np.sum(self.delta, axis=0, keepdims=True) if self.has_bias else None # compute partial derivative of loss with respect to biases
        
        dLda_prev = np.dot(self.delta, self.weights.T) # compute partial derivative of loss with respect the activation of previous layer (dl/da for the previous layer)
        return dLda_prev
    
    
    def _calculate_delta(self, dLda: np.array) -> np.array:
        """
        Calculates the delta value used in the backward pass.

        This method computes the product of the gradient of the loss with respect to the activated output and 
        the gradient of the activation function with respect to the weighted input.

        Parameters
        ----------
        dLda : np.array
            Gradient of the loss with respect to the output of this layer.

        Returns
        -------
        np.array
            Delta value for the layer.
        """
        dadz = self.activation.gradient(self.z) # gradient of activation function w.r.t. z
        
        # check shape of da_dz, if it's 3D then we have a batch of matrices, a.k.a. activation is softmax and optimize_gradient is False
        if len(dadz.shape) == 3:
            return np.einsum('ijk,ik->ij', dLda, dadz) # in this case we need to use einsum to multiply the matrices element-wise
        
        return np.multiply(dLda, dadz) # otherwise we can use regular element-wise multiplication
    
    
    def get_weights(self) -> dict:
        """
        Retrieves the weights and biases of the dense layer.

        Returns
        -------
        dict
            Dictionary containing the weights and biases of the layer.
        """
        return {'weights': self.weights, 'biases': self.biases}
    
    
    def set_weights(self, weights: dict) -> None:
        """
        Sets the weights and biases for the dense layer.

        Parameters
        ----------
        weights : dict
            Dictionary containing the new weights and biases for the layer.
        """
        self.weights = weights['weights']
        self.biases = weights['biases']
        
        
    def get_config(self) -> dict:
        """
        Retrieves the configuration of the dense layer.

        Returns
        -------
        dict
            Dictionary containing the configuration attributes of the layer.
        """
        return {
            'name': "Dense",
            "units": self.n_units,
            "activation": self.activation, 
            "initializer": self.initializer,
            "input_dim": self.input_dim,
            "has_bias": self.has_bias,
            "activation_name": self.activation.name,
            "initialize_name": self.initializer.name,
        }
