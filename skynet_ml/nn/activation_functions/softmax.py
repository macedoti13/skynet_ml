from skynet_ml.nn.activation_functions.activation import Activation
import numpy as np

class SoftmaxActivation(Activation):
    """
    Implementation of the Softmax activation function for neural networks.

    The Softmax function is primarily used in the output layer of a neural network 
    for multi-class classification problems. It returns the probabilities of each class, 
    given an input, ensuring that the probabilities sum up to 1.

    When combined with Cross Entropy loss, the derivative simplifies to (yhat - y), and 
    there's no need to compute the Jacobian of the Softmax. This optimization is incorporated 
    in the class if `with_cross_entropy` is set to True.

    Attributes
    ----------
    with_cross_entropy : bool
        Indicates whether the Softmax activation is combined with Cross Entropy loss. 
        If True, the derivative simplifies to a vector of ones.

    Methods
    -------
    compute(z: np.array) -> np.array:
        Compute the Softmax activation function for a given input.

    gradient(z: np.array) -> np.array:
        Compute the derivative of the Softmax activation function or returns a vector 
        of ones if combined with Cross Entropy.

    _compute_jacobian_for_sample(zi: np.array) -> np.array:
        Compute the Jacobian matrix of the Softmax activation for a single sample.

    _compute_jacobian_batch(z: np.array) -> np.array:
        Compute the Jacobian matrices of the Softmax activation for a batch of samples.

    """
    
    def __init__(self, with_cross_entropy: bool = True) -> None:
        """
        Initialize the SoftmaxActivation class with an optional flag to indicate the use 
        in combination with Cross Entropy loss.

        Parameters
        ----------
        with_cross_entropy : bool, optional
            If True (default), the gradient function will return a vector of ones. 
            This is due to the simplification of the derivative when combined with 
            Cross Entropy loss.
        """
        self.with_cross_entropy = with_cross_entropy
        self.name = "Softmax"
        
        
    def compute(self, z: np.array) -> np.array:
        """
        Compute the Softmax activation function for a given input.

        Parameters
        ----------
        z : np.array
            Input data, typically the weighted sum of inputs and weights.

        Returns
        -------
        np.array
            Output probabilities after applying the Softmax activation function.

        """
        self._check_shape(z)
        exps = np.exp(z - np.max(z, axis=1, keepdims=True)) # for numerical stability
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    
    def gradient(self, z: np.array) -> np.array:
        """
        Compute the derivative of the Softmax activation function. If used in conjunction 
        with Cross Entropy loss (`with_cross_entropy=True`), then this method will return 
        a vector of ones due to the simplification of the Softmax and Cross Entropy gradient.

        Parameters
        ----------
        z : np.array
            Input data, typically the weighted sum of inputs and weights.

        Returns
        -------
        np.array
            Gradient of the Softmax activation function or a vector of ones if combined 
            with Cross Entropy.

        """
        self._check_shape(z)
        if self.with_cross_entropy:
            return np.ones_like(z) # If combined with cross entropy, uses yhat - y simplification, that is all done in CrossEntropyLoss class. 
        else:
            return self._compute_jacobian_batch(z)
    
        
    def _compute_jacobian_for_sample(self, zi: np.array) -> np.array:
        """
        Compute the Jacobian matrix of the Softmax activation for a single sample.
        
        The Jacobian matrix is the matrix of all first-order partial derivatives of 
        the Softmax function. Each element (i, j) represents the derivative of 
        the ith output with respect to the jth input.

        Parameters
        ----------
        zi : np.array
            Input data for a single sample. Expected shape is (1, input_dim).

        Returns
        -------
        np.array
            Jacobian matrix of shape (input_dim, input_dim) representing the 
            derivatives of the Softmax output with respect to its input.

        Raises
        ------
        ValueError
            If the shape of zi is not (1, input_dim).
        """
        # Ensure zi is of shape (1, input_dim)
        if len(zi.shape) != 2 or zi.shape[0] != 1:
            raise ValueError(f"Expected zi to be of shape (1, input_dim), got shape {zi.shape} instead.")
        
        softmax = self.compute(zi).ravel()  # Flatten the softmax to shape (n_inputs,)
        jacobian = -np.outer(softmax, softmax) + np.diag(softmax * (1 - softmax))
        
        # Ensuring the Jacobian is of shape (n_inputs, n_inputs)
        if jacobian.shape != (zi.shape[1], zi.shape[1]):
            raise ValueError(f"Expected Jacobian to be of shape ({zi.shape[1]}, {zi.shape[1]}), got {jacobian.shape} instead.")

        return jacobian
    
        
    def _compute_jacobian_batch(self, z: np.array) -> np.array:
        """
        Compute the Jacobian matrices of the Softmax activation for a batch of samples.

        For each sample in the batch, this method computes the Jacobian matrix and 
        returns them stacked together. This is useful when backpropagating errors 
        through a batch of data in neural networks.

        Parameters
        ----------
        z : np.array
            Input data for a batch of samples. Expected shape is (batch_size, input_dim).

        Returns
        -------
        np.array
            Batch of Jacobian matrices. Each matrix is of shape (input_dim, input_dim), 
            and the returned array has shape (batch_size, input_dim, input_dim).

        Raises
        ------
        ValueError
            If the shape of the resultant Jacobian matrices does not match expectations.
        """
        jacobians = [] # List of jacobians for each sample in z
        
        for i in z:
            jacobians.append(self._compute_jacobian_for_sample(i.reshape(1, -1))) # compute the jacobian for each sample in z and append to the list

        jacobians = np.array(jacobians) # Convert to numpy array
            
        # Validate the shape
        if jacobians.shape != (z.shape[0], z.shape[1], z.shape[1]):
            raise ValueError(f"Expected shape ({z.shape[0]}, {z.shape[1]}, {z.shape[1]}), but got {jacobians.shape}")
        
        return jacobians
