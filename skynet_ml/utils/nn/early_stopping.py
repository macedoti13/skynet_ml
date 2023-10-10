class EarlyStopping:
    """
    Implements the early stopping mechanism to prevent overfitting during training.

    Attributes
    ----------
    patience : int
        Number of epochs with no improvement after which training will be stopped.
    min_delta : float
        Minimum change in the monitored quantity to qualify as an improvement.
    wait : int
        Number of epochs that have been waited without improvement.
    best_loss : float
        Best observed loss value.

    Methods
    -------
    should_stop(current_loss: float) -> bool:
        Checks if early stopping conditions are met based on the provided loss value.
    """
    
    
    def __init__(self, patience: int = 10, min_delta: float = 0.00001):
        """
        Initializes the early stopping object with the given patience and delta values.

        Parameters
        ----------
        patience : int, optional
            Number of epochs with no improvement after which training will be stopped.
            Default is 5.
        min_delta : float, optional
            Minimum change in the monitored quantity to qualify as an improvement.
            Default is 0.001.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = float('inf')


    def should_stop(self, current_loss: float) -> bool:
        """
        Checks if early stopping conditions are met based on the provided loss value.

        Parameters
        ----------
        current_loss : float
            The current loss value to be compared with the best observed loss.

        Returns
        -------
        bool
            True if the training should be stopped early, otherwise False.
        """
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            return True
        return False
