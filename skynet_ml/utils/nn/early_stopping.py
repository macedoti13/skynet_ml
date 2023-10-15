class EarlyStopping:
    """
    Early stopping utility.

    This utility checks for improvement in a given metric (usually validation loss) and stops the training process if 
    no improvement is observed for a specified number of iterations (patience). It is often used to prevent overfitting 
    and potentially save computational resources.

    Attributes:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        wait (int): Current number of epochs with no improvement.
        best_loss (float): The best (lowest) value of the loss observed so far.
    """


    def __init__(self, patience: int = 10, min_delta: float = 0.00001):
        """
        Initialize the EarlyStopping utility.

        Args:
            patience (int, optional): Number of epochs with no improvement after which training will be stopped. 
                                      Defaults to 10.
            min_delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. 
                                         Defaults to 0.00001.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = float('inf')


    def should_stop(self, current_loss: float) -> bool:
        """
        Check if the training process should be halted based on the current loss.

        Args:
            current_loss (float): The current value of the loss.

        Returns:
            bool: True if the training process should be halted, False otherwise.
        """
        
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            return True
        
        return False
