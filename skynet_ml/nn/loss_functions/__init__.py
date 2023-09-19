from skynet_ml.nn.loss_functions.bce import BinaryCrossEntropyLoss
from skynet_ml.nn.loss_functions.mse import MeanSquaredErrorLoss
from skynet_ml.nn.loss_functions.cce import CrossEntropyLoss

LOSS_MAP = {
    "mean_squared_error": MeanSquaredErrorLoss,
    "mse": MeanSquaredErrorLoss,  # Alias for "mean_squared_error
    "binary_cross_entropy": BinaryCrossEntropyLoss,
    "bce": BinaryCrossEntropyLoss,  # Alias for "binary_cross_entropy
    "cross_entropy": CrossEntropyLoss,
    "cce": CrossEntropyLoss,  # Alias for "cross_entropy
    "categorical_crossentropy": CrossEntropyLoss,  # Alias for "cross_entropy
}