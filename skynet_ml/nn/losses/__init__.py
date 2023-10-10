from skynet_ml.nn.losses.bce import BinaryCrossEntropy
from skynet_ml.nn.losses.cce import CategoricalCrossEntropy
from skynet_ml.nn.losses.mse import MeanSquaredError


LOSSES_MAP = {
    "binary_crossentropy": BinaryCrossEntropy,
    "bce": BinaryCrossEntropy,
    "categorical_crossentropy": CategoricalCrossEntropy,
    "cce": CategoricalCrossEntropy,
    "mean_squared_error": MeanSquaredError,
    "mse": MeanSquaredError
}
